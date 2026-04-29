# -*- coding: utf-8 -*-
"""env.py
NetworkTrafficEnv with revised lower neighbor policy input and edge-level upper observation.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

from utils import build_sumo_cmd, Node
from xml_mapping import load_network_mapping
from emission_lookup import EmissionFactorLookup


@dataclass
class StaticNetwork:
    mapping_dict: Dict[str, Any]
    network_graph: Any
    edge_graph: Any
    tls_ids: List[str]
    tls_neighbors: Dict[str, Dict[str, str]]
    adj_matrix: np.ndarray
    edge_ids: List[str]
    A_edge_up: np.ndarray
    A_edge_down: np.ndarray
    edge_to_tls_matrix: np.ndarray
    edge_lanes: Dict[str, List[str]]
    edge_capacity: Dict[str, float]


class NetworkTrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config, port: int = 8000, output_path: str = ""):
        super().__init__()
        self.config = config
        self.port = int(port)
        self.output_path = output_path
        self.is_gui = bool(config.SUMO_GUI)
        self.sim_step = float(config.SIM_STEP)
        self.delta_t = float(config.DELTA_T)
        self.yellow_time = float(config.YELLOW_TIME)
        self.max_steps = int(config.MAX_STEPS)
        self.current_step = 0
        self.simulation_time = 0.0
        self.sumo_cmd = build_sumo_cmd(self.config, gui=self.is_gui)

        network_bundle = load_network_mapping(config=self.config, output_dir=None, print_summary=False, generate_csv=False)
        eg = network_bundle.edge_graph
        self.static = StaticNetwork(
            mapping_dict=network_bundle.mapping_dict,
            network_graph=network_bundle.network_graph,
            edge_graph=eg,
            tls_ids=list(network_bundle.network_graph.tls_ids),
            tls_neighbors=dict(network_bundle.network_graph.tls_neighbors),
            adj_matrix=np.asarray(network_bundle.network_graph.adj_matrix, dtype=np.float32),
            edge_ids=list(eg.edge_ids),
            A_edge_up=np.asarray(eg.A_edge_up, dtype=np.float32),
            A_edge_down=np.asarray(eg.A_edge_down, dtype=np.float32),
            edge_to_tls_matrix=np.asarray(eg.edge_to_tls_matrix, dtype=np.float32),
            edge_lanes=dict(eg.edge_lanes),
            edge_capacity=dict(eg.edge_capacity),
        )
        self.config.NUM_EDGES = len(self.static.edge_ids)

        shared_lookup = EmissionFactorLookup(
            csv_path=self.config.EMISSION_FACTOR_FILE,
            default_pollutant=self.config.EMISSION_POLLUTANT,
            default_vehicle_type=self.config.DEFAULT_VEHICLE_TYPE,
            enable_cache=self.config.EMISSION_ENABLE_CACHE,
        )
        self.shared_emission_lookup = shared_lookup
        self.nodes: Dict[str, Node] = {
            tl_id: Node(name=tl_id, config=self.config, lane_mapping=self.static.mapping_dict[tl_id], emission_lookup=shared_lookup)
            for tl_id in self.static.tls_ids
        }
        self._vehicle_type_cache: Dict[str, str] = {}
        self.upper_weights = np.full((len(self.static.tls_ids), 2), 0.5, dtype=np.float32)
        self.last_upper_metrics = {"E_net": 0.0, "Q_net": 0.0, "H_net": 0.0, "B_net": 0.0, "E_sum": 0.0, "Q_sum": 0.0, "P_hot": 0.0}
        self.last_edge_metrics: Dict[str, Dict[str, float]] = {}
        self.last_edge_weight_matrix: Optional[np.ndarray] = None
        self.last_global_reward = 0.0
        self.last_reward_dict: Dict[str, float] = {}
        self.last_action_dict: Dict[str, int] = {}
        self.last_local_stats: Dict[str, Dict[str, float]] = {}
        self.episode_global_rewards: List[float] = []
        self.episode_Q_net: List[float] = []
        self.episode_E_net: List[float] = []

        self._validate_network_consistency()
        self._build_spaces()

    def _validate_network_consistency(self):
        if not self.static.tls_ids:
            raise ValueError("No traffic lights found in network mapping.")
        action_nums = [self.static.mapping_dict[tid].num_actions for tid in self.static.tls_ids]
        if len(set(action_nums)) != 1:
            raise ValueError(f"Inconsistent num_actions across tls_ids: {dict(zip(self.static.tls_ids, action_nums))}")
        self.config.NUM_TLS = len(self.static.tls_ids)
        self.config.TLS_IDS = list(self.static.tls_ids)
        self.config.NUM_ACTIONS = int(action_nums[0])
        self.config.A_MAX = int(action_nums[0])
        self.config.A_NET_SHAPE = self.static.adj_matrix.shape

    def _build_spaces(self):
        self.action_space = spaces.Dict({tl_id: spaces.Discrete(self.static.mapping_dict[tl_id].num_actions) for tl_id in self.static.tls_ids})
        per_tls_spaces = {}
        a_max = int(self.config.A_MAX)
        for tl_id in self.static.tls_ids:
            n = len(self.static.mapping_dict[tl_id].ordered_incoming_lanes)
            per_tls_spaces[tl_id] = spaces.Dict({
                "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(n, int(self.config.NODE_FEATURE_DIM)), dtype=np.float32),
                "A_same": spaces.Box(low=0.0, high=1.0, shape=(n, n), dtype=np.float32),
                "A_diff": spaces.Box(low=0.0, high=1.0, shape=(n, n), dtype=np.float32),
                "lane_exist_mask": spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32),
                "neighbor_dir_mask": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "neighbor_policy_dir": spaces.Box(low=0.0, high=1.0, shape=(4, a_max), dtype=np.float32),
                "neighbor_policy_action_mask": spaces.Box(low=0.0, high=1.0, shape=(4, a_max), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(a_max,), dtype=np.float32),
            })
        E = len(self.static.edge_ids)
        N = len(self.static.tls_ids)
        self.observation_space = spaces.Dict({
            "per_tls_obs": spaces.Dict(per_tls_spaces),
            "network_obs": spaces.Dict({
                "X_edge": spaces.Box(low=-np.inf, high=np.inf, shape=(E, int(self.config.UPPER_EDGE_FEATURE_DIM)), dtype=np.float32),
                "A_edge_up": spaces.Box(low=0.0, high=1.0, shape=(E, E), dtype=np.float32),
                "A_edge_down": spaces.Box(low=0.0, high=1.0, shape=(E, E), dtype=np.float32),
                "edge_to_tls": spaces.Box(low=0.0, high=1.0, shape=(N, E), dtype=np.float32),
                "edge_weight": spaces.Box(low=0.0, high=np.inf, shape=(E,), dtype=np.float32),
            }),
        })

    def _subscribe_all(self):
        all_lanes = set()
        all_detectors = set()
        for node in self.nodes.values():
            all_lanes.update(node.lanes_in)
            all_detectors.update(node.detectors)
        for lanes in self.static.edge_lanes.values():
            all_lanes.update(lanes)
        for lane_id in all_lanes:
            try:
                traci.lane.subscribe(lane_id, [
                    traci.constants.LAST_STEP_VEHICLE_NUMBER,
                    traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    traci.constants.LAST_STEP_MEAN_SPEED,
                    traci.constants.LAST_STEP_VEHICLE_ID_LIST,
                    traci.constants.LAST_STEP_LENGTH,
                ])
            except Exception:
                pass
        for det_id in all_detectors:
            try:
                traci.lanearea.subscribe(det_id, [traci.constants.LAST_STEP_VEHICLE_NUMBER, traci.constants.LAST_STEP_VEHICLE_ID_LIST])
            except Exception:
                pass

    def _start_sumo(self):
        try:
            traci.start(self.sumo_cmd, port=self.port)
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO: {e}") from e

    def _close_if_loaded(self):
        if traci.isLoaded():
            try:
                traci.close(wait=True)
                sys.stdout.flush()
            except Exception:
                pass

    def close(self):
        self._close_if_loaded()

    def set_upper_weights(self, weights):
        if isinstance(weights, dict):
            mat = np.zeros((len(self.static.tls_ids), 2), dtype=np.float32)
            for i, tl_id in enumerate(self.static.tls_ids):
                mat[i] = np.asarray(weights.get(tl_id, [0.5, 0.5]), dtype=np.float32)
        else:
            mat = np.asarray(weights, dtype=np.float32)
        if mat.shape != (len(self.static.tls_ids), 2):
            raise ValueError(f"upper_weights shape mismatch, expected {(len(self.static.tls_ids), 2)}, got {mat.shape}")
        mat = np.clip(mat, 1e-6, None)
        self.upper_weights = (mat / mat.sum(axis=1, keepdims=True)).astype(np.float32)

    def get_upper_weights_dict(self) -> Dict[str, np.ndarray]:
        return {tl_id: self.upper_weights[i].copy() for i, tl_id in enumerate(self.static.tls_ids)}

    # ---------------- lower obs ----------------
    def _build_node_features(self, lower_state_dict: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        norm = lower_state_dict["norm"]
        raw = lower_state_dict["raw"]
        wave = np.asarray(norm["wave"], dtype=np.float32)
        speed = np.asarray(norm["speed"], dtype=np.float32)
        truck_ratio = np.asarray(norm["truck_ratio"], dtype=np.float32)
        wave_raw = np.asarray(raw.get("wave", np.zeros_like(wave)), dtype=np.float32)
        queue_raw = np.asarray(raw.get("queue", np.zeros_like(wave)), dtype=np.float32)
        queue_ratio = np.clip(queue_raw / np.maximum(wave_raw, 1.0), 0.0, 1.0).astype(np.float32)
        return np.stack([wave, speed, truck_ratio, queue_ratio], axis=-1).astype(np.float32)

    def _get_graph_static_inputs_for_tls(self, tl_id: str) -> Tuple[np.ndarray, np.ndarray]:
        b = self.static.mapping_dict[tl_id]
        return np.asarray(b.same_group_mask, dtype=np.float32), np.asarray(b.diff_group_mask, dtype=np.float32)

    def _build_lane_exist_mask(self, tl_id: str) -> np.ndarray:
        return np.ones(len(self.static.mapping_dict[tl_id].ordered_incoming_lanes), dtype=np.float32)

    def _build_action_mask(self, tl_id: str) -> np.ndarray:
        a_max = int(self.config.A_MAX)
        mask = np.zeros(a_max, dtype=np.float32)
        mask[:int(self.static.mapping_dict[tl_id].num_actions)] = 1.0
        return mask

    def _init_neighbor_policy_placeholders(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a_max = int(self.config.A_MAX)
        return np.zeros(4, dtype=np.float32), np.zeros((4, a_max), dtype=np.float32), np.zeros((4, a_max), dtype=np.float32)

    def _build_per_tls_obs(self) -> Dict[str, Dict[str, Any]]:
        per_tls_obs: Dict[str, Dict[str, Any]] = {}
        all_lower_states = {tl_id: self.nodes[tl_id].get_lower_state_dict() for tl_id in self.static.tls_ids}
        for tl_id in self.static.tls_ids:
            lower_state = all_lower_states[tl_id]
            neighbor_dir_mask, neighbor_policy_dir, neighbor_policy_action_mask = self._init_neighbor_policy_placeholders()
            for idx, d in enumerate(self.config.NEIGHBOR_DIRECTIONS):
                nbr_id = self.static.tls_neighbors.get(tl_id, {}).get(d, None)
                if nbr_id is not None:
                    neighbor_dir_mask[idx] = 1.0
            A_same, A_diff = self._get_graph_static_inputs_for_tls(tl_id)
            per_tls_obs[tl_id] = {
                "node_features": self._build_node_features(lower_state),
                "A_same": A_same,
                "A_diff": A_diff,
                "lane_exist_mask": self._build_lane_exist_mask(tl_id),
                "neighbor_dir_mask": neighbor_dir_mask,
                "neighbor_policy_dir": neighbor_policy_dir,
                "neighbor_policy_action_mask": neighbor_policy_action_mask,
                "action_mask": self._build_action_mask(tl_id),
                "raw": lower_state["raw"],
                "norm": lower_state["norm"],
            }
        return per_tls_obs

    # ---------------- upper edge obs / metrics ----------------
    def _is_truck_type(self, vtype: str) -> bool:
        return "truck" in str(vtype).lower() or str(vtype).lower() in {"hdv", "lorry"}

    def _get_vehicle_type_cached(self, vid: str) -> str:
        vt = self._vehicle_type_cache.get(vid)
        if vt is not None:
            return vt
        try:
            vt = traci.vehicle.getTypeID(vid)
        except Exception:
            vt = str(getattr(self.config, "DEFAULT_VEHICLE_TYPE", "sedan"))
        self._vehicle_type_cache[vid] = vt
        return vt

    def _collect_edge_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        eps = 1e-6
        for edge_id in self.static.edge_ids:
            veh_ids = []
            vehcount = 0.0
            halt = 0.0
            emission_mg = 0.0
            for lane_id in self.static.edge_lanes.get(edge_id, []):
                try:
                    sub = traci.lane.getSubscriptionResults(lane_id) or {}
                    vehcount += float(sub.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, traci.lane.getLastStepVehicleNumber(lane_id)))
                    halt += float(sub.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, traci.lane.getLastStepHaltingNumber(lane_id)))
                    lane_vids = list(sub.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, traci.lane.getLastStepVehicleIDs(lane_id)))
                    veh_ids.extend(lane_vids)
                except Exception:
                    continue
            truck_count = 0.0
            for vid in veh_ids:
                vt = self._get_vehicle_type_cached(vid)
                if self._is_truck_type(vt):
                    truck_count += 1.0
                try:
                    emission_g = self.shared_emission_lookup.get_emission(
                        vehicle_type=vt,
                        speed_ms=traci.vehicle.getSpeed(vid),
                        accel_ms2=traci.vehicle.getAcceleration(vid),
                        sim_step=self.config.SIM_STEP,
                        pollutant=self.config.EMISSION_POLLUTANT,
                    )
                    emission_mg += float(emission_g) * 1000.0
                except Exception:
                    pass
            cap = float(self.static.edge_capacity.get(edge_id, max(vehcount, 1.0)))
            cap = max(cap, 1.0)
            lane_num = len(self.static.edge_lanes.get(edge_id, []))
            lane_num = max(float(lane_num), 1.0)
            vehcount_norm = np.clip(vehcount / max(cap, eps), 0.0, 2.0)
            truck_ratio = truck_count / max(len(veh_ids), 1.0)
            queue_norm = np.clip(halt / max(cap, eps), 0.0, 2.0)
            emission_ref = float(getattr(self.config, "EMISSION_MAX", 150.0)) * lane_num
            emission_ref = max(emission_ref, eps)
            emission_norm = np.clip(emission_mg / emission_ref, 0.0, 2.0)
            q_thr = float(getattr(self.config, "UPPER_QUEUE_THR_NORM", 0.7))
            e_thr = float(getattr(self.config, "UPPER_EMISSION_THR_NORM", 0.7))
            q_hot = max(0.0, (queue_norm - q_thr) / max(q_thr, eps))
            e_hot = max(0.0, (emission_norm - e_thr) / max(e_thr, eps))
            metrics[edge_id] = {
                "vehcount_raw": float(vehcount),
                "vehcount_norm": float(vehcount_norm),
                "truck_count": float(truck_count),
                "truck_ratio": float(truck_ratio),
                "queue_raw": float(halt),
                "queue_norm": float(queue_norm),
                "emission_raw": float(emission_mg),
                "emission_ref": float(emission_ref),
                "emission_norm": float(emission_norm),
                "queue_hotspot": float(q_hot),
                "emission_hotspot": float(e_hot),
                "edge_capacity": float(cap),
            }
        self.last_edge_metrics = metrics
        return metrics

    def _build_edge_features(self, edge_metrics: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros((len(self.static.edge_ids), int(self.config.UPPER_EDGE_FEATURE_DIM)), dtype=np.float32)
        edge_weight = np.ones(len(self.static.edge_ids), dtype=np.float32)
        for i, eid in enumerate(self.static.edge_ids):
            m = edge_metrics.get(eid, {})
            X[i, 0] = float(m.get("vehcount_norm", 0.0))
            X[i, 1] = float(m.get("truck_ratio", 0.0))
            X[i, 2] = float(m.get("queue_norm", 0.0))
            edge_weight[i] = max(float(m.get("vehcount_raw", 0.0)), 1.0)
        return X, edge_weight

    def _update_upper_metrics_from_edges(self, edge_metrics: Dict[str, Dict[str, float]]):
        q = np.asarray([m.get("queue_norm", 0.0) for m in edge_metrics.values()], dtype=np.float32)
        e = np.asarray([m.get("emission_norm", 0.0) for m in edge_metrics.values()], dtype=np.float32)
        qhot = np.asarray([m.get("queue_hotspot", 0.0) for m in edge_metrics.values()], dtype=np.float32)
        ehot = np.asarray([m.get("emission_hotspot", 0.0) for m in edge_metrics.values()], dtype=np.float32)
        eta_q = float(getattr(self.config, "UPPER_HOTSPOT_ETA_Q", 0.5))
        eta_e = float(getattr(self.config, "UPPER_HOTSPOT_ETA_E", 0.5))
        Q_sum = float(q.sum()) if q.size else 0.0
        E_sum = float(e.sum()) if e.size else 0.0
        P_hot = float((eta_q * qhot + eta_e * ehot).sum()) if q.size else 0.0
        self.last_upper_metrics = {
            "E_net": E_sum, "Q_net": Q_sum, "H_net": P_hot, "B_net": 0.0,
            "E_sum": E_sum, "Q_sum": Q_sum, "P_hot": P_hot,
        }

    def _build_network_obs(self) -> Dict[str, Any]:
        edge_metrics = self._collect_edge_metrics()
        X_edge, edge_weight = self._build_edge_features(edge_metrics)
        self._update_upper_metrics_from_edges(edge_metrics)
        return {
            "X_edge": X_edge,
            "A_edge_up": self.static.A_edge_up.copy().astype(np.float32),
            "A_edge_down": self.static.A_edge_down.copy().astype(np.float32),
            "edge_to_tls": self.static.edge_to_tls_matrix.copy().astype(np.float32),
            "edge_weight": edge_weight.astype(np.float32),
        }

    def _build_obs(self) -> Dict[str, Any]:
        return {"per_tls_obs": self._build_per_tls_obs(), "network_obs": self._build_network_obs()}

    # ---------------- simulation helpers ----------------
    @staticmethod
    def _make_yellow_state(prev_state: str) -> str:
        return "".join("y" if ch in ("g", "G", "y", "Y") else "r" for ch in prev_state)

    def _phase_state_from_action(self, tl_id: str, action: int) -> str:
        return str(self.static.mapping_dict[tl_id].rl_phases[int(action)])

    def _run_simulation_seconds(self, seconds: float):
        if seconds <= 0:
            return
        num_steps = int(round(seconds / self.sim_step))
        for _ in range(num_steps):
            try:
                traci.simulationStep()
                self.simulation_time += self.sim_step
                for node in self.nodes.values():
                    node.accumulate_step_emission()
            except traci.exceptions.FatalTraCIError:
                break

    def _change_phase(self, action_dict: Dict[str, int]):
        changed = []
        for tl_id, new_action in action_dict.items():
            prev_action = int(getattr(self.nodes[tl_id], "prev_action", new_action))
            prev_state = self._phase_state_from_action(tl_id, prev_action)
            new_state = self._phase_state_from_action(tl_id, int(new_action))
            if int(new_action) != prev_action and self.yellow_time > 0:
                try:
                    traci.trafficlight.setRedYellowGreenState(tl_id, self._make_yellow_state(prev_state))
                    changed.append((tl_id, new_state))
                except Exception:
                    pass
            else:
                try:
                    traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
                except Exception:
                    pass
        if changed and self.yellow_time > 0:
            self._run_simulation_seconds(self.yellow_time)
            for tl_id, new_state in changed:
                try:
                    traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
                except Exception:
                    pass
            self._run_simulation_seconds(max(0.0, self.delta_t - self.yellow_time))
        else:
            self._run_simulation_seconds(self.delta_t)
        for tl_id, a in action_dict.items():
            self.nodes[tl_id].prev_action = int(a)

    # ---------------- env api ----------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._close_if_loaded()
        self.current_step = 0
        self.simulation_time = 0.0
        self.last_global_reward = 0.0
        self.last_reward_dict = {}
        self.last_action_dict = {}
        self.last_local_stats = {}
        self.last_edge_metrics = {}
        self.episode_global_rewards = []
        self.episode_Q_net = []
        self.episode_E_net = []
        if seed is not None:
            self.config.SUMO_SEED = int(seed)
            self.sumo_cmd = build_sumo_cmd(self.config, gui=self.is_gui)
        self._start_sumo()
        self._subscribe_all()
        for node in self.nodes.values():
            node.prev_action = 0
            node.reset_step_emission()
            node.cache_speed_limits()
        return self._build_obs(), {}

    def _collect_node_statistics(self) -> Dict[str, Dict[str, float]]:
        return {tl_id: self.nodes[tl_id].get_statistics() for tl_id in self.static.tls_ids}

    def _compute_reward_dict(self, stats: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], float]:
        local_cost = {}
        q_coef = float(getattr(self.config, "LOWER_EFF_QUEUE_COEF", 0.5))
        v_coef = float(getattr(self.config, "LOWER_EFF_SPEEDLACK_COEF", 0.5))
        alpha = float(getattr(self.config, "LOWER_REWARD_SPATIAL_ALPHA", 0.5))
        beta = float(getattr(self.config, "LOWER_NEIGHBOR_REWARD_COEF", 0.3))
        for i, tl_id in enumerate(self.static.tls_ids):
            w_em, w_eff = self.upper_weights[i]
            s = stats[tl_id]
            e = float(s.get("mean_emission_norm", 0.0))
            q = float(s.get("mean_queue_norm", 0.0))
            v = float(s.get("mean_speed_lack", 0.0))
            eff = q_coef * q + v_coef * v
            local_cost[tl_id] = float(w_em) * e + float(w_eff) * eff
        reward_dict = {}
        for tl_id in self.static.tls_ids:
            nbr_ids = [x for x in dict.fromkeys(self.static.tls_neighbors.get(tl_id, {}).values()) if x in local_cost]
            nbr_cost = float(np.mean([local_cost[x] for x in nbr_ids])) if nbr_ids else 0.0
            reward_dict[tl_id] = -float(local_cost[tl_id] + beta * alpha * nbr_cost)
        global_reward = float(np.mean(list(reward_dict.values()))) if reward_dict else 0.0
        return reward_dict, global_reward

    def _build_info(self, reward_dict: Dict[str, float], global_reward: float) -> Dict[str, Any]:
        return {
            "simulation_time": float(self.simulation_time),
            "current_step": int(self.current_step),
            "global_reward": float(global_reward),
            "reward_dict": reward_dict,
            "upper_metrics": dict(self.last_upper_metrics),
            "local_stats": self.last_local_stats,
            "edge_metrics": self.last_edge_metrics,
            "upper_weights": self.get_upper_weights_dict(),
        }

    def step(self, action_dict: Dict[str, int]):
        for tl_id in self.static.tls_ids:
            if tl_id not in action_dict:
                raise KeyError(f"Missing action for tls_id={tl_id}")
        for node in self.nodes.values():
            node.reset_step_emission()
        self._change_phase(action_dict)
        for node in self.nodes.values():
            node.finalize_step_emission()
        self.current_step += 1
        obs = self._build_obs()
        self.last_local_stats = self._collect_node_statistics()
        reward_dict, global_reward = self._compute_reward_dict(self.last_local_stats)
        self.last_reward_dict = dict(reward_dict)
        self.last_global_reward = float(global_reward)
        self.last_action_dict = {k: int(v) for k, v in action_dict.items()}
        terminated = self.current_step >= self.max_steps
        truncated = False
        self.episode_global_rewards.append(global_reward)
        self.episode_Q_net.append(float(self.last_upper_metrics.get("Q_net", 0.0)))
        self.episode_E_net.append(float(self.last_upper_metrics.get("E_net", 0.0)))
        return obs, reward_dict, terminated, truncated, self._build_info(reward_dict, global_reward)

    # ---------------- snapshots / logger helpers ----------------
    def get_upper_step_snapshot(self) -> Dict[str, Any]:
        return {**self.last_upper_metrics, "step": int(self.current_step)}

    def get_network_step_metrics(self) -> Dict[str, float]:
        w_em = self.upper_weights[:, 0] if len(self.upper_weights) > 0 else np.array([0.0], dtype=np.float32)
        w_eff = self.upper_weights[:, 1] if len(self.upper_weights) > 0 else np.array([0.0], dtype=np.float32)
        return {
            "step": int(self.current_step),
            "simulation_time": float(self.simulation_time),
            "global_reward": float(self.last_global_reward),
            "E_net": float(self.last_upper_metrics.get("E_net", 0.0)),
            "Q_net": float(self.last_upper_metrics.get("Q_net", 0.0)),
            "H_net": float(self.last_upper_metrics.get("H_net", 0.0)),
            "B_net": float(self.last_upper_metrics.get("B_net", 0.0)),
            "E_sum": float(self.last_upper_metrics.get("E_sum", 0.0)),
            "Q_sum": float(self.last_upper_metrics.get("Q_sum", 0.0)),
            "P_hot": float(self.last_upper_metrics.get("P_hot", 0.0)),
            "mean_w_em": float(np.mean(w_em)),
            "std_w_em": float(np.std(w_em)),
            "mean_w_eff": float(np.mean(w_eff)),
            "std_w_eff": float(np.std(w_eff)),
        }

    def get_edge_step_metrics(self) -> List[Dict[str, Any]]:
        rows = []
        for eid in self.static.edge_ids:
            m = self.last_edge_metrics.get(eid, {})
            row = {"step": int(self.current_step), "simulation_time": float(self.simulation_time), "edge_id": eid}
            row.update({k: float(v) for k, v in m.items()})
            rows.append(row)
        return rows

    def get_tls_step_metrics(self, reward_dict: Optional[Dict[str, float]] = None, action_dict: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        reward_dict = self.last_reward_dict if reward_dict is None else reward_dict
        action_dict = self.last_action_dict if action_dict is None else action_dict
        rows = []
        for i, tl_id in enumerate(self.static.tls_ids):
            s = self.last_local_stats.get(tl_id, {})
            rows.append({
                "step": int(self.current_step), "simulation_time": float(self.simulation_time), "tls_id": tl_id,
                "mean_wave_norm": float(s.get("mean_wave_norm", 0.0)),
                "mean_queue_norm": float(s.get("mean_queue_norm", 0.0)),
                "mean_wait_norm": float(s.get("mean_wait_norm", 0.0)),
                "mean_speed_lack": float(s.get("mean_speed_lack", 0.0)),
                "mean_truck_ratio_norm": float(s.get("mean_truck_ratio_norm", 0.0)),
                "mean_emission_norm": float(s.get("mean_emission_norm", 0.0)),
                "action": int(action_dict.get(tl_id, 0)),
                "current_phase": int(getattr(self.nodes[tl_id], "prev_action", 0)),
                "green_duration": float(self.delta_t),
                "local_reward": float(reward_dict.get(tl_id, 0.0)),
                "w_em": float(self.upper_weights[i, 0]),
                "w_eff": float(self.upper_weights[i, 1]),
            })
        return rows

    def get_tls_node_feature_rows(self, mode: str = "raw") -> List[Dict[str, Any]]:
        rows = []
        key = "raw" if mode == "raw" else "norm"
        per_tls_obs = self._build_per_tls_obs()
        for tl_id in self.static.tls_ids:
            state = per_tls_obs[tl_id][key]
            lanes = self.static.mapping_dict[tl_id].ordered_incoming_lanes
            for idx, lane_id in enumerate(lanes):
                rows.append({
                    "step": int(self.current_step), "simulation_time": float(self.simulation_time), "tls_id": tl_id,
                    "node_idx": int(idx), "lane_id": lane_id,
                    "wave": float(np.asarray(state["wave"])[idx]),
                    "queue": float(np.asarray(state.get("queue", np.zeros(len(lanes), dtype=np.float32)))[idx]),
                    "wait": float(np.asarray(state.get("wait", np.zeros(len(lanes), dtype=np.float32)))[idx]),
                    "speed": float(np.asarray(state["speed"])[idx]),
                    "truck_ratio": float(np.asarray(state["truck_ratio"])[idx]),
                    "emission": float(np.asarray(state.get("emission", np.zeros(len(lanes), dtype=np.float32)))[idx]),
                })
        return rows

    def get_vehicle_trip_rows(self) -> List[Dict[str, Any]]:
        return []

    def get_episode_summary_raw_row(self) -> Dict[str, Any]:
        return {"episode": 0, **self.get_episode_statistics()}

    def get_episode_statistics(self) -> Dict[str, Any]:
        total_global_reward = float(np.sum(self.episode_global_rewards)) if self.episode_global_rewards else 0.0
        return {
            "steps": int(self.current_step),
            "total_global_reward": total_global_reward,
            "avg_global_reward": float(np.mean(self.episode_global_rewards)) if self.episode_global_rewards else 0.0,
            "avg_Q_net": float(np.mean(self.episode_Q_net)) if self.episode_Q_net else 0.0,
            "avg_E_net": float(np.mean(self.episode_E_net)) if self.episode_E_net else 0.0,
        }
