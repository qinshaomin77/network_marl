# -*- coding: utf-8 -*-
"""
utils.py
"""

from __future__ import annotations

from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np
import traci

from emission_lookup import EmissionFactorLookup


# =============================================================================
# SUMO 命令构建
# =============================================================================

def build_sumo_cmd(config, gui: Optional[bool] = None):
    if gui is None:
        gui = config.SUMO_GUI
    sumo_binary = "sumo-gui" if gui else "sumo"
    output_dir = config.SIM_DIR if hasattr(config, "SIM_DIR") else config.OUTPUT_DIR
    output_dir = Path(output_dir) / f"seed_{config.SUMO_SEED}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.xml"
    tripinfo_path = output_dir / "tripinfo.xml"
    vehroute_path = output_dir / "vehroutes.xml"
    fcd_path = output_dir / "fcd.xml"

    cmd = [
        sumo_binary,
        "-n", str(config.NET_FILE),
        "-r", str(config.ROUTE_FILE),
        "--additional-files", str(config.ADDITIONAL_FILE),
        "--summary-output", str(summary_path),
        "--tripinfo-output", str(tripinfo_path),
        # "--vehroute-output", str(vehroute_path),
        # "--fcd-output", str(fcd_path),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
        "--seed", str(config.SUMO_SEED),
    ]
    return cmd


# =============================================================================
# 单路口动态测量器
# =============================================================================

class Node:
    """
    单个交叉口的动态状态采集器
    """

    def __init__(self, name: str, config: Any, lane_mapping, emission_lookup=None):
        self.name = name
        self.config = config
        self.mapping = lane_mapping

        self.lanes_in = list(lane_mapping.ordered_incoming_lanes)
        self.detectors = list(lane_mapping.ordered_detector_ids)
        self.num_nodes = len(self.lanes_in)

        self.lane_to_detector = dict(lane_mapping.lane_to_detector)
        self.lane_to_link_index = dict(lane_mapping.lane_to_link_index)
        self.link_index_to_lane = dict(lane_mapping.link_index_to_lane)

        self.phase_id = lane_mapping.traffic_light_id
        self.n_a = lane_mapping.num_actions
        self.rl_phases = list(lane_mapping.rl_phases)

        self.prev_action: Optional[int] = None
        self.current_phase_duration: float = 0.0

        self.last_wave_raw = None
        self.last_wave_norm = None

        self.last_queue_raw = None
        self.last_queue_norm = None

        self.last_wait_raw = None
        self.last_wait_norm = None

        self.last_speed_raw = None
        self.last_speed_norm = None

        self.last_truck_ratio_raw = None
        self.last_truck_ratio_norm = None

        self.last_emission_raw = None
        self.last_emission_norm = None

        self.last_phase_raw = None
        self.last_phase_norm = None

        self.last_state_dict = None
        
        self._vehicle_type_cache: Dict[str, str] = {}
        self.step_emission_accumulator_raw = np.zeros(self.num_nodes, dtype=np.float32)
        if emission_lookup is not None:
            self.emission_lookup = emission_lookup
        else:
            self.emission_lookup = EmissionFactorLookup(
                csv_path=self.config.EMISSION_FACTOR_FILE,
                default_pollutant=self.config.EMISSION_POLLUTANT,
                default_vehicle_type=self.config.DEFAULT_VEHICLE_TYPE,
                enable_cache=self.config.EMISSION_ENABLE_CACHE,
            )

        self._speed_limit_cache: Dict[str, float] = {}
        self._vehicle_type_cache: Dict[str, str] = {}

        self._validate_order()
        self._init_zero_states()

    # -------------------------------------------------------------------------
    # 初始化与工具
    # -------------------------------------------------------------------------

    def _validate_order(self):
        link_indices = sorted(self.link_index_to_lane.keys())
        expected = list(range(self.num_nodes))
        if link_indices != expected:
            raise ValueError(
                f"[Node:{self.name}] linkIndex not continuous: "
                f"got {link_indices}, expected {expected}"
            )

        if self.rl_phases:
            phase_len = len(self.rl_phases[0])
            if phase_len != self.num_nodes:
                raise ValueError(
                    f"[Node:{self.name}] node count mismatch: "
                    f"mapping has {self.num_nodes}, phase string has {phase_len}"
                )

    def _init_zero_states(self):
        zero_nodes = np.zeros(self.num_nodes, dtype=np.float32)
        zero_phase = np.zeros(self.n_a, dtype=np.float32)

        self.step_emission_accumulator_raw = zero_nodes.copy()

        self.last_wave_raw = zero_nodes.copy()
        self.last_wave_norm = zero_nodes.copy()

        self.last_queue_raw = zero_nodes.copy()
        self.last_queue_norm = zero_nodes.copy()

        self.last_wait_raw = zero_nodes.copy()
        self.last_wait_norm = zero_nodes.copy()

        self.last_speed_raw = zero_nodes.copy()
        self.last_speed_norm = zero_nodes.copy()

        self.last_truck_ratio_raw = zero_nodes.copy()
        self.last_truck_ratio_norm = zero_nodes.copy()

        self.last_emission_raw = zero_nodes.copy()
        self.last_emission_norm = zero_nodes.copy()

        self.last_phase_raw = zero_phase.copy()
        self.last_phase_norm = zero_phase.copy()

    def _normalize(self, arr, max_value, clip_value=None):
        arr = np.asarray(arr, dtype=np.float32)
        if max_value is None or max_value <= 0:
            out = np.zeros_like(arr, dtype=np.float32)
        else:
            out = arr / float(max_value)

        if clip_value is not None:
            out = np.clip(out, 0.0, float(clip_value))

        return out.astype(np.float32)

    def _safe_copy(self, arr):
        return None if arr is None else np.asarray(arr, dtype=np.float32).copy()

    # -------------------------------------------------------------------------
    # 排放累计
    # -------------------------------------------------------------------------

    def reset_step_emission(self):
        self.step_emission_accumulator_raw = np.zeros(self.num_nodes, dtype=np.float32)

    def _measure_emission_instant(self) -> np.ndarray:
        raw = []
        if self.emission_lookup is None:
            return np.zeros(self.num_nodes, dtype=np.float32)
        for det_id in self.detectors:
            lane_emission_mg = 0.0
            try:
                sub = traci.lanearea.getSubscriptionResults(det_id)
                veh_ids = sub.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, ())
                if not veh_ids:
                    raw.append(0.0)
                    continue
                for vid in veh_ids:
                    try:
                        vtype = self._vehicle_type_cache.get(vid)
                        if vtype is None:
                            vtype = traci.vehicle.getTypeID(vid)
                            self._vehicle_type_cache[vid] = vtype
                        speed_ms = traci.vehicle.getSpeed(vid)
                        accel_ms2 = traci.vehicle.getAcceleration(vid)
                        emission_g = self.emission_lookup.get_emission(
                            vehicle_type=vtype,
                            speed_ms=speed_ms,
                            accel_ms2=accel_ms2,
                            sim_step=self.config.SIM_STEP,
                            pollutant=self.config.EMISSION_POLLUTANT,
                        )
                        lane_emission_mg += emission_g * 1000.0
                    except traci.exceptions.TraCIException:
                        continue
                raw.append(lane_emission_mg)
            except Exception:
                raw.append(0.0)
        return np.asarray(raw, dtype=np.float32)

    def accumulate_step_emission(self):
        instant_emission = self._measure_emission_instant()
        self.step_emission_accumulator_raw += instant_emission

    def finalize_step_emission(self):
        self.last_emission_raw = self.step_emission_accumulator_raw.copy().astype(np.float32)
        self.last_emission_norm = self._normalize(
            self.last_emission_raw,
            self.config.EMISSION_MAX,
            self.config.CLIP_EMISSION,
        )

    # -------------------------------------------------------------------------
    # 单项特征测量
    # -------------------------------------------------------------------------

    def _measure_wave(self):
        raw = []
        for det_id in self.detectors:
            try:
                sub = traci.lanearea.getSubscriptionResults(det_id)
                raw.append(sub.get(traci.constants.LAST_STEP_VEHICLE_NUMBER, 0))
            except Exception:
                raw.append(0)
        raw = np.asarray(raw, dtype=np.float32)
        norm = self._normalize(raw, self.config.WAVE_MAX, self.config.CLIP_WAVE)
        self.last_wave_raw = raw
        self.last_wave_norm = norm
        return raw, norm

    def _measure_queue(self):
        raw = []
        for lane in self.lanes_in:
            try:
                sub = traci.lane.getSubscriptionResults(lane)
                raw.append(sub.get(traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, 0))
            except Exception:
                raw.append(0)
        raw = np.asarray(raw, dtype=np.float32)
        norm = self._normalize(raw, self.config.QUEUE_MAX, self.config.CLIP_QUEUE)
        self.last_queue_raw = raw
        self.last_queue_norm = norm
        return raw, norm

    def _measure_wait(self):
        raw = []
        for det_id in self.detectors:
            try:
                sub = traci.lanearea.getSubscriptionResults(det_id)
                veh_ids = sub.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, ())
                if not veh_ids:
                    raw.append(0.0)
                    continue
                leader_pos = -1.0
                leader_wait = 0.0
                for vid in veh_ids:
                    pos = traci.vehicle.getLanePosition(vid)
                    if pos > leader_pos:
                        leader_pos = pos
                        leader_wait = traci.vehicle.getWaitingTime(vid)
                raw.append(leader_wait)
            except Exception:
                raw.append(0.0)
        raw = np.asarray(raw, dtype=np.float32)
        norm = self._normalize(raw, self.config.WAIT_MAX, self.config.CLIP_WAIT)
        self.last_wait_raw = raw
        self.last_wait_norm = norm
        return raw, norm

    def cache_speed_limits(self):
        """reset 时调用一次，缓存所有 lane 的限速"""
        for lane in self.lanes_in:
            try:
                self._speed_limit_cache[lane] = traci.lane.getMaxSpeed(lane)
            except traci.exceptions.TraCIException:
                self._speed_limit_cache[lane] = 13.89

    def _measure_speed(self):
        raw = []
        norm = []

        for lane in self.lanes_in:
            try:
                avg_speed = traci.lane.getLastStepMeanSpeed(lane)
                speed_limit = traci.lane.getMaxSpeed(lane)
                veh_count = traci.lane.getLastStepVehicleNumber(lane)

                raw.append(avg_speed)

                if veh_count == 0:
                    norm.append(0.0)
                elif speed_limit > 0:
                    speed_ratio = min(1.0, avg_speed / speed_limit)
                    norm.append(1.0 - speed_ratio)
                else:
                    norm.append(1.0)

            except traci.exceptions.TraCIException:
                raw.append(0.0)
                norm.append(0.0)

        raw = np.asarray(raw, dtype=np.float32)
        norm = np.asarray(norm, dtype=np.float32)

        self.last_speed_raw = raw
        self.last_speed_norm = norm
        return raw, norm

    def _measure_truck_ratio(self):
        raw = []
        for lane in self.lanes_in:
            try:
                sub = traci.lane.getSubscriptionResults(lane)
                veh_ids = sub.get(traci.constants.LAST_STEP_VEHICLE_ID_LIST, ())
                if not veh_ids:
                    raw.append(0.0)
                    continue
                total_count = len(veh_ids)
                truck_count = 0
                for vid in veh_ids:
                    vtype = self._vehicle_type_cache.get(vid)
                    if vtype is None:
                        try:
                            vtype = traci.vehicle.getTypeID(vid)
                            self._vehicle_type_cache[vid] = vtype
                        except traci.exceptions.TraCIException:
                            continue
                    if vtype == "truck":
                        truck_count += 1
                raw.append(truck_count / total_count)
            except Exception:
                raw.append(0.0)
        raw = np.asarray(raw, dtype=np.float32)
        norm = self._normalize(raw, self.config.TRUCK_RATIO_MAX)
        self.last_truck_ratio_raw = raw
        self.last_truck_ratio_norm = norm
        return raw, norm

    def _get_emission_state(self):
        if self.last_emission_raw is None or self.last_emission_norm is None:
            self.finalize_step_emission()

        return (
            np.asarray(self.last_emission_raw, dtype=np.float32).copy(),
            np.asarray(self.last_emission_norm, dtype=np.float32).copy(),
        )

    def _get_phase_state(self):
        phase_onehot = np.zeros(self.n_a, dtype=np.float32)
        if self.prev_action is not None and 0 <= self.prev_action < self.n_a:
            phase_onehot[self.prev_action] = 1.0

        self.last_phase_raw = phase_onehot.copy()
        self.last_phase_norm = phase_onehot.copy()
        return self.last_phase_raw.copy(), self.last_phase_norm.copy()

    # -------------------------------------------------------------------------
    # 结构化状态输出
    # -------------------------------------------------------------------------

    def get_state_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        state_raw = {}
        state_norm = {}

        raw, norm = self._measure_wave()
        state_raw["wave"] = raw
        state_norm["wave"] = norm

        raw, norm = self._measure_queue()
        state_raw["queue"] = raw
        state_norm["queue"] = norm

        raw, norm = self._measure_wait()
        state_raw["wait"] = raw
        state_norm["wait"] = norm

        raw, norm = self._measure_speed()
        state_raw["speed"] = raw
        state_norm["speed"] = norm

        raw, norm = self._measure_truck_ratio()
        state_raw["truck_ratio"] = raw
        state_norm["truck_ratio"] = norm

        raw, norm = self._get_emission_state()
        state_raw["emission"] = raw
        state_norm["emission"] = norm

        raw, norm = self._get_phase_state()
        state_raw["phase"] = raw
        state_norm["phase"] = norm

        state = {"raw": state_raw, "norm": state_norm}
        self.last_state_dict = {
            "raw": {k: v.copy() for k, v in state_raw.items()},
            "norm": {k: v.copy() for k, v in state_norm.items()},
        }
        return state

    def get_lower_state_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        返回供 env 组织 per_tls_obs 使用的状态字典。

        说明：
        - 下层模型真正消费的仍然只是 wave / speed / truck_ratio
        - 这里返回完整 raw / norm 字典，是为了让 env.get_tls_node_feature_rows()
          在记录节点级 CSV 时拿到真实 queue / wait / emission，而不是补零
        """
        state = self.get_state_dict()
        return {
            "raw": {k: v.copy() for k, v in state["raw"].items()},
            "norm": {k: v.copy() for k, v in state["norm"].items()},
        }

    def _sum_or_zero(self, arr):
        return float(np.sum(arr)) if arr is not None else 0.0

    def _mean_or_zero(self, arr):
        return float(np.mean(arr)) if arr is not None else 0.0

    def get_statistics(self) -> Dict[str, float]:
        return {
            "total_wave_raw": self._sum_or_zero(self.last_wave_raw),
            "total_queue_raw": self._sum_or_zero(self.last_queue_raw),
            "avg_queue_raw": self._mean_or_zero(self.last_queue_raw),
            "max_queue_raw": float(np.max(self.last_queue_raw)) if self.last_queue_raw is not None and len(self.last_queue_raw) > 0 else 0.0,
            "total_wait_raw": self._sum_or_zero(self.last_wait_raw),
            "avg_wait_raw": self._mean_or_zero(self.last_wait_raw),
            "max_wait_raw": float(np.max(self.last_wait_raw)) if self.last_wait_raw is not None and len(self.last_wait_raw) > 0 else 0.0,
            "total_emission_raw": self._sum_or_zero(self.last_emission_raw),
            "avg_emission_raw": self._mean_or_zero(self.last_emission_raw),
            "mean_speed_raw": self._mean_or_zero(self.last_speed_raw),
            "min_speed_raw": float(np.min(self.last_speed_raw)) if self.last_speed_raw is not None and len(self.last_speed_raw) > 0 else 0.0,
            "mean_truck_ratio_raw": self._mean_or_zero(self.last_truck_ratio_raw),

            "mean_wave_norm": self._mean_or_zero(self.last_wave_norm),
            "mean_queue_norm": self._mean_or_zero(self.last_queue_norm),
            "mean_wait_norm": self._mean_or_zero(self.last_wait_norm),
            "mean_speed_lack": self._mean_or_zero(self.last_speed_norm),
            "mean_truck_ratio_norm": self._mean_or_zero(self.last_truck_ratio_norm),
            "mean_emission_norm": self._mean_or_zero(self.last_emission_norm),

            "current_phase": -1 if self.prev_action is None else int(self.prev_action),
            "phase_duration": float(self.current_phase_duration),
        }

    def get_debug_features(self):
        return {
            "wave_raw": self._safe_copy(self.last_wave_raw),
            "wave_norm": self._safe_copy(self.last_wave_norm),

            "queue_raw": self._safe_copy(self.last_queue_raw),
            "queue_norm": self._safe_copy(self.last_queue_norm),

            "wait_raw": self._safe_copy(self.last_wait_raw),
            "wait_norm": self._safe_copy(self.last_wait_norm),

            "speed_raw": self._safe_copy(self.last_speed_raw),
            "speed_norm": self._safe_copy(self.last_speed_norm),

            "truck_ratio_raw": self._safe_copy(self.last_truck_ratio_raw),
            "truck_ratio_norm": self._safe_copy(self.last_truck_ratio_norm),

            "step_emission_accumulator_raw": self._safe_copy(self.step_emission_accumulator_raw),
            "emission_raw": self._safe_copy(self.last_emission_raw),
            "emission_norm": self._safe_copy(self.last_emission_norm),

            "phase_raw": self._safe_copy(self.last_phase_raw),
            "phase_norm": self._safe_copy(self.last_phase_norm),

            "state_dict": None if self.last_state_dict is None else {
                "raw": {k: v.copy() for k, v in self.last_state_dict["raw"].items()},
                "norm": {k: v.copy() for k, v in self.last_state_dict["norm"].items()},
            }
        }