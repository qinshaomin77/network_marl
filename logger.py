# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import csv
import json
import numpy as np

# CSV-based experiment logger with buffered writes.
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return x

class _CsvBatchWriter:
    # Buffered CSV writer to reduce frequent I/O flush overhead.
    def __init__(self, path: Path, fieldnames: List[str], flush_interval: int = 100):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.flush_interval = int(flush_interval)
        self.buffer: List[Dict[str, Any]] = []
        self.file = None
        self.writer = None

    def _init(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def add_row(self, row: Dict[str, Any]):
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def add_rows(self, rows: Iterable[Dict[str, Any]]):
        for r in rows:
            self.buffer.append(r)
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        if self.writer is None:
            self._init()
        self.writer.writerows(self.buffer)
        self.file.flush()
        self.buffer.clear()

    def close(self):
        self.flush()
        if self.file is not None:
            self.file.close()

class TrafficLogger:
    # Centralized logging for step-level, update-level and episode-level metrics.
    def __init__(self, config, env=None, output_dir: str | Path = "logs", flush_interval: int = 100, log_params: bool = True):
        self.config = config
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = int(flush_interval)
        self.log_network_step_flag = bool(getattr(config, "LOG_NETWORK_STEP", True))
        self.log_tls_step_flag = bool(getattr(config, "LOG_TLS_STEP", True))
        self.log_edge_step_flag = bool(getattr(config, "LOG_EDGE_STEP", True))
        self.log_node_raw_flag = bool(getattr(config, "LOG_NODE_RAW", True))
        self.log_node_norm_flag = bool(getattr(config, "LOG_NODE_NORM", False))

        self.network_step_writer = _CsvBatchWriter(self.output_dir / "network_step_metrics.csv", [
            "episode", "step", "simulation_time", "global_reward",
            "E_net", "Q_net", "H_net", "B_net", "E_sum", "Q_sum", "P_hot",
            "mean_w_em", "std_w_em", "mean_w_eff", "std_w_eff",
        ], self.flush_interval)
        self.tls_step_writer = _CsvBatchWriter(self.output_dir / "tls_step_metrics.csv", [
            "episode", "step", "simulation_time", "tls_id", "mean_wave_norm", "mean_queue_norm",
            "mean_wait_norm", "mean_speed_lack", "mean_truck_ratio_norm", "mean_emission_norm",
            "action", "current_phase", "green_duration", "local_reward", "w_em", "w_eff",
        ], self.flush_interval)
        self.edge_step_writer = _CsvBatchWriter(self.output_dir / "edge_step_metrics.csv", [
            "episode", "step", "simulation_time", "edge_id",
            "vehcount_raw", "vehcount_norm", "truck_count", "truck_ratio",
            "queue_raw", "queue_norm", "emission_raw", "emission_norm",
            "queue_hotspot", "emission_hotspot", "edge_w_em", "edge_w_eff",
        ], self.flush_interval)
        self.upper_step_writer = _CsvBatchWriter(self.output_dir / "upper_step_metrics.csv", [
            "episode", "upper_step", "start_step", "end_step", "sim_time_start", "sim_time_end",
            "E_sum_before", "Q_sum_before", "P_hot_before", "E_sum_after", "Q_sum_after", "P_hot_after",
            "raw_lam", "delta_mean", "delta_std", "delta_min", "delta_max",
            "w_em_mean", "w_em_std", "w_em_min", "w_em_max",
            "w_eff_mean", "w_eff_std", "w_eff_min", "w_eff_max",
            "upper_value", "upper_log_prob", "upper_entropy", "upper_reward",
        ], self.flush_interval)
        self.training_lower_writer = _CsvBatchWriter(self.output_dir / "training_metrics_lower.csv", [
            "episode", "lower_update_idx", "total_env_steps", "batch_size", "policy_loss", "value_loss",
            "entropy_loss", "total_loss", "actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm",
            "return_mean", "return_std", "adv_mean", "adv_std",
        ], self.flush_interval)
        self.training_upper_writer = _CsvBatchWriter(self.output_dir / "training_metrics_upper.csv", [
            "episode", "upper_update_idx", "total_upper_steps", "batch_size", "policy_loss", "value_loss",
            "entropy_loss", "total_loss", "actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm",
            "return_mean", "return_std", "adv_mean", "adv_std",
        ], self.flush_interval)
        self.episode_summary_writer = _CsvBatchWriter(self.output_dir / "episode_summary.csv", [
            "episode", "steps", "total_global_reward", "avg_global_reward", "avg_Q_net", "avg_E_net",
            "lower_updates", "upper_updates", "lower_actor_loss", "lower_critic_loss", "upper_actor_loss", "upper_critic_loss",
            "episode_elapsed_sec", "total_elapsed_sec",
            "prof_env_step_sec", "prof_lower_act_sec", "prof_upper_act_sec",
            "prof_lower_update_sec", "prof_upper_update_sec", "prof_logging_sec",
        ], self.flush_interval)
        self.node_raw_writer = _CsvBatchWriter(self.output_dir / "tls_node_step_features_raw.csv", [
            "episode", "step", "simulation_time", "tls_id", "node_idx", "lane_id", "wave", "queue", "wait", "speed", "truck_ratio", "emission",
        ], self.flush_interval)
        if log_params:
            with (self.output_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
                json.dump(_jsonable(vars(config)), f, ensure_ascii=False, indent=2)

    def log_network_architecture(self, model):
        with (self.output_dir / "network_architecture.txt").open("w", encoding="utf-8") as f:
            f.write(str(model))

    @staticmethod
    def _adv_stats(batch: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not batch:
            return {"return_mean": 0.0, "return_std": 0.0, "adv_mean": 0.0, "adv_std": 0.0}
        returns = np.asarray(batch.get("returns", []), dtype=np.float32)
        advs = np.asarray(batch.get("advantages", []), dtype=np.float32)
        return {
            "return_mean": float(np.mean(returns)) if returns.size else 0.0,
            "return_std": float(np.std(returns)) if returns.size else 0.0,
            "adv_mean": float(np.mean(advs)) if advs.size else 0.0,
            "adv_std": float(np.std(advs)) if advs.size else 0.0,
        }

    def log_network_step(self, episode: int):
        if not self.log_network_step_flag or self.env is None:
            return
        row = {"episode": int(episode), **self.env.get_network_step_metrics()}
        self.network_step_writer.add_row(row)
        if self.log_edge_step_flag:
            edge_rows = []
            edge_w = getattr(self.env, "last_edge_weight_matrix", None)
            for i, r in enumerate(self.env.get_edge_step_metrics()):
                rr = {"episode": int(episode), **r}
                if edge_w is not None and i < len(edge_w):
                    rr["edge_w_em"] = _safe_float(edge_w[i, 0])
                    rr["edge_w_eff"] = _safe_float(edge_w[i, 1])
                edge_rows.append(rr)
            self.edge_step_writer.add_rows(edge_rows)

    def log_tls_step(self, episode: int, reward_dict=None, action_dict=None):
        if not self.log_tls_step_flag or self.env is None:
            return
        rows = [{"episode": int(episode), **r} for r in self.env.get_tls_step_metrics(reward_dict, action_dict)]
        self.tls_step_writer.add_rows(rows)

    def log_node_features(self, episode: int, step: int):
        if self.env is None or not self.log_node_raw_flag:
            return
        rows = [{"episode": int(episode), **r} for r in self.env.get_tls_node_feature_rows(mode="raw")]
        self.node_raw_writer.add_rows(rows)

    def log_upper_step(self, episode: int, upper_step: int, start_snapshot: Dict[str, Any], end_snapshot: Dict[str, Any], upper_out: Dict[str, Any], upper_reward: float, start_step: int, end_step: int):
        W = np.asarray(upper_out.get("weight_matrix", upper_out.get("W", np.zeros((1, 2)))), dtype=np.float32)
        raw_delta = np.asarray(_jsonable(upper_out.get("raw_delta", [])), dtype=np.float32)
        row = {
            "episode": int(episode), "upper_step": int(upper_step), "start_step": int(start_step), "end_step": int(end_step),
            "sim_time_start": float(start_snapshot.get("simulation_time", 0.0)), "sim_time_end": float(end_snapshot.get("simulation_time", 0.0)),
            "E_sum_before": _safe_float(start_snapshot.get("E_sum", start_snapshot.get("E_net", 0.0))),
            "Q_sum_before": _safe_float(start_snapshot.get("Q_sum", start_snapshot.get("Q_net", 0.0))),
            "P_hot_before": _safe_float(start_snapshot.get("P_hot", start_snapshot.get("H_net", 0.0))),
            "E_sum_after": _safe_float(end_snapshot.get("E_sum", end_snapshot.get("E_net", 0.0))),
            "Q_sum_after": _safe_float(end_snapshot.get("Q_sum", end_snapshot.get("Q_net", 0.0))),
            "P_hot_after": _safe_float(end_snapshot.get("P_hot", end_snapshot.get("H_net", 0.0))),
            "raw_lam": _safe_float(upper_out.get("raw_lam", 0.0)),
            "delta_mean": float(np.mean(raw_delta)) if raw_delta.size else 0.0,
            "delta_std": float(np.std(raw_delta)) if raw_delta.size else 0.0,
            "delta_min": float(np.min(raw_delta)) if raw_delta.size else 0.0,
            "delta_max": float(np.max(raw_delta)) if raw_delta.size else 0.0,
            "w_em_mean": float(np.mean(W[:, 0])) if W.ndim == 2 else 0.0,
            "w_em_std": float(np.std(W[:, 0])) if W.ndim == 2 else 0.0,
            "w_em_min": float(np.min(W[:, 0])) if W.ndim == 2 else 0.0,
            "w_em_max": float(np.max(W[:, 0])) if W.ndim == 2 else 0.0,
            "w_eff_mean": float(np.mean(W[:, 1])) if W.ndim == 2 else 0.0,
            "w_eff_std": float(np.std(W[:, 1])) if W.ndim == 2 else 0.0,
            "w_eff_min": float(np.min(W[:, 1])) if W.ndim == 2 else 0.0,
            "w_eff_max": float(np.max(W[:, 1])) if W.ndim == 2 else 0.0,
            "upper_value": _safe_float(upper_out.get("value", 0.0)),
            "upper_log_prob": _safe_float(upper_out.get("log_prob", 0.0)),
            "upper_entropy": _safe_float(upper_out.get("entropy", 0.0)),
            "upper_reward": float(upper_reward),
        }
        self.upper_step_writer.add_row(row)

    def log_lower_training(self, episode: int, lower_update_idx: int, total_env_steps: int, batch: Dict[str, Any], loss_dict: Dict[str, Any]):
        st = self._adv_stats(batch)
        self.training_lower_writer.add_row({"episode": episode, "lower_update_idx": lower_update_idx, "total_env_steps": total_env_steps, "batch_size": len(batch.get("actions", [])), **{k: _safe_float(loss_dict.get(k, 0.0)) for k in ["policy_loss","value_loss","entropy_loss","total_loss","actor_loss","critic_loss","actor_grad_norm","critic_grad_norm"]}, **st})

    def log_upper_training(self, episode: int, upper_update_idx: int, total_upper_steps: int, batch: Dict[str, Any], loss_dict: Dict[str, Any]):
        st = self._adv_stats(batch)
        self.training_upper_writer.add_row({"episode": episode, "upper_update_idx": upper_update_idx, "total_upper_steps": total_upper_steps, "batch_size": len(batch.get("returns", [])), **{k: _safe_float(loss_dict.get(k, 0.0)) for k in ["policy_loss","value_loss","entropy_loss","total_loss","actor_loss","critic_loss","actor_grad_norm","critic_grad_norm"]}, **st})

    def log_episode_summary(
        self,
        episode: int,
        env_stats: Dict[str, Any],
        lower_updates: int,
        upper_updates: int,
        last_lower_loss: Optional[Dict[str, Any]],
        last_upper_loss: Optional[Dict[str, Any]],
        episode_elapsed_sec: Optional[float] = None,
        total_elapsed_sec: Optional[float] = None,
        profile_stats: Optional[Dict[str, float]] = None,
    ):
        ps = profile_stats or {}
        self.episode_summary_writer.add_row({
            "episode": int(episode), **env_stats,
            "lower_updates": int(lower_updates), "upper_updates": int(upper_updates),
            "lower_actor_loss": _safe_float((last_lower_loss or {}).get("actor_loss", 0.0)),
            "lower_critic_loss": _safe_float((last_lower_loss or {}).get("critic_loss", 0.0)),
            "upper_actor_loss": _safe_float((last_upper_loss or {}).get("actor_loss", 0.0)),
            "upper_critic_loss": _safe_float((last_upper_loss or {}).get("critic_loss", 0.0)),
            "episode_elapsed_sec": _safe_float(episode_elapsed_sec, 0.0),
            "total_elapsed_sec": _safe_float(total_elapsed_sec, 0.0),
            "prof_env_step_sec": _safe_float(ps.get("env_step_sec", 0.0), 0.0),
            "prof_lower_act_sec": _safe_float(ps.get("lower_act_sec", 0.0), 0.0),
            "prof_upper_act_sec": _safe_float(ps.get("upper_act_sec", 0.0), 0.0),
            "prof_lower_update_sec": _safe_float(ps.get("lower_update_sec", 0.0), 0.0),
            "prof_upper_update_sec": _safe_float(ps.get("upper_update_sec", 0.0), 0.0),
            "prof_logging_sec": _safe_float(ps.get("logging_sec", 0.0), 0.0),
        })

    def log_vehicle_trip_rows(self, episode: int):
        return

    def log_episode_summary_raw(self, episode: int):
        return

    def flush(self):
        for w in vars(self).values():
            if isinstance(w, _CsvBatchWriter):
                w.flush()

    def close(self):
        for w in vars(self).values():
            if isinstance(w, _CsvBatchWriter):
                w.close()
