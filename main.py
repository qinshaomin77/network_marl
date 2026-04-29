# -*- coding: utf-8 -*-
"""
main.py
"""

from __future__ import annotations

import random
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from config import TrafficConfig
from env import NetworkTrafficEnv
from buffer import A2CBuffer, PPOBuffer
from logger import TrafficLogger
from utils import build_sumo_cmd
from model import HierarchicalTrafficModel


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log(f"Global seed set to {seed}")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_episode_seeds(global_seed: int, num_episodes: int) -> List[int]:
    rng = np.random.RandomState(global_seed)
    return rng.randint(0, 2**31 - 1, size=num_episodes).tolist()


def save_checkpoint(ckpt_path: Path, model: HierarchicalTrafficModel, meta: Dict[str, Any]) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.get_state_dicts(), "meta": meta}, ckpt_path)


def load_checkpoint(ckpt_path: Path, model: HierarchicalTrafficModel, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location=map_location or get_device())
    model.load_state_dicts(payload["model"])
    return payload.get("meta", {})


def current_weight_dict_from_matrix(tls_ids: List[str], weight_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    return {tl_id: np.asarray(weight_matrix[i], dtype=np.float32) for i, tl_id in enumerate(tls_ids)}


def init_prev_policy_dict(tls_ids: List[str], a_max: int) -> Dict[str, np.ndarray]:
    base = np.full(a_max, 1.0 / max(a_max, 1), dtype=np.float32)
    return {tl_id: base.copy() for tl_id in tls_ids}

def _detach_hidden(hidden):
    if hidden is None:
        return None
    h, c = hidden
    return (h.detach().cpu().clone(), c.detach().cpu().clone())

def build_neighbor_policy_pack(env: NetworkTrafficEnv, prev_policy_dict: Dict[str, np.ndarray], a_max: int) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    directions = list(env.config.NEIGHBOR_DIRECTIONS)
    for tl_id in env.static.tls_ids:
        p_dir = np.zeros((len(directions), a_max), dtype=np.float32)
        m_dir = np.zeros((len(directions), a_max), dtype=np.float32)
        for d_idx, d in enumerate(directions):
            nbr_id = env.static.tls_neighbors.get(tl_id, {}).get(d, None)
            if nbr_id is None:
                continue
            probs = np.asarray(prev_policy_dict.get(nbr_id, np.full(a_max, 1.0 / max(a_max, 1))), dtype=np.float32)
            probs = probs[:a_max]
            p_dir[d_idx, :len(probs)] = probs
            m_dir[d_idx, :len(probs)] = 1.0
        out[tl_id] = {
            "neighbor_policy_dir": p_dir,
            "neighbor_policy_action_mask": m_dir,
        }
    return out


def inject_neighbor_policy_into_obs(per_tls_obs: Dict[str, Dict[str, Any]], neighbor_policy_pack: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
    for tl_id, obs in per_tls_obs.items():
        pack = neighbor_policy_pack.get(tl_id, None)
        if pack is None:
            continue
        obs["neighbor_policy_dir"] = pack["neighbor_policy_dir"]
        obs["neighbor_policy_action_mask"] = pack["neighbor_policy_action_mask"]
    return per_tls_obs


def compute_upper_reward(window_global_rewards, start_snapshot, end_snapshot, current_W, prev_W, config: TrafficConfig) -> float:
    """Upper reward: edge-level normalized queue sum + emission sum + hotspot penalty.

    The upper reward no longer mixes lower/global rewards. It evaluates the
    network state at the end of the upper window.
    """
    q_sum = float(end_snapshot.get("Q_sum", end_snapshot.get("Q_net", 0.0)))
    e_sum = float(end_snapshot.get("E_sum", end_snapshot.get("E_net", 0.0)))
    p_hot = float(end_snapshot.get("P_hot", end_snapshot.get("H_net", 0.0)))

    reward = -(
        float(getattr(config, "UPPER_BETA_QUEUE", 0.40)) * q_sum
        + float(getattr(config, "UPPER_BETA_EMISSION", 0.40)) * e_sum
        + float(getattr(config, "UPPER_BETA_HOTSPOT", 0.20)) * p_hot
    )
    return float(reward)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--log-params", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--deterministic-upper", action="store_true")
    parser.add_argument("--deterministic-lower", action="store_true")
    return parser.parse_args()


def build_run_dirs(config: TrafficConfig) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(config.OUTPUT_DIR / f"{config.NETWORK_NAME}_{timestamp}")
    log_dir = ensure_dir(run_dir / "logs")
    simulation_dir = ensure_dir(run_dir / "simulations")
    model_dir = ensure_dir(run_dir / "models")
    tb_dir = ensure_dir(log_dir / "tensorboard")
    return {
        "run_dir": run_dir,
        "log_dir": log_dir,
        "model_dir": model_dir,
        "simulation_dir": simulation_dir,
        "tb_dir": tb_dir,
    }


def build_components(config: TrafficConfig, run_dirs: Dict[str, Path], args):
    device = get_device()
    log(f">>> device = {device}")
    env = NetworkTrafficEnv(config)
    log(">>> env 创建完成")
    model = HierarchicalTrafficModel(config=config, device=device)
    log(">>> model 创建完成")

    lower_buffer = A2CBuffer(
        gamma=float(getattr(config, "LOWER_GAMMA", 0.99)),
        rollout_size=int(getattr(config, "N_STEP", 10)),
    )
    upper_buffer = PPOBuffer(
        gamma=float(getattr(config, "UPPER_GAMMA", 0.98)),
        gae_lambda=float(getattr(config, "UPPER_GAE_LAMBDA", 0.95)),
        use_gae=True,
        rollout_size=int(getattr(config, "UPPER_ROLLOUT_SIZE", 32)),
    )

    logger = TrafficLogger(
        config=config,
        env=env,
        output_dir=run_dirs["log_dir"],
        flush_interval=int(getattr(config, "LOG_FLUSH_INTERVAL", 100)),
        log_params=bool(args.log_params),
    )
    logger.log_network_architecture(model)

    tb_writer = None
    use_tb = bool(args.tensorboard) or bool(getattr(config, "USE_TENSORBOARD", False))
    if use_tb and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(run_dirs["tb_dir"]))

    return {
        "device": device,
        "env": env,
        "model": model,
        "lower_buffer": lower_buffer,
        "upper_buffer": upper_buffer,
        "logger": logger,
        "tb_writer": tb_writer,
    }


def run_one_episode(
    episode: int,
    episode_seed: int,
    config: TrafficConfig,
    env: NetworkTrafficEnv,
    model: HierarchicalTrafficModel,
    lower_buffer: A2CBuffer,
    upper_buffer: PPOBuffer,
    logger: TrafficLogger,
    tb_writer: Optional["SummaryWriter"],
    total_env_steps: int,
    total_upper_steps: int,
    lower_update_idx: int,
    upper_update_idx: int,
    deterministic_upper: bool = False,
    deterministic_lower: bool = False,
) -> Dict[str, Any]:
    config.SUMO_SEED = int(episode_seed)
    env.sumo_cmd = build_sumo_cmd(config, gui=config.SUMO_GUI)

    obs, _ = env.reset(seed=episode_seed)
    per_tls_obs = obs["per_tls_obs"]
    network_obs = obs["network_obs"]

    tls_ids = list(env.static.tls_ids)
    actor_hidden, critic_hidden = model.init_hidden_batched(num_tls=len(tls_ids))

    prev_policy_dict = init_prev_policy_dict(tls_ids, int(config.A_MAX))
    lower_buffer.clear()
    # ← 改动 A：不再在 episode 开头清空 upper_buffer
    #   upper_buffer 跨 episode 累积，只在 is_full 触发更新后才 clear

    done = False
    episode_step_count = 0
    episode_upper_step_count = 0
    episode_lower_update_count = 0   # ← 新增
    episode_upper_update_count = 0   # ← 新增（可选，如果也想跳过上层冷启动）

    current_upper_out: Optional[Dict[str, Any]] = None
    current_upper_W_np = np.full((len(tls_ids), 2), 0.5, dtype=np.float32)
    prev_upper_W_np: Optional[np.ndarray] = None
    upper_window_start_snapshot: Optional[Dict[str, Any]] = None
    upper_window_global_rewards: List[float] = []

    last_lower_loss: Optional[Dict[str, Any]] = None
    last_upper_loss: Optional[Dict[str, Any]] = None

    while not done:
        if episode_step_count % int(config.UPPER_K) == 0:
            upper_window_start_snapshot = env.get_upper_step_snapshot()
            current_upper_out = model.act_upper(network_obs=network_obs, deterministic=deterministic_upper)
            current_upper_W_np = current_upper_out["W"].detach().cpu().numpy().astype(np.float32)
            env.set_upper_weights(current_upper_W_np)
            upper_window_global_rewards = []

        neighbor_policy_pack = build_neighbor_policy_pack(env, prev_policy_dict, int(config.A_MAX))
        per_tls_obs = inject_neighbor_policy_into_obs(per_tls_obs, neighbor_policy_pack)

        pre_forward_critic_hidden = _detach_hidden(critic_hidden)
        tls_order = list(tls_ids)

        lower_out = model.act_lower_all_batched(
            per_tls_obs=per_tls_obs,
            tls_ids=tls_ids,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            upper_weight_matrix=current_upper_W_np,
            deterministic=deterministic_lower,
        )

        action_dict = lower_out["action_dict"]
        log_prob_dict = lower_out["log_prob_dict"]
        entropy_dict = lower_out["entropy_dict"]
        value_dict = lower_out["value_dict"]
        actor_hidden = lower_out["actor_hidden"]
        critic_hidden = lower_out["critic_hidden"]
        prev_policy_dict = lower_out["policy_dict"]

        next_obs, reward_dict, terminated, truncated, info = env.step(action_dict)
        done = bool(terminated or truncated)

        next_per_tls_obs = next_obs["per_tls_obs"]
        next_network_obs = next_obs["network_obs"]
        upper_window_global_rewards.append(float(info.get("global_reward", 0.0)))

        current_weight_dict = current_weight_dict_from_matrix(tls_ids, current_upper_W_np)
        lower_buffer.add(
            per_tls_obs=per_tls_obs,
            action_dict=action_dict,
            reward_dict=reward_dict,
            done=done,
            value_dict=value_dict,
            log_prob_dict=log_prob_dict,
            entropy_dict=entropy_dict,
            upper_weight_dict=current_weight_dict,
            critic_hidden=pre_forward_critic_hidden,
            tls_order=tls_order,
        )

        logger.log_network_step(episode)
        logger.log_tls_step(episode, reward_dict=reward_dict, action_dict=action_dict)
        logger.log_node_features(episode, step=int(env.current_step))

        if lower_buffer.is_full or done:
            if done:
                last_value_dict = {tl_id: 0.0 for tl_id in tls_ids}
            else:
                last_value_dict = model.get_lower_bootstrap_values_batched(
                    per_tls_obs=next_per_tls_obs,
                    tls_ids=tls_ids,
                    critic_hidden=critic_hidden,
                    upper_weight_matrix=current_upper_W_np,
                )

            lower_buffer.compute_returns(last_value_dict=last_value_dict)
            lower_batch = lower_buffer.get()

            episode_lower_update_count += 1
            
            # ← 新增：跳过 episode 第一次 update（hidden 冷启动）
            if episode_lower_update_count > 1:
                last_lower_loss = model.update_lower_batched(lower_batch)
                lower_update_idx += 1
                
                logger.log_lower_training(
                    episode=episode,
                    lower_update_idx=lower_update_idx,
                    total_env_steps=total_env_steps + 1,
                    batch=lower_batch,
                    loss_dict=last_lower_loss,
                )
            
            lower_buffer.clear()

        if ((episode_step_count + 1) % int(config.UPPER_K) == 0) or done:
            end_snapshot = env.get_upper_step_snapshot()

            upper_reward = compute_upper_reward(
                window_global_rewards=upper_window_global_rewards,
                start_snapshot=upper_window_start_snapshot or {},
                end_snapshot=end_snapshot,
                current_W=current_upper_W_np,
                prev_W=prev_upper_W_np,
                config=config,
            )

            assert current_upper_out is not None
            old_log_prob = float(
                current_upper_out["log_prob"].item()
                if isinstance(current_upper_out["log_prob"], torch.Tensor)
                else current_upper_out["log_prob"]
            )

            upper_buffer.add(
                network_obs=network_obs,
                raw_lam=current_upper_out["raw_lam"],
                raw_delta=current_upper_out["raw_delta"],
                old_log_prob=old_log_prob,
                reward=upper_reward,
                done=done,
                value=current_upper_out["value"],
                W=current_upper_W_np,
                upper_metrics=end_snapshot,
            )

            logger.log_upper_step(
                episode=episode,
                upper_step=episode_upper_step_count + 1,
                start_snapshot=upper_window_start_snapshot or {},
                end_snapshot=end_snapshot,
                upper_out=current_upper_out,
                upper_reward=upper_reward,
                start_step=int((upper_window_start_snapshot or {}).get("step", max(0, env.current_step - int(config.UPPER_K)))),
                end_step=int(end_snapshot.get("step", env.current_step)),
            )

            total_upper_steps += 1
            episode_upper_step_count += 1
            prev_upper_W_np = current_upper_W_np.copy()
            if total_upper_steps % 20 == 0:
                log(f"  upper_steps={total_upper_steps}, ep={episode}") 

            # ← 改动 B：只在 buffer 满时触发上层更新，episode 结束不再强制触发
            if upper_buffer.is_full:
                # 显式判断最后一个样本的 done 状态决定 bootstrap value
                if len(upper_buffer.dones) > 0 and upper_buffer.dones[-1]:
                    last_upper_value = 0.0
                else:
                    last_upper_value = model.get_upper_bootstrap_value(next_network_obs)
                log(f"  [upper update] buffer full, size={len(upper_buffer)}, ep={episode}, total_upper_steps={total_upper_steps}")
                upper_buffer.compute_returns(last_value=last_upper_value)
                upper_batch = upper_buffer.get()

                last_upper_loss = model.update_upper(upper_batch)
                upper_update_idx += 1

                logger.log_upper_training(
                    episode=episode,
                    upper_update_idx=upper_update_idx,
                    total_upper_steps=total_upper_steps,
                    batch=upper_batch,
                    loss_dict=last_upper_loss,
                )
                upper_buffer.clear()

        per_tls_obs = next_per_tls_obs
        network_obs = next_network_obs
        total_env_steps += 1
        episode_step_count += 1

        if tb_writer is not None:
            tb_writer.add_scalar("env/global_reward_step", float(info.get("global_reward", 0.0)), total_env_steps)
            tb_writer.add_scalar("env/E_net_step", float(info.get("upper_metrics", {}).get("E_net", 0.0)), total_env_steps)
            tb_writer.add_scalar("env/Q_net_step", float(info.get("upper_metrics", {}).get("Q_net", 0.0)), total_env_steps)

    env_stats = env.get_episode_statistics()

    logger.log_episode_summary(
        episode=episode,
        env_stats=env_stats,
        lower_updates=lower_update_idx,
        upper_updates=upper_update_idx,
        last_lower_loss=last_lower_loss,
        last_upper_loss=last_upper_loss,
    )
    logger.log_vehicle_trip_rows(episode=episode)
    logger.log_episode_summary_raw(episode=episode)
    logger.flush()

    if tb_writer is not None:
        tb_writer.add_scalar("episode/avg_global_reward", float(env_stats.get("avg_global_reward", 0.0)), episode)
        tb_writer.add_scalar("episode/avg_Q_net", float(env_stats.get("avg_Q_net", 0.0)), episode)
        tb_writer.add_scalar("episode/avg_E_net", float(env_stats.get("avg_E_net", 0.0)), episode)

    return {
        "env_stats": env_stats,
        "total_env_steps": total_env_steps,
        "total_upper_steps": total_upper_steps,
        "lower_update_idx": lower_update_idx,
        "upper_update_idx": upper_update_idx,
        "last_lower_loss": last_lower_loss,
        "last_upper_loss": last_upper_loss,
    }


def train(args):
    set_global_seed(args.seed)

    config = TrafficConfig()
    config.SUMO_GUI = bool(args.gui)

    if args.episodes is not None and args.episodes > 0:
        config.NUM_EPISODES = int(args.episodes)

    episode_seeds = generate_episode_seeds(args.seed, config.NUM_EPISODES)
    log(">>> 构建目录...")
    run_dirs = build_run_dirs(config)
    config.LOG_DIR = run_dirs["log_dir"]
    config.MODEL_DIR = run_dirs["model_dir"]
    config.SIM_DIR = run_dirs["simulation_dir"]

    log(">>> 构建组件（env/model/buffer）...")
    components = build_components(config, run_dirs, args)
    log(">>> 组件构建完成")

    env: NetworkTrafficEnv = components["env"]
    model: HierarchicalTrafficModel = components["model"]
    lower_buffer: A2CBuffer = components["lower_buffer"]
    upper_buffer: PPOBuffer = components["upper_buffer"]
    logger: TrafficLogger = components["logger"]
    tb_writer = components["tb_writer"]

    best_reward = float("-inf")
    total_env_steps = 0
    total_upper_steps = 0
    lower_update_idx = 0
    upper_update_idx = 0

    if args.resume:
        ckpt_meta = load_checkpoint(Path(args.resume), model)
        log(f"Resumed from checkpoint: {args.resume}")
        if ckpt_meta:
            log(f"Checkpoint meta: {ckpt_meta}")

    try:
        for episode in range(1, config.NUM_EPISODES + 1):
            log(f">>> Episode {episode}/{config.NUM_EPISODES} 开始...")
            ep_summary = run_one_episode(
                episode=episode,
                episode_seed=int(episode_seeds[episode - 1]),
                config=config,
                env=env,
                model=model,
                lower_buffer=lower_buffer,
                upper_buffer=upper_buffer,
                logger=logger,
                tb_writer=tb_writer,
                total_env_steps=total_env_steps,
                total_upper_steps=total_upper_steps,
                lower_update_idx=lower_update_idx,
                upper_update_idx=upper_update_idx,
                deterministic_upper=bool(args.deterministic_upper),
                deterministic_lower=bool(args.deterministic_lower),
            )

            env_stats = ep_summary["env_stats"]
            total_env_steps = int(ep_summary["total_env_steps"])
            total_upper_steps = int(ep_summary["total_upper_steps"])
            lower_update_idx = int(ep_summary["lower_update_idx"])
            upper_update_idx = int(ep_summary["upper_update_idx"])

            avg_reward = float(env_stats.get("avg_global_reward", float("-inf")))

            if avg_reward > best_reward:
                best_reward = avg_reward
                save_checkpoint(
                    ckpt_path=ensure_dir(config.MODEL_DIR / "best") / "checkpoint_best.pt",
                    model=model,
                    meta={"episode": episode, "best_reward": best_reward, "type": "best"},
                )

            if episode % int(getattr(config, "SAVE_INTERVAL", 20)) == 0:
                save_checkpoint(
                    ckpt_path=ensure_dir(config.MODEL_DIR / "latest") / f"checkpoint_ep{episode}.pt",
                    model=model,
                    meta={"episode": episode, "avg_global_reward": avg_reward, "type": "periodic"},
                )

        log("Training finished.")

    except KeyboardInterrupt:
        log("Training interrupted by user.", level="WARN")
    except Exception as e:
        log(f"Training crashed: {e}", level="ERROR")
        traceback.print_exc()
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)