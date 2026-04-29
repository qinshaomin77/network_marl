# -*- coding: utf-8 -*-
"""
buffer.py

路网版双缓冲结构，方案 B：彻底清理版：
1. A2CBuffer
   - 给下层共享参数局部控制器使用
   - 按 control step 存全网数据
   - 计算每个 tls 的 returns / advantages
   - get() 时展开成单路口样本 batch
   - 不再保留 z_k_dict / z_list

2. PPOBuffer
   - 给上层网络协调器使用
   - 按 upper-step 存网络级 transition
   - 保存 old_log_prob
   - 支持 returns / advantages（可选 GAE）
   - 不再保留 z_k_dict_list
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import copy

import numpy as np
import torch


# =============================================================================
# 工具函数
# =============================================================================

def _clone_item(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, dict):
        return {k: _clone_item(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_item(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_item(v) for v in x)
    if isinstance(x, (int, float, str, bool, type(None))):
        return x
    return copy.deepcopy(x)


def _to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


# =============================================================================
# Base
# =============================================================================

class _BaseBuffer:
    def clear(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def is_ready(self) -> bool:
        return not self.is_empty


# =============================================================================
# 下层：A2CBuffer
# =============================================================================

class A2CBuffer(_BaseBuffer):
    """
    下层共享参数 A2C 缓冲器
    """

    def __init__(
        self,
        gamma: float = 0.99,
        rollout_size: Optional[int] = None,
    ):
        self.gamma = float(gamma)
        self.rollout_size = rollout_size
        self.clear()

    def clear(self):
        self.per_tls_obs_steps: List[Dict[str, Dict[str, Any]]] = []
        self.action_dict_steps: List[Dict[str, int]] = []
        self.reward_dict_steps: List[Dict[str, float]] = []
        self.done_steps: List[bool] = []
        self.value_dict_steps: List[Dict[str, float]] = []

        self.log_prob_dict_steps: List[Optional[Dict[str, Any]]] = []
        self.entropy_dict_steps: List[Optional[Dict[str, Any]]] = []
        self.upper_weight_dict_steps: List[Optional[Dict[str, Any]]] = []

        self.critic_hidden_steps: List[Optional[Any]] = []
        self.tls_order_steps: List[Optional[List[str]]] = []

        self.return_dict_steps: List[Optional[Dict[str, float]]] = []
        self.advantage_dict_steps: List[Optional[Dict[str, float]]] = []

    def __len__(self):
        return len(self.per_tls_obs_steps)

    @property
    def is_full(self) -> bool:
        if self.rollout_size is None:
            return False
        return len(self) >= self.rollout_size

    def add(
        self,
        per_tls_obs: Dict[str, Dict[str, Any]],
        action_dict: Dict[str, int],
        reward_dict: Dict[str, float],
        done: bool,
        value_dict: Dict[str, Any],
        log_prob_dict: Optional[Dict[str, Any]] = None,
        entropy_dict: Optional[Dict[str, Any]] = None,
        upper_weight_dict: Optional[Dict[str, Any]] = None,
        critic_hidden: Optional[Any] = None,
        tls_order: Optional[List[str]] = None,
    ):
        if not isinstance(per_tls_obs, dict) or len(per_tls_obs) == 0:
            raise ValueError("A2CBuffer.add(): per_tls_obs must be a non-empty dict")

        tls_ids = list(per_tls_obs.keys())
        for name, d in [
            ("action_dict", action_dict),
            ("reward_dict", reward_dict),
            ("value_dict", value_dict),
        ]:
            if not isinstance(d, dict):
                raise TypeError(f"A2CBuffer.add(): {name} must be a dict")
            missing = [tid for tid in tls_ids if tid not in d]
            if missing:
                raise KeyError(f"A2CBuffer.add(): {name} missing tls_ids: {missing}")

        self.per_tls_obs_steps.append(_clone_item(per_tls_obs))
        self.action_dict_steps.append(_clone_item(action_dict))
        self.reward_dict_steps.append({tid: _to_float(v) for tid, v in reward_dict.items()})
        self.done_steps.append(bool(done))
        self.value_dict_steps.append({tid: _to_float(v) for tid, v in value_dict.items()})

        self.log_prob_dict_steps.append(_clone_item(log_prob_dict) if log_prob_dict is not None else None)
        self.entropy_dict_steps.append(_clone_item(entropy_dict) if entropy_dict is not None else None)
        self.upper_weight_dict_steps.append(_clone_item(upper_weight_dict) if upper_weight_dict is not None else None)

        self.critic_hidden_steps.append(_clone_item(critic_hidden) if critic_hidden is not None else None)
        self.tls_order_steps.append(list(tls_order) if tls_order is not None else None)

        self.return_dict_steps.append(None)
        self.advantage_dict_steps.append(None)

    def compute_returns(
        self,
        last_value_dict: Optional[Dict[str, Any]] = None,
    ):
        T = len(self)
        if T == 0:
            raise RuntimeError("A2CBuffer.compute_returns(): buffer is empty")

        if last_value_dict is None:
            last_value_dict = {}

        self.return_dict_steps = [dict() for _ in range(T)]
        self.advantage_dict_steps = [dict() for _ in range(T)]

        all_tls_ids = set()
        for obs_dict in self.per_tls_obs_steps:
            all_tls_ids.update(obs_dict.keys())

        for tls_id in all_tls_ids:
            next_return = float(last_value_dict.get(tls_id, 0.0))
            for t in reversed(range(T)):
                if tls_id not in self.reward_dict_steps[t]:
                    continue

                done = bool(self.done_steps[t])
                reward = float(self.reward_dict_steps[t][tls_id])
                value = float(self.value_dict_steps[t][tls_id])

                if done:
                    next_return = 0.0

                ret = reward + self.gamma * next_return
                adv = ret - value

                self.return_dict_steps[t][tls_id] = float(ret)
                self.advantage_dict_steps[t][tls_id] = float(adv)
                next_return = ret

    def get(self) -> Dict[str, Any]:
        if len(self) == 0:
            raise RuntimeError("A2CBuffer.get(): buffer is empty")
        if any(x is None for x in self.return_dict_steps) or any(x is None for x in self.advantage_dict_steps):
            raise RuntimeError("A2CBuffer.get(): returns/advantages not computed yet")

        obs_list = []
        actions = []
        returns = []
        advantages = []

        tls_id_list = []
        step_index_list = []

        log_prob_list = []
        entropy_list = []
        upper_weight_list = []

        # 新增：为每个样本收集对应的 critic hidden slice
        critic_h_list = []
        critic_c_list = []

        has_log_prob = any(x is not None for x in self.log_prob_dict_steps)
        has_entropy = any(x is not None for x in self.entropy_dict_steps)
        has_upper_weight = any(x is not None for x in self.upper_weight_dict_steps)
        has_critic_hidden = any(x is not None for x in self.critic_hidden_steps)

        for t in range(len(self)):
            obs_dict = self.per_tls_obs_steps[t]
            action_dict = self.action_dict_steps[t]
            ret_dict = self.return_dict_steps[t]
            adv_dict = self.advantage_dict_steps[t]

            logp_dict = self.log_prob_dict_steps[t]
            ent_dict = self.entropy_dict_steps[t]
            uw_dict = self.upper_weight_dict_steps[t]

            c_hidden = self.critic_hidden_steps[t]
            tls_order = self.tls_order_steps[t]

            # 关键：
            # hidden 的 batch 维顺序必须使用 rollout 当时真实保存的 tls_order，
            # 不能自己 sorted() 猜顺序
            tls_id_to_idx = None
            if has_critic_hidden:
                if c_hidden is not None and tls_order is None:
                    raise RuntimeError(
                        f"A2CBuffer.get(): tls_order is missing for step {t} with critic_hidden"
                    )
                if c_hidden is not None:
                    tls_id_to_idx = {tid: idx for idx, tid in enumerate(tls_order)}

            for tls_id in obs_dict.keys():
                obs_list.append(_clone_item(obs_dict[tls_id]))
                actions.append(int(action_dict[tls_id]))
                returns.append(float(ret_dict[tls_id]))
                advantages.append(float(adv_dict[tls_id]))

                tls_id_list.append(tls_id)
                step_index_list.append(t)

                if has_log_prob:
                    log_prob_list.append(
                        None if logp_dict is None else _clone_item(logp_dict.get(tls_id, None))
                    )
                if has_entropy:
                    entropy_list.append(
                        None if ent_dict is None else _clone_item(ent_dict.get(tls_id, None))
                    )
                if has_upper_weight:
                    upper_weight_list.append(
                        None if uw_dict is None else _clone_item(uw_dict.get(tls_id, None))
                    )

                # 新增：抽取当前样本对应的 critic hidden slice
                if has_critic_hidden:
                    if c_hidden is None:
                        raise RuntimeError(
                            f"A2CBuffer.get(): critic_hidden missing on step {t} while has_critic_hidden=True"
                        )
                    if tls_id_to_idx is None or tls_id not in tls_id_to_idx:
                        raise RuntimeError(
                            f"A2CBuffer.get(): tls_id={tls_id} not found in tls_order for step {t}"
                        )

                    idx = tls_id_to_idx[tls_id]

                    # c_hidden = (h, c), 其中 h/c shape = [B, H]
                    critic_h_list.append(c_hidden[0][idx].clone())  # [H]
                    critic_c_list.append(c_hidden[1][idx].clone())  # [H]

        batch = {
            "obs_list": obs_list,
            "actions": np.asarray(actions, dtype=np.int64),
            "returns": np.asarray(returns, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
            "tls_id_list": tls_id_list,
            "step_index_list": step_index_list,
        }

        if has_log_prob:
            batch["log_prob_list"] = log_prob_list
        if has_entropy:
            batch["entropy_list"] = entropy_list
        if has_upper_weight:
            batch["upper_weight_list"] = upper_weight_list

        # 新增：把 critic hidden 打包成 [N, H]
        if has_critic_hidden:
            if len(critic_h_list) == 0 or len(critic_c_list) == 0:
                raise RuntimeError(
                    "A2CBuffer.get(): critic_hidden is enabled but no hidden slices were collected"
                )
            batch["critic_hidden_h"] = torch.stack(critic_h_list, dim=0)  # [N, H]
            batch["critic_hidden_c"] = torch.stack(critic_c_list, dim=0)  # [N, H]

        return batch


# =============================================================================
# 上层：PPOBuffer
# =============================================================================

class PPOBuffer(_BaseBuffer):
    """
    上层 PPO 缓冲器
    """

    def __init__(
        self,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        use_gae: bool = True,
        rollout_size: Optional[int] = None,
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.use_gae = bool(use_gae)
        self.rollout_size = rollout_size
        self.clear()

    def clear(self):
        self.network_obs_list: List[Dict[str, Any]] = []
        self.raw_lam_list: List[Any] = []
        self.raw_delta_list: List[Any] = []
        self.old_log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

        self.W_list: List[Optional[Any]] = []
        self.upper_metrics_list: List[Optional[Dict[str, Any]]] = []

        self.returns: Optional[np.ndarray] = None
        self.advantages: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.network_obs_list)

    @property
    def is_full(self) -> bool:
        if self.rollout_size is None:
            return False
        return len(self) >= self.rollout_size

    def add(
        self,
        network_obs: Dict[str, Any],
        raw_lam: Any,
        raw_delta: Any,
        old_log_prob: Any,
        reward: Any,
        done: bool,
        value: Any,
        W: Optional[Any] = None,
        upper_metrics: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(network_obs, dict):
            raise TypeError("PPOBuffer.add(): network_obs must be dict")

        self.network_obs_list.append(_clone_item(network_obs))
        self.raw_lam_list.append(_clone_item(raw_lam))
        self.raw_delta_list.append(_clone_item(raw_delta))
        self.old_log_probs.append(_to_float(old_log_prob))
        self.rewards.append(_to_float(reward))
        self.dones.append(bool(done))
        self.values.append(_to_float(value))

        self.W_list.append(_clone_item(W) if W is not None else None)
        self.upper_metrics_list.append(_clone_item(upper_metrics) if upper_metrics is not None else None)

        self.returns = None
        self.advantages = None

    def compute_returns(
        self,
        last_value: float = 0.0,
        use_gae: Optional[bool] = None,
    ):
        T = len(self)
        if T == 0:
            raise RuntimeError("PPOBuffer.compute_returns(): buffer is empty")

        use_gae = self.use_gae if use_gae is None else bool(use_gae)

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        returns = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)

        if use_gae:
            gae = 0.0
            next_value = float(last_value)
            for t in reversed(range(T)):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_value * mask - values[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae

                advantages[t] = gae
                returns[t] = gae + values[t]
                next_value = values[t]
        else:
            next_return = float(last_value)
            for t in reversed(range(T)):
                if dones[t] > 0.5:
                    next_return = 0.0
                ret = rewards[t] + self.gamma * next_return
                adv = ret - values[t]

                returns[t] = ret
                advantages[t] = adv
                next_return = ret

        self.returns = returns
        self.advantages = advantages

    def get(self) -> Dict[str, Any]:
        if len(self) == 0:
            raise RuntimeError("PPOBuffer.get(): buffer is empty")
        if self.returns is None or self.advantages is None:
            raise RuntimeError("PPOBuffer.get(): returns/advantages not computed yet")

        batch = {
            "network_obs_list": _clone_item(self.network_obs_list),
            "raw_lam_list": _clone_item(self.raw_lam_list),
            "raw_delta_list": _clone_item(self.raw_delta_list),
            "old_log_probs": np.asarray(self.old_log_probs, dtype=np.float32),
            "returns": np.asarray(self.returns, dtype=np.float32),
            "advantages": np.asarray(self.advantages, dtype=np.float32),
        }

        if any(x is not None for x in self.W_list):
            batch["W_list"] = _clone_item(self.W_list)
        if any(x is not None for x in self.upper_metrics_list):
            batch["upper_metrics_list"] = _clone_item(self.upper_metrics_list)

        return batch
