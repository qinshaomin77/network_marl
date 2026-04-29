# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

from node_model import LowerA2CAgent
from network_model import UpperPPOAgent

# Thin facade around lower/upper agents for main.py.
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def current_weight_dict_from_matrix(tls_ids: List[str], weight_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    return {tl_id: np.asarray(weight_matrix[i], dtype=np.float32) for i, tl_id in enumerate(tls_ids)}

class HierarchicalTrafficModel:
    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or get_device()
        # Lower: local intersection control (A2C), Upper: network coordination (PPO).
        self.lower_agent = LowerA2CAgent(config=self.config, device=self.device)
        self.upper_agent = UpperPPOAgent(config=self.config, device=self.device)

    def init_hidden_dicts(self, tls_ids: List[str], batch_size: int = 1):
        actor_hidden_dict, critic_hidden_dict = {}, {}
        for tl_id in tls_ids:
            ah, ch = self.lower_agent.init_hidden(batch_size=batch_size)
            actor_hidden_dict[tl_id] = ah
            critic_hidden_dict[tl_id] = ch
        return actor_hidden_dict, critic_hidden_dict

    def init_hidden_batched(self, num_tls: int):
        actor_h = self.lower_agent.actor.lstm.init_hidden(self.device, batch_size=num_tls)
        critic_h = self.lower_agent.critic.lstm.init_hidden(self.device, batch_size=num_tls)
        return actor_h, critic_h

    @staticmethod
    def validate_lower_obs_schema(obs_i: Dict[str, Any]) -> None:
        required = [
            "node_features", "A_same", "A_diff", "lane_exist_mask",
            "neighbor_dir_mask", "neighbor_policy_dir", "neighbor_policy_action_mask",
            "action_mask",
        ]
        missing = [k for k in required if k not in obs_i]
        if missing:
            raise KeyError(f"Lower obs missing required keys: {missing}")

    def act_upper(self, network_obs: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        out = self.upper_agent.act_upper(network_obs=network_obs, deterministic=deterministic)
        if isinstance(out.get("weight_matrix"), torch.Tensor):
            out["W"] = out["weight_matrix"]
            out["weight_matrix"] = out["weight_matrix"].detach().cpu().numpy().astype(np.float32)
        elif "weight_matrix" in out:
            out["W"] = torch.tensor(out["weight_matrix"], dtype=torch.float32, device=self.device)
        return out

    def act_lower_all_batched(
        self,
        per_tls_obs: Dict[str, Dict[str, Any]],
        tls_ids: List[str],
        actor_hidden: Tuple[torch.Tensor, torch.Tensor],
        critic_hidden: Tuple[torch.Tensor, torch.Tensor],
        upper_weight_matrix: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        for tl_id in tls_ids:
            self.validate_lower_obs_schema(per_tls_obs[tl_id])
        if upper_weight_matrix is None:
            upper_weight_matrix = np.full((len(tls_ids), 2), 0.5, dtype=np.float32)
        uw = torch.tensor(upper_weight_matrix, dtype=torch.float32, device=self.device)
        return self.lower_agent.act_all_batched(
            per_tls_obs=per_tls_obs,
            tls_ids=tls_ids,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            upper_weight=uw,
            deterministic=deterministic,
        )

    def get_lower_bootstrap_values_batched(self, per_tls_obs, tls_ids, critic_hidden, upper_weight_matrix=None) -> Dict[str, float]:
        if upper_weight_matrix is None:
            upper_weight_matrix = np.full((len(tls_ids), 2), 0.5, dtype=np.float32)
        uw = torch.tensor(upper_weight_matrix, dtype=torch.float32, device=self.device)
        return self.lower_agent.get_bootstrap_values_batched(per_tls_obs, tls_ids, critic_hidden, uw)

    def get_upper_bootstrap_value(self, network_obs: Dict[str, Any]) -> float:
        out = self.upper_agent.act_upper(network_obs=network_obs, deterministic=True)
        v = out["value"]
        return float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)

    def update_lower_batched(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return self.lower_agent.update_lower(batch)

    def update_lower(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return self.lower_agent.update_lower(batch)

    def update_upper(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return self.upper_agent.update_upper(batch)

    def get_state_dicts(self) -> Dict[str, Any]:
        return {"lower": self.lower_agent.get_state_dicts(), "upper": self.upper_agent.get_state_dicts()}

    def load_state_dicts(self, state_dicts: Dict[str, Any]) -> None:
        self.lower_agent.load_state_dicts(state_dicts["lower"])
        self.upper_agent.load_state_dicts(state_dicts["upper"])
