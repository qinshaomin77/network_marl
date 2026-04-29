# -*- coding: utf-8 -*-
"""node_model.py
Lower layer A2C agent.
Revised design:
- local lane/connection feature: [wave, speed_lack, truck_ratio, queue_ratio]
- neighbor feature embedding is removed
- neighbor input is only policy probability matrix [4, A]
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from model_layers import orthogonal_init, _as_tensor, RelationGATLayer, LSTMCore


class LocalIntersectionGATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.node_input_dim = int(getattr(config, "NODE_FEATURE_DIM", 4))
        self.hidden_dim = int(getattr(config, "LOCAL_GAT_HIDDEN", 64))
        self.num_heads = int(getattr(config, "LOCAL_GAT_NUM_HEADS", 4))
        self.dropout_p = float(getattr(config, "LOCAL_GAT_DROPOUT", 0.1))
        self.readout = str(getattr(config, "LOCAL_GAT_READOUT", "paper_concat")).lower()
        self.negative_slope = float(getattr(config, "LOCAL_GAT_NEGATIVE_SLOPE", 0.05))
        self.max_nodes = int(getattr(config, "LOCAL_GAT_MAX_NODES", 12))

        self.fc_in = orthogonal_init(nn.Linear(self.node_input_dim, self.hidden_dim), gain=1.0)
        self.same_gat = RelationGATLayer(self.hidden_dim, self.hidden_dim, self.num_heads, self.dropout_p, self.negative_slope)
        self.diff_gat = RelationGATLayer(self.hidden_dim, self.hidden_dim, self.num_heads, self.dropout_p, self.negative_slope)
        self.node_update = nn.Sequential(
            orthogonal_init(nn.Linear(self.hidden_dim * 3, self.hidden_dim), gain=1.0),
            nn.LeakyReLU(self.negative_slope),
            orthogonal_init(nn.Linear(self.hidden_dim, self.hidden_dim), gain=1.0),
            nn.LeakyReLU(self.negative_slope),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        if self.readout == "paper_concat":
            self.output_dim = self.hidden_dim * self.max_nodes
        elif self.readout == "mean_max":
            self.output_dim = self.hidden_dim * 2
        else:
            self.output_dim = self.hidden_dim

    @staticmethod
    def _ensure_batch(x: torch.Tensor, ndim: int) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == ndim - 1 else x

    @staticmethod
    def _masked_mean(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(H.dtype)
        return (H * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _masked_max(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.full_like(H, -1e9)
        Hm = torch.where(mask.unsqueeze(-1).bool(), H, neg)
        out = Hm.max(dim=1).values
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    def _paper_concat_readout(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, D = H.shape
        if N > self.max_nodes:
            H_use, M_use = H[:, :self.max_nodes], mask[:, :self.max_nodes]
        elif N < self.max_nodes:
            pad_h = torch.zeros(B, self.max_nodes - N, D, device=H.device, dtype=H.dtype)
            pad_m = torch.zeros(B, self.max_nodes - N, device=mask.device, dtype=mask.dtype)
            H_use, M_use = torch.cat([H, pad_h], dim=1), torch.cat([mask, pad_m], dim=1)
        else:
            H_use, M_use = H, mask
        return (H_use * M_use.unsqueeze(-1).to(H_use.dtype)).reshape(B, self.max_nodes * D)

    def forward(self, obs_graph_dict: Dict[str, torch.Tensor], return_node_embeds: bool = False):
        x = self._ensure_batch(obs_graph_dict["node_features"], 3)
        A_same = self._ensure_batch(obs_graph_dict["A_same"], 3)
        A_diff = self._ensure_batch(obs_graph_dict["A_diff"], 3)
        lane_mask = self._ensure_batch(obs_graph_dict["lane_exist_mask"], 2)
        H0 = F.leaky_relu(self.fc_in(x), negative_slope=self.negative_slope)
        H_same = self.same_gat(H0, A_same, lane_mask)
        H_diff = self.diff_gat(H0, A_diff, lane_mask)
        H = self.node_update(torch.cat([H0, H_same, H_diff], dim=-1))
        H = self.dropout(self.norm(H)) * lane_mask.unsqueeze(-1).to(H.dtype)
        if self.readout == "mean":
            graph = self._masked_mean(H, lane_mask)
        elif self.readout == "max":
            graph = self._masked_max(H, lane_mask)
        elif self.readout == "paper_concat":
            graph = self._paper_concat_readout(H, lane_mask)
        else:
            graph = torch.cat([self._masked_mean(H, lane_mask), self._masked_max(H, lane_mask)], dim=-1)
        return (graph, H) if return_node_embeds else graph


class NeighborPolicyEncoder(nn.Module):
    """Encode neighbor policy matrix [B,4,A] using mask + flatten + MLP."""

    def __init__(self, config):
        super().__init__()
        a_max = int(getattr(config, "A_MAX", getattr(config, "NUM_ACTIONS", 4)) or 4)
        hidden = int(getattr(config, "NEIGHBOR_POLICY_HIDDEN", 32))
        self.a_max = a_max
        self.output_dim = hidden
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(4 * a_max, hidden), gain=1.0),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden, hidden), gain=1.0),
            nn.ReLU(),
        )

    def forward(self, neighbor_policy_dir: torch.Tensor, neighbor_dir_mask: torch.Tensor, neighbor_policy_action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if neighbor_policy_dir.dim() == 2:
            neighbor_policy_dir = neighbor_policy_dir.unsqueeze(0)
        if neighbor_dir_mask.dim() == 1:
            neighbor_dir_mask = neighbor_dir_mask.unsqueeze(0)
        P = neighbor_policy_dir
        B, D, A = P.shape
        if A < self.a_max:
            P = F.pad(P, (0, self.a_max - A))
        elif A > self.a_max:
            P = P[:, :, :self.a_max]
        M_dir = neighbor_dir_mask[:, :4].unsqueeze(-1).to(P.dtype)
        P = P[:, :4, :] * M_dir
        if neighbor_policy_action_mask is not None:
            if neighbor_policy_action_mask.dim() == 2:
                neighbor_policy_action_mask = neighbor_policy_action_mask.unsqueeze(0)
            M_act = neighbor_policy_action_mask[:, :4, :self.a_max].to(P.dtype)
            if M_act.shape[-1] < self.a_max:
                M_act = F.pad(M_act, (0, self.a_max - M_act.shape[-1]))
            P = P * M_act
        return self.net(P.reshape(B, 4 * self.a_max))


class LowerActorNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_encoder = LocalIntersectionGATEncoder(config)
        self.neighbor_policy_encoder = NeighborPolicyEncoder(config)
        self.upper_weight_dim = int(getattr(config, "UPPER_WEIGHT_DIM", 2))
        fusion_dim = self.local_encoder.output_dim + self.neighbor_policy_encoder.output_dim + self.upper_weight_dim
        hidden = int(getattr(config, "LOWER_MLP_HIDDEN", 128))
        lstm_dim = int(getattr(config, "LSTM_DIM", 64))
        self.fusion = nn.Sequential(
            orthogonal_init(nn.Linear(fusion_dim, hidden), gain=1.0),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden, lstm_dim), gain=1.0),
            nn.ReLU(),
        )
        self.lstm = LSTMCore(lstm_dim, lstm_dim)
        self.output_layer = orthogonal_init(nn.Linear(lstm_dim, int(getattr(config, "A_MAX", 4) or 4)), gain=0.01)

    def _encode(self, obs: Dict[str, torch.Tensor], upper_weight: torch.Tensor) -> torch.Tensor:
        g_local = self.local_encoder(obs)
        p_nbr = self.neighbor_policy_encoder(obs["neighbor_policy_dir"], obs["neighbor_dir_mask"], obs.get("neighbor_policy_action_mask"))
        if upper_weight.dim() == 1:
            upper_weight = upper_weight.unsqueeze(0)
        return self.fusion(torch.cat([g_local, p_nbr, upper_weight], dim=-1))

    def forward_batched(self, obs: Dict[str, torch.Tensor], upper_weight: torch.Tensor, hidden=None):
        z = self._encode(obs, upper_weight)
        h, new_hidden = self.lstm(z, hidden)
        logits = self.output_layer(h)
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(action_mask <= 0.0, -1e9)
        return logits, new_hidden


class LowerCriticNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_encoder = LocalIntersectionGATEncoder(config)
        self.neighbor_policy_encoder = NeighborPolicyEncoder(config)
        self.upper_weight_dim = int(getattr(config, "UPPER_WEIGHT_DIM", 2))
        fusion_dim = self.local_encoder.output_dim + self.neighbor_policy_encoder.output_dim + self.upper_weight_dim
        hidden = int(getattr(config, "LOWER_MLP_HIDDEN", 128))
        lstm_dim = int(getattr(config, "LSTM_DIM", 64))
        self.fusion = nn.Sequential(
            orthogonal_init(nn.Linear(fusion_dim, hidden), gain=1.0),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden, lstm_dim), gain=1.0),
            nn.ReLU(),
        )
        self.lstm = LSTMCore(lstm_dim, lstm_dim)
        self.value_layer = orthogonal_init(nn.Linear(lstm_dim, 1), gain=1.0)

    def _encode(self, obs: Dict[str, torch.Tensor], upper_weight: torch.Tensor) -> torch.Tensor:
        g_local = self.local_encoder(obs)
        p_nbr = self.neighbor_policy_encoder(obs["neighbor_policy_dir"], obs["neighbor_dir_mask"], obs.get("neighbor_policy_action_mask"))
        if upper_weight.dim() == 1:
            upper_weight = upper_weight.unsqueeze(0)
        return self.fusion(torch.cat([g_local, p_nbr, upper_weight], dim=-1))

    def forward_batched(self, obs: Dict[str, torch.Tensor], upper_weight: torch.Tensor, hidden=None):
        z = self._encode(obs, upper_weight)
        h, new_hidden = self.lstm(z, hidden)
        return self.value_layer(h).squeeze(-1), new_hidden


def _stack_obs_list(obs_list: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = ["node_features", "A_same", "A_diff", "lane_exist_mask", "neighbor_dir_mask", "neighbor_policy_dir", "neighbor_policy_action_mask", "action_mask"]
    return {k: torch.stack([_as_tensor(o[k], device) for o in obs_list], dim=0) for k in keys}


def _obs_dict_to_tensors(obs: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: _as_tensor(v, device) for k, v in obs.items() if k not in ("raw", "norm")}


class LowerA2CAgent:
    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = LowerActorNet(config).to(self.device)
        self.critic = LowerCriticNet(config).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, "LOWER_LR_ACTOR", 1.5e-4)))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(getattr(config, "LOWER_LR_CRITIC", 3e-4)))
        self.value_coef = float(getattr(config, "LOWER_VALUE_COEF", 0.1))
        self.entropy_coef = float(getattr(config, "LOWER_ENTROPY_COEF", 0.02))
        self.max_grad_norm = float(getattr(config, "LOWER_MAX_GRAD_NORM", 5.0))
        self.adv_norm = bool(getattr(config, "LOWER_ADV_NORM", True))

    def init_hidden(self, batch_size: int = 1):
        return self.actor.lstm.init_hidden(self.device, batch_size), self.critic.lstm.init_hidden(self.device, batch_size)

    def act_all_batched(self, per_tls_obs: Dict[str, Dict[str, Any]], tls_ids: List[str], actor_hidden, critic_hidden, upper_weight: torch.Tensor, deterministic: bool = False):
        obs_list = [per_tls_obs[tid] for tid in tls_ids]
        obs_t = _stack_obs_list(obs_list, self.device)
        upper_weight = upper_weight.to(self.device, dtype=torch.float32)
        logits, new_actor_hidden = self.actor.forward_batched(obs_t, upper_weight, actor_hidden)
        dist = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values, new_critic_hidden = self.critic.forward_batched(obs_t, upper_weight, critic_hidden)
        probs = dist.probs.detach().cpu().numpy().astype(np.float32)
        return {
            "action_dict": {tid: int(actions[i].item()) for i, tid in enumerate(tls_ids)},
            "log_prob_dict": {tid: logp[i].detach() for i, tid in enumerate(tls_ids)},
            "entropy_dict": {tid: entropy[i].detach() for i, tid in enumerate(tls_ids)},
            "value_dict": {tid: float(values[i].detach().cpu().item()) for i, tid in enumerate(tls_ids)},
            "policy_dict": {tid: probs[i] for i, tid in enumerate(tls_ids)},
            "actor_hidden": tuple(x.detach() for x in new_actor_hidden),
            "critic_hidden": tuple(x.detach() for x in new_critic_hidden),
        }

    def get_bootstrap_values_batched(self, per_tls_obs, tls_ids, critic_hidden, upper_weight):
        obs_t = _stack_obs_list([per_tls_obs[tid] for tid in tls_ids], self.device)
        with torch.no_grad():
            values, _ = self.critic.forward_batched(obs_t, upper_weight.to(self.device, dtype=torch.float32), critic_hidden)
        return {tid: float(values[i].cpu().item()) for i, tid in enumerate(tls_ids)}

    def compute_lower_loss_batched(self, batch: Dict[str, Any]):
        obs_t = _stack_obs_list(batch["obs_list"], self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        returns_raw = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        if bool(getattr(self.config, "LOWER_RETURN_NORM", True)) and returns_raw.numel() > 1:
            returns = (returns_raw - returns_raw.mean()) / (returns_raw.std(unbiased=False) + 1e-8)
        else:
            returns = returns_raw
        uw_list = batch.get("upper_weight_list", None)
        if uw_list is None or any(x is None for x in uw_list):
            upper_weight = torch.full((len(actions), 2), 0.5, dtype=torch.float32, device=self.device)
        else:
            upper_weight = torch.stack([_as_tensor(x, self.device) for x in uw_list], dim=0)
        logits, _ = self.actor.forward_batched(obs_t, upper_weight, hidden=None)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values, _ = self.critic.forward_batched(obs_t, upper_weight, hidden=None)
        adv = returns.detach() - values.detach()
        if self.adv_norm and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        policy_loss = -(logp * adv.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total_loss, policy_loss, value_loss, entropy

    @staticmethod
    def _collect_grad_norms(module: nn.Module) -> Dict[str, float]:
        out = {}
        for name, p in module.named_parameters():
            if p.grad is not None:
                out[name] = float(p.grad.detach().data.norm(2).cpu().item())
        return out

    def update_lower(self, batch: Dict[str, Any]) -> Dict[str, float]:
        total_loss, policy_loss, value_loss, entropy = self.compute_lower_loss_batched(batch)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        actor_grads = self._collect_grad_norms(self.actor)
        critic_grads = self._collect_grad_norms(self.critic)
        self.actor_opt.step()
        self.critic_opt.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy_loss": float((-self.entropy_coef * entropy).item()),
            "total_loss": float(total_loss.item()),
            "actor_loss": float(policy_loss.item()),
            "critic_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "actor_grad_norm": float(sum(actor_grads.values())) if actor_grads else 0.0,
            "critic_grad_norm": float(sum(critic_grads.values())) if critic_grads else 0.0,
            "actor_layer_grads": actor_grads,
            "critic_layer_grads": critic_grads,
        }

    def get_state_dicts(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict()}

    def load_state_dicts(self, payload):
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_opt" in payload:
            self.actor_opt.load_state_dict(payload["actor_opt"])
        if "critic_opt" in payload:
            self.critic_opt.load_state_dict(payload["critic_opt"])
