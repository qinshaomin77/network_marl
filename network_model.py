# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_layers import orthogonal_init, _as_tensor, RelationGATLayer

# Upper encoder over directed edge graph (upstream/downstream relations).
class UpperEdgeGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = int(getattr(config, "UPPER_EDGE_FEATURE_DIM", getattr(config, "X_UP_DIM", 3)))
        hidden = int(getattr(config, "UPPER_EDGE_HIDDEN_DIM", 64))
        heads = int(getattr(config, "UPPER_EDGE_GAT_HEADS", 4))
        dropout = float(getattr(config, "UPPER_EDGE_GAT_DROPOUT", 0.1))
        neg = float(getattr(config, "UPPER_EDGE_GAT_NEGATIVE_SLOPE", 0.05))
        self.hidden = hidden
        self.edge_mlp = nn.Sequential(
            orthogonal_init(nn.Linear(in_dim, hidden), gain=1.0),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden, hidden), gain=1.0),
            nn.ReLU(),
        )
        self.up_gat = RelationGATLayer(hidden, hidden, heads, dropout, neg)
        self.down_gat = RelationGATLayer(hidden, hidden, heads, dropout, neg)
        self.fusion = nn.Sequential(
            orthogonal_init(nn.Linear(hidden * 3, hidden), gain=1.0),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.output_dim = hidden

    def forward(self, X_edge: torch.Tensor, A_up: torch.Tensor, A_down: torch.Tensor) -> torch.Tensor:
        if X_edge.dim() == 2:
            X_edge = X_edge.unsqueeze(0)
        H0 = self.edge_mlp(X_edge)
        H_up = self.up_gat(H0, A_up)
        H_down = self.down_gat(H0, A_down)
        return self.fusion(torch.cat([H0, H_up, H_down], dim=-1))

class UpperCoordinatorNet(nn.Module):
    # PPO actor: samples edge preferences, then aggregates to tls-level weights.

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = UpperEdgeGraphEncoder(config)
        h = self.encoder.output_dim
        self.edge_mu_head = nn.Sequential(
            orthogonal_init(nn.Linear(h, h), gain=1.0),
            nn.ReLU(),
            orthogonal_init(nn.Linear(h, 1), gain=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(1))
        self.w_min = float(getattr(config, "UPPER_W_MIN", 0.15))
        self.w_max = float(getattr(config, "UPPER_W_MAX", 0.85))

    def _dist(self, X_edge: torch.Tensor, A_up: torch.Tensor, A_down: torch.Tensor):
        Z = self.encoder(X_edge, A_up, A_down)  # [B,E,H]
        mu = self.edge_mu_head(Z).squeeze(-1)   # [B,E]
        std = self.log_std.exp().clamp(min=1e-4, max=10.0)
        return torch.distributions.Normal(mu, std), Z

    def _build_weights(self, raw_delta: torch.Tensor, edge_to_tls: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_to_tls.dim() == 2:
            edge_to_tls = edge_to_tls.unsqueeze(0).expand(raw_delta.size(0), -1, -1)
        p = torch.sigmoid(raw_delta).clamp(self.w_min, self.w_max)  # emission preference [B,E]
        W_edge = torch.stack([p, 1.0 - p], dim=-1)                  # [B,E,2]
        M = edge_to_tls.to(W_edge.dtype)
        if edge_weight is None:
            weights = M
        else:
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(0)
            weights = M * edge_weight.unsqueeze(1).to(M.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)    # [B,N,1]
        W_tls = torch.matmul(weights, W_edge) / denom               # [B,N,2]
        W_tls = W_tls.clamp(min=1e-6)
        W_tls = W_tls / W_tls.sum(dim=-1, keepdim=True)
        return W_tls, W_edge, p

    def sample_action(self, X_edge: torch.Tensor, A_up: torch.Tensor, A_down: torch.Tensor, edge_to_tls: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, deterministic: bool = False):
        dist, Z = self._dist(X_edge, A_up, A_down)
        raw_delta = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_delta).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        W_tls, W_edge, p = self._build_weights(raw_delta, edge_to_tls, edge_weight)
        raw_lam = raw_delta.mean(dim=-1)  # compatibility with old PPOBuffer/logger
        return {
            "raw_lam": raw_lam.squeeze(0),
            "raw_delta": raw_delta.squeeze(0),
            "lam": torch.sigmoid(raw_lam).squeeze(0),
            "delta": raw_delta.squeeze(0),
            "edge_pref": p.squeeze(0),
            "edge_weight_matrix": W_edge.squeeze(0),
            "weight_matrix": W_tls.squeeze(0),
            "log_prob": log_prob.squeeze(0),
            "entropy": entropy.squeeze(0),
        }

    def evaluate_action(self, X_edge: torch.Tensor, A_up: torch.Tensor, A_down: torch.Tensor, edge_to_tls: torch.Tensor, raw_delta: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        dist, _ = self._dist(X_edge, A_up, A_down)
        if raw_delta.dim() == 1:
            raw_delta = raw_delta.unsqueeze(0)
        log_prob = dist.log_prob(raw_delta).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        W_tls, W_edge, p = self._build_weights(raw_delta, edge_to_tls, edge_weight)
        return log_prob, entropy, W_tls, W_edge, p

class UpperValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = UpperEdgeGraphEncoder(config)
        h = self.encoder.output_dim
        val_h = int(getattr(config, "UPPER_EDGE_VALUE_DIM", 128))
        self.value_head = nn.Sequential(
            orthogonal_init(nn.Linear(h * 2, val_h), gain=1.0),
            nn.ReLU(),
            orthogonal_init(nn.Linear(val_h, 1), gain=1.0),
        )

    def forward(self, X_edge: torch.Tensor, A_up: torch.Tensor, A_down: torch.Tensor) -> torch.Tensor:
        Z = self.encoder(X_edge, A_up, A_down)  # [B,E,H]
        h_mean = Z.mean(dim=1)
        h_max = Z.max(dim=1).values
        return self.value_head(torch.cat([h_mean, h_max], dim=-1)).squeeze(-1)

class UpperPPOAgent:
    # Upper-level PPO update logic.
    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = UpperCoordinatorNet(config).to(self.device)
        self.critic = UpperValueNet(config).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, "UPPER_LR_ACTOR", 3e-4)))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(getattr(config, "UPPER_LR_CRITIC", 5e-4)))
        self.clip_ratio = float(getattr(config, "UPPER_CLIP_RATIO", 0.2))
        self.update_epochs = int(getattr(config, "UPPER_UPDATE_EPOCHS", 4))
        self.batch_size = int(getattr(config, "UPPER_BATCH_SIZE", 10))
        self.value_coef = float(getattr(config, "UPPER_VALUE_COEF", 0.5))
        self.entropy_coef = float(getattr(config, "UPPER_ENTROPY_COEF", 0.02))
        self.max_grad_norm = float(getattr(config, "UPPER_MAX_GRAD_NORM", 5.0))

    def _to_tensors(self, network_obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            "X_edge": _as_tensor(network_obs["X_edge"], self.device),
            "A_edge_up": _as_tensor(network_obs["A_edge_up"], self.device),
            "A_edge_down": _as_tensor(network_obs["A_edge_down"], self.device),
            "edge_to_tls": _as_tensor(network_obs["edge_to_tls"], self.device),
            "edge_weight": _as_tensor(network_obs.get("edge_weight", np.ones(len(network_obs["X_edge"]), dtype=np.float32)), self.device),
        }

    def act_upper(self, network_obs: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        t = self._to_tensors(network_obs)
        with torch.no_grad():
            out = self.actor.sample_action(t["X_edge"], t["A_edge_up"], t["A_edge_down"], t["edge_to_tls"], t["edge_weight"], deterministic)
            value = self.critic(t["X_edge"], t["A_edge_up"], t["A_edge_down"]).squeeze(0)
        out["value"] = value
        out["weight_matrix_np"] = out["weight_matrix"].detach().cpu().numpy().astype(np.float32)
        return out

    @staticmethod
    def _collect_grad_norms(module: nn.Module) -> Dict[str, float]:
        return {name: float(p.grad.detach().data.norm(2).cpu().item()) for name, p in module.named_parameters() if p.grad is not None}
    
    def update_upper(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs_list = batch["network_obs_list"]
        raw_delta_list = batch["raw_delta_list"]
        old_log_probs = torch.as_tensor(
            batch["old_log_probs"],
            dtype=torch.float32,
            device=self.device,
        )
        returns_raw = torch.as_tensor(
            batch["returns"],
            dtype=torch.float32,
            device=self.device,
        )
        if bool(getattr(self.config, "UPPER_RETURN_NORM", True)) and returns_raw.numel() > 1:
            returns = (returns_raw - returns_raw.mean()) / (returns_raw.std(unbiased=False) + 1e-8)
        else:
            returns = returns_raw
        n = len(obs_list)
        idxs = np.arange(n)
        last = {}
        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                mb = idxs[start:start + self.batch_size]
                pol_losses = []
                entropies = []
                values = []
                for j in mb:
                    j = int(j)
                    t = self._to_tensors(obs_list[j])
                    raw_delta = _as_tensor(raw_delta_list[j], self.device)
                    logp, ent, _, _, _ = self.actor.evaluate_action(
                        t["X_edge"],
                        t["A_edge_up"],
                        t["A_edge_down"],
                        t["edge_to_tls"],
                        raw_delta,
                        t["edge_weight"],
                    )
                    value_j = self.critic(
                        t["X_edge"],
                        t["A_edge_up"],
                        t["A_edge_down"],
                    ).squeeze(0)
                    adv_j = returns[j].detach() - value_j.detach()
                    ratio = torch.exp(logp.squeeze(0) - old_log_probs[j])
                    surr1 = ratio * adv_j
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    ) * adv_j
                    pol_losses.append(-torch.min(surr1, surr2))
                    entropies.append(ent.squeeze(0))
                    values.append(value_j)
                policy_loss = torch.stack(pol_losses).mean()
                entropy = torch.stack(entropies).mean()
                values_t = torch.stack(values)
                value_loss = F.mse_loss(values_t, returns[mb])
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                actor_grads = self._collect_grad_norms(self.actor)
                critic_grads = self._collect_grad_norms(self.critic)
                self.actor_opt.step()
                self.critic_opt.step()
                last = {
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
        return last

    def get_state_dicts(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict()}

    def load_state_dicts(self, payload):
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_opt" in payload:
            self.actor_opt.load_state_dict(payload["actor_opt"])
        if "critic_opt" in payload:
            self.critic_opt.load_state_dict(payload["critic_opt"])
