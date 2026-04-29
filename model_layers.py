# -*- coding: utf-8 -*-
"""model_layers.py
Shared neural-network layers used by lower and upper models.
The same class can be reused, but lower and upper instantiate separate modules,
so their parameters are not shared.
"""

from __future__ import annotations
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


def _as_tensor(x: Any, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, activation=nn.ReLU):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(max(1, int(n_layers) - 1)):
            layers += [orthogonal_init(nn.Linear(d, int(hidden_dim)), gain=1.0), activation()]
            d = int(hidden_dim)
        layers.append(orthogonal_init(nn.Linear(d, int(out_dim)), gain=1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationGATLayer(nn.Module):
    """Additive multi-head GAT with relation-specific adjacency mask.

    Input:
        H: [B,N,D]
        adj_mask: [B,N,N] or [N,N]
        node_mask: [B,N] optional
    Output:
        [B,N,out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout_p: float = 0.1, negative_slope: float = 0.05):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.dropout_p = float(dropout_p)
        self.negative_slope = float(negative_slope)
        self.w_proj = orthogonal_init(nn.Linear(self.in_dim, self.num_heads * self.out_dim, bias=False), gain=1.0)
        self.attn_vec = nn.Parameter(torch.empty(1, self.num_heads, 1, 1, self.out_dim * 2))
        nn.init.xavier_uniform_(self.attn_vec)
        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.out_dropout = nn.Dropout(self.dropout_p)

    @staticmethod
    def _ensure_batch_adj(adj: torch.Tensor, B: int) -> torch.Tensor:
        return adj.unsqueeze(0).expand(B, -1, -1) if adj.dim() == 2 else adj

    @staticmethod
    def _ensure_self_loop(adj: torch.Tensor, valid_nodes: torch.Tensor) -> torch.Tensor:
        B, N, _ = adj.shape
        eye = torch.eye(N, device=adj.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        self_loop = eye & valid_nodes.unsqueeze(-1) & valid_nodes.unsqueeze(-2)
        return adj | self_loop

    def forward(self, H: torch.Tensor, adj_mask: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if H.dim() == 2:
            H = H.unsqueeze(0)
        B, N, _ = H.shape
        if node_mask is None:
            valid_nodes = torch.ones(B, N, device=H.device, dtype=torch.bool)
        else:
            valid_nodes = node_mask.bool()
            if valid_nodes.dim() == 1:
                valid_nodes = valid_nodes.unsqueeze(0).expand(B, -1)
        adj = self._ensure_batch_adj(adj_mask.to(H.device), B).bool()
        pair_valid = valid_nodes.unsqueeze(-1) & valid_nodes.unsqueeze(-2)
        adj = self._ensure_self_loop(adj & pair_valid, valid_nodes)

        Wh = self.w_proj(H).view(B, N, self.num_heads, self.out_dim).permute(0, 2, 1, 3)
        Wh_i = Wh.unsqueeze(3).expand(B, self.num_heads, N, N, self.out_dim)
        Wh_j = Wh.unsqueeze(2).expand(B, self.num_heads, N, N, self.out_dim)
        pair = torch.cat([Wh_i, Wh_j], dim=-1)
        e = F.leaky_relu((pair * self.attn_vec).sum(dim=-1), negative_slope=self.negative_slope)
        e = e.masked_fill(~adj.unsqueeze(1), -1e9)
        alpha = self.attn_dropout(F.softmax(e, dim=-1))
        out = torch.matmul(alpha, Wh).mean(dim=1)
        out = self.out_dropout(out)
        return out * valid_nodes.unsqueeze(-1).to(out.dtype)


class LSTMCore(nn.Module):
    """One-step LSTMCell wrapper."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cell = nn.LSTMCell(int(input_dim), self.hidden_dim)
        for name, p in self.cell.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)

    def init_hidden(self, device: torch.device, batch_size: int = 1):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(self, x: torch.Tensor, hidden):
        if hidden is None:
            hidden = self.init_hidden(x.device, x.size(0))
        h, c = self.cell(x, hidden)
        return h, (h, c)
