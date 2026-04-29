#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def smooth(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()

def prepare_training_df(df: pd.DataFrame, required: set[str], name: str) -> pd.DataFrame:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少字段: {sorted(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required)).copy()
    return df

def main() -> None:
    window = 10
    BASE_DIR = Path(r"E:\DRL_project\network_marl_batch_v5\results")
    run_dir = BASE_DIR / "grid_5x5_20260428_112944"

    logs_dir = run_dir / "logs"
    out_dir = run_dir / "结果分析"
    out_dir.mkdir(parents=True, exist_ok=True)

    lower_path = logs_dir / "training_metrics_lower.csv"
    upper_path = logs_dir / "training_metrics_upper.csv"
    out_path = out_dir / "fig5_loss_grad_stability.png"

    lower = pd.read_csv(lower_path)
    upper = pd.read_csv(upper_path)

    req_lower = {
        "lower_update_idx",
        "policy_loss",
        "value_loss",
        "actor_grad_norm",
        "critic_grad_norm",
    }
    req_upper = {
        "upper_update_idx",
        "policy_loss",
        "value_loss",
        "actor_grad_norm",
        "critic_grad_norm",
    }

    lower = prepare_training_df(lower, req_lower, "training_metrics_lower.csv")
    upper = prepare_training_df(upper, req_upper, "training_metrics_upper.csv")

    lower = lower.sort_values("lower_update_idx").copy()
    upper = upper.sort_values("upper_update_idx").copy()

    if lower.empty:
        raise ValueError("training_metrics_lower.csv 中没有有效训练数据。")

    if upper.empty:
        raise ValueError("training_metrics_upper.csv 中没有有效训练数据。")

    window = max(int(window), 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(
        lower["lower_update_idx"],
        lower["policy_loss"],
        alpha=0.25,
        linewidth=0.8,
        label="policy_loss raw",
    )
    axes[0, 0].plot(
        lower["lower_update_idx"],
        smooth(lower["policy_loss"], window),
        linewidth=1.8,
        label="policy_loss smoothed",
    )
    axes[0, 0].plot(
        lower["lower_update_idx"],
        smooth(lower["value_loss"], window),
        linewidth=1.8,
        label="value_loss smoothed",
    )
    axes[0, 0].set_title("Lower Layer Loss")
    axes[0, 0].set_xlabel("lower_update_idx")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        upper["upper_update_idx"],
        upper["policy_loss"],
        alpha=0.25,
        linewidth=0.8,
        label="policy_loss raw",
    )
    axes[0, 1].plot(
        upper["upper_update_idx"],
        smooth(upper["policy_loss"], window),
        linewidth=1.8,
        label="policy_loss smoothed",
    )
    axes[0, 1].plot(
        upper["upper_update_idx"],
        smooth(upper["value_loss"], window),
        linewidth=1.8,
        label="value_loss smoothed",
    )
    axes[0, 1].set_title("Upper Layer Loss")
    axes[0, 1].set_xlabel("upper_update_idx")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        lower["lower_update_idx"],
        smooth(lower["actor_grad_norm"], window),
        linewidth=1.8,
        label="actor_grad_norm",
    )
    axes[1, 0].plot(
        lower["lower_update_idx"],
        smooth(lower["critic_grad_norm"], window),
        linewidth=1.8,
        label="critic_grad_norm",
    )
    axes[1, 0].set_title("Lower Layer Gradient Norm")
    axes[1, 0].set_xlabel("lower_update_idx")
    axes[1, 0].set_ylabel("Gradient Norm")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(
        upper["upper_update_idx"],
        smooth(upper["actor_grad_norm"], window),
        linewidth=1.8,
        label="actor_grad_norm",
    )
    axes[1, 1].plot(
        upper["upper_update_idx"],
        smooth(upper["critic_grad_norm"], window),
        linewidth=1.8,
        label="critic_grad_norm",
    )
    axes[1, 1].set_title("Upper Layer Gradient Norm")
    axes[1, 1].set_xlabel("upper_update_idx")
    axes[1, 1].set_ylabel("Gradient Norm")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] run_dir = {run_dir}")
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
