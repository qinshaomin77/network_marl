#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    window = 10
    BASE_DIR = Path(r"E:\DRL_project\network_marl_batch_v5\results")
    run_dir = BASE_DIR / "grid_5x5_20260428_112944"

    logs_dir = run_dir / "logs"
    out_dir = run_dir / "结果分析"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "upper_step_metrics.csv"
    out_path = out_dir / "fig3_upper_weight_evolution.png"

    df = pd.read_csv(csv_path)

    required = {
        "episode",
        "w_em_mean",
        "w_em_std",
        "w_eff_mean",
        "w_eff_std",
        "upper_reward",
        "upper_entropy",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"upper_step_metrics.csv 缺少字段: {sorted(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["episode", "w_em_mean", "w_eff_mean"])
    df = df.sort_values("episode").copy()

    if df.empty:
        raise ValueError("upper_step_metrics.csv 中没有有效上层权重数据。")

    ep = (
        df.groupby("episode", as_index=False)
        .agg(
            w_em_mean=("w_em_mean", "mean"),
            w_em_std=("w_em_std", "mean"),
            w_eff_mean=("w_eff_mean", "mean"),
            w_eff_std=("w_eff_std", "mean"),
            upper_reward=("upper_reward", "mean"),
            upper_entropy=("upper_entropy", "mean"),
        )
        .sort_values("episode")
    )

    for col in ["w_em_std", "w_eff_std"]:
        ep[col] = ep[col].fillna(0.0)

    window = max(int(window), 1)
    ep["upper_reward_ma"] = ep["upper_reward"].rolling(
        window=window,
        min_periods=1,
    ).mean()
    ep["upper_entropy_ma"] = ep["upper_entropy"].rolling(
        window=window,
        min_periods=1,
    ).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(
        ep["episode"],
        ep["w_em_mean"],
        linewidth=2,
        label="w_em_mean",
    )
    axes[0].fill_between(
        ep["episode"],
        ep["w_em_mean"] - ep["w_em_std"],
        ep["w_em_mean"] + ep["w_em_std"],
        alpha=0.18,
        label="w_em ± std",
    )

    axes[0].plot(
        ep["episode"],
        ep["w_eff_mean"],
        linewidth=2,
        label="w_eff_mean",
    )
    axes[0].fill_between(
        ep["episode"],
        ep["w_eff_mean"] - ep["w_eff_std"],
        ep["w_eff_mean"] + ep["w_eff_std"],
        alpha=0.18,
        label="w_eff ± std",
    )

    axes[0].axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)
    axes[0].set_title("Upper Weight Evolution")
    axes[0].set_ylabel("Weight")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    axes[1].plot(
        ep["episode"],
        ep["upper_reward"],
        alpha=0.35,
        linewidth=1,
        label="upper_reward",
    )
    axes[1].plot(
        ep["episode"],
        ep["upper_reward_ma"],
        linewidth=2,
        label=f"moving average ({window})",
    )
    axes[1].set_title("Upper Reward Trend")
    axes[1].set_ylabel("Upper Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        ep["episode"],
        ep["upper_entropy"],
        alpha=0.35,
        linewidth=1,
        label="upper_entropy",
    )
    axes[2].plot(
        ep["episode"],
        ep["upper_entropy_ma"],
        linewidth=2,
        label=f"moving average ({window})",
    )
    axes[2].set_title("Upper Entropy Trend")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Upper Entropy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    last = ep.iloc[-1]
    axes[0].text(
        0.98,
        0.05,
        f"Final w_em = {last['w_em_mean']:.3f}\n"
        f"Final w_eff = {last['w_eff_mean']:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] run_dir = {run_dir}")
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
