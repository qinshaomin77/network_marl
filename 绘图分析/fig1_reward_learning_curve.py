#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图1：Reward 学习曲线
输入：episode_summary.csv
输出：analysis_figures/fig1_reward_learning_curve.png
"""

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main() -> None:
    window = 10
    BASE_DIR = Path(r"E:\DRL_project\network_marl_batch_v5\results")
    run_dir = BASE_DIR / "grid_5x5_20260428_112944"
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "结果分析"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "episode_summary.csv"
    out_path = out_dir / "fig1_reward_learning_curve.png"

    df = pd.read_csv(csv_path)

    required = {"episode", "avg_global_reward"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"episode_summary.csv 缺少字段: {sorted(missing)}")

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["avg_global_reward"] = pd.to_numeric(df["avg_global_reward"], errors="coerce")
    df = df.dropna(subset=["episode", "avg_global_reward"])
    df = df.sort_values("episode").copy()

    if df.empty:
        raise ValueError("episode_summary.csv 中没有有效 episode / avg_global_reward 数据。")

    window = max(int(window), 1)
    df["reward_ma"] = df["avg_global_reward"].rolling(
        window=window,
        min_periods=1,
    ).mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        df["episode"],
        df["avg_global_reward"],
        alpha=0.35,
        linewidth=1.0,
        label="avg_global_reward",
    )
    ax.plot(
        df["episode"],
        df["reward_ma"],
        linewidth=2.0,
        label=f"moving average ({window})",
    )

    best_idx = df["avg_global_reward"].idxmax()
    best_ep = df.loc[best_idx, "episode"]
    best_val = df.loc[best_idx, "avg_global_reward"]

    ax.scatter([best_ep], [best_val], s=70, zorder=5)
    ax.annotate(
        f"Best episode = {int(best_ep)}\nreward = {best_val:.4f}",
        xy=(best_ep, best_val),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", linewidth=1),
    )

    n = min(5, len(df))
    early = df["avg_global_reward"].iloc[:n].mean()
    late = df["avg_global_reward"].iloc[-n:].mean()

    if abs(early) > 1e-8:
        change_pct = (late - early) / abs(early) * 100.0
        change_text = f"{change_pct:+.2f}%"
    else:
        change_text = "N/A"

    ax.text(
        0.02,
        0.98,
        f"First {n} episodes avg: {early:.4f}\n"
        f"Last {n} episodes avg: {late:.4f}\n"
        f"Change: {change_text}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
    )

    ax.set_title("Reward Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Global Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] run_dir = {run_dir}")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()