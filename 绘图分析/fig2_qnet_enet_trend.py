#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图2：Q_net / E_net 趋势图
输入：episode_summary.csv
输出：analysis_figures/fig2_qnet_enet_trend.png
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def pct_change_text(early: float, late: float) -> str:
    """
    计算前后变化百分比。
    注意：
    - Q_net / E_net 一般是成本指标，数值下降代表改善；
    - 这里只负责显示数学变化率，不判断好坏。
    """
    if abs(early) <= 1e-8:
        return "N/A"
    return f"{(late - early) / abs(early) * 100.0:+.2f}%"

def main() -> None:
    window = 10
    BASE_DIR = Path(r"E:\DRL_project\network_marl_batch_v5\results")
    run_dir = BASE_DIR / "grid_5x5_20260428_112944"

    logs_dir = run_dir / "logs"
    out_dir = run_dir / "结果分析"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "episode_summary.csv"
    out_path = out_dir / "fig2_qnet_enet_trend.png"

    df = pd.read_csv(csv_path)

    required = {"episode", "avg_Q_net", "avg_E_net"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"episode_summary.csv 缺少字段: {sorted(missing)}")

    for col in ["episode", "avg_Q_net", "avg_E_net"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["episode", "avg_Q_net", "avg_E_net"])
    df = df.sort_values("episode").copy()

    if df.empty:
        raise ValueError("episode_summary.csv 中没有有效 Q_net / E_net 数据。")

    window = max(int(window), 1)
    df["Q_ma"] = df["avg_Q_net"].rolling(window=window, min_periods=1).mean()
    df["E_ma"] = df["avg_E_net"].rolling(window=window, min_periods=1).mean()

    n = min(5, len(df))

    q_early = df["avg_Q_net"].iloc[:n].mean()
    q_late = df["avg_Q_net"].iloc[-n:].mean()

    e_early = df["avg_E_net"].iloc[:n].mean()
    e_late = df["avg_E_net"].iloc[-n:].mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        df["episode"],
        df["avg_Q_net"],
        alpha=0.35,
        linewidth=1.0,
        label="avg_Q_net",
    )
    axes[0].plot(
        df["episode"],
        df["Q_ma"],
        linewidth=2.0,
        label=f"moving average ({window})",
    )
    axes[0].set_title("Q_net Trend")
    axes[0].set_ylabel("Average Q_net")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[0].text(
        0.98,
        0.95,
        f"First {n}: {q_early:.4f}\n"
        f"Last {n}: {q_late:.4f}\n"
        f"Change: {pct_change_text(q_early, q_late)}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.15),
    )

    axes[1].plot(
        df["episode"],
        df["avg_E_net"],
        alpha=0.35,
        linewidth=1.0,
        label="avg_E_net",
    )
    axes[1].plot(
        df["episode"],
        df["E_ma"],
        linewidth=2.0,
        label=f"moving average ({window})",
    )
    axes[1].set_title("E_net Trend")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average E_net")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[1].text(
        0.98,
        0.95,
        f"First {n}: {e_early:.4f}\n"
        f"Last {n}: {e_late:.4f}\n"
        f"Change: {pct_change_text(e_early, e_late)}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
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