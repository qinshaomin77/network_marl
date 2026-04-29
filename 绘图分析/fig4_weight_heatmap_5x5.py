#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图4：5×5 路口权重热力图

输入：
    tls_step_metrics.csv

输出：
    analysis_figures/fig4_weight_heatmap_5x5_epXXX.png

功能：
1. 可以通过 --episodes 指定多个 episode：
   python fig4_weight_heatmap_5x5.py --episodes 5,10,15,20

2. 如果不指定 --episodes，则默认绘制 episode 为 5 的倍数：
   ep=5,10,15,20,...

3. 每个 episode 单独输出一张 5×5 双热力图：
   左图：w_em
   右图：w_eff
"""

from __future__ import annotations

import argparse
import re, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 路径解析
# =============================================================================

def resolve_run_dir(run_dir: str | None) -> Path:
    """
    如果用户指定 --run-dir，则使用指定目录；
    否则自动寻找 results/grid_5x5_* 中最新的结果目录。
    """
    if run_dir:
        p = Path(run_dir)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"

    candidates = [
        p for p in results_dir.glob("grid_5x5_*")
        if (p / "logs" / "tls_step_metrics.csv").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            "未找到 results/grid_5x5_*/logs/tls_step_metrics.csv，"
            "请使用 --run-dir 手动指定结果目录。"
        )

    return sorted(candidates, key=lambda x: x.name)[-1].resolve()


# =============================================================================
# episode 选择
# =============================================================================

def parse_episodes_arg(episodes_str: str | None) -> list[int] | None:
    """
    解析 --episodes 参数。

    示例：
        --episodes 5,10,15
        --episodes 1,2,3,4,5

    返回：
        [5, 10, 15]
    """
    if episodes_str is None or str(episodes_str).strip() == "":
        return None

    out = []
    for item in str(episodes_str).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(float(item)))

    return sorted(set(out))


def select_target_episodes(
    all_episodes: list[int],
    episodes_arg: list[int] | None,
    default_multiple: int = 5,
) -> list[int]:
    """
    选择需要绘制的 episode。

    规则：
    1. 如果用户指定了 --episodes，则只绘制这些 episode；
    2. 如果未指定，则默认绘制 5 的倍数；
    3. 如果没有找到 5 的倍数，则退化为绘制最后一个 episode。
    """
    all_set = set(all_episodes)

    if episodes_arg is not None:
        valid = [ep for ep in episodes_arg if ep in all_set]
        invalid = [ep for ep in episodes_arg if ep not in all_set]

        if invalid:
            print(f"[WARN] 以下 episode 不存在，已跳过: {invalid}")

        if not valid:
            raise ValueError(
                f"--episodes 指定的 episode 均不存在。"
                f"当前 CSV 中可用 episode 范围: {min(all_episodes)} ~ {max(all_episodes)}"
            )

        return valid

    # 默认：只绘制 5 的倍数
    multiple = max(int(default_multiple), 1)
    target = [ep for ep in all_episodes if ep % multiple == 0]

    if not target:
        print(
            f"[WARN] 没有找到 {multiple} 的倍数 episode，"
            f"默认绘制最后一个 episode: {all_episodes[-1]}"
        )
        target = [all_episodes[-1]]

    return target


# =============================================================================
# tls_id 网格位置解析
# =============================================================================

def extract_grid_pos(tls_id: str) -> tuple[int, int] | None:
    """
    支持以下形式：
    - nt00, nt01, nt44
    - tl_0_0, tls_0_0
    - 任意以两个数字结尾的 id
    """
    s = str(tls_id)

    m = re.search(r"nt(\d)(\d)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r"(?:tl|tls)[_\-]?(\d+)[_\-](\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r"(\d)(\d)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None


# =============================================================================
# 构造 5×5 权重矩阵
# =============================================================================

def build_matrices(
    df_ep: pd.DataFrame,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对单个 episode 内的数据按 tls_id 聚合，构造：
    - mat_em:  5×5 w_em
    - mat_eff: 5×5 w_eff
    - labels:  5×5 tls_id 标签
    """
    g = (
        df_ep.groupby("tls_id", as_index=False)
        .agg(
            w_em=("w_em", "mean"),
            w_eff=("w_eff", "mean"),
        )
        .sort_values("tls_id")
        .reset_index(drop=True)
    )

    mat_em = np.full((grid_size, grid_size), np.nan, dtype=float)
    mat_eff = np.full((grid_size, grid_size), np.nan, dtype=float)
    labels = np.full((grid_size, grid_size), "", dtype=object)

    tls_ids = g["tls_id"].astype(str).tolist()
    parsed = {tid: extract_grid_pos(tid) for tid in tls_ids}
    valid_positions = [pos for pos in parsed.values() if pos is not None]

    # 优先按 tls_id 中解析出的行列位置放置
    if len(valid_positions) >= max(1, len(tls_ids) // 2):
        rows = [r for r, _ in valid_positions]
        cols = [c for _, c in valid_positions]

        # 兼容 1~5 编号，自动平移到 0~4
        row_offset = min(rows) if max(rows) - min(rows) + 1 <= grid_size else 0
        col_offset = min(cols) if max(cols) - min(cols) + 1 <= grid_size else 0

        for _, row in g.iterrows():
            tid = str(row["tls_id"])
            pos = parsed.get(tid)

            if pos is None:
                continue

            r, c = pos
            r -= row_offset
            c -= col_offset

            if 0 <= r < grid_size and 0 <= c < grid_size:
                mat_em[r, c] = float(row["w_em"])
                mat_eff[r, c] = float(row["w_eff"])
                labels[r, c] = tid

        return mat_em, mat_eff, labels

    # 如果 tls_id 解析失败，则按 tls_id 排序后的顺序填入 5×5
    n = min(grid_size * grid_size, len(g))
    for idx in range(n):
        r = idx // grid_size
        c = idx % grid_size

        mat_em[r, c] = float(g.loc[idx, "w_em"])
        mat_eff[r, c] = float(g.loc[idx, "w_eff"])
        labels[r, c] = str(g.loc[idx, "tls_id"])

    return mat_em, mat_eff, labels


# =============================================================================
# 绘图
# =============================================================================

def draw_heatmap(
    ax,
    data: np.ndarray,
    labels: np.ndarray,
    title: str,
):
    im = ax.imshow(data, vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                continue

            text = f"{data[i, j]:.3f}"
            if labels[i, j]:
                text = f"{labels[i, j]}\n{text}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    return im


def plot_one_episode(
    df: pd.DataFrame,
    episode: int,
    out_dir: Path,
    grid_size: int,
) -> Path:
    """
    绘制单个 episode 的 w_em / w_eff 双热力图。
    """
    df_ep = df[df["episode"] == episode].copy()

    if df_ep.empty:
        raise ValueError(f"episode={episode} 没有数据。")

    mat_em, mat_eff, labels = build_matrices(df_ep, grid_size=grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = draw_heatmap(
        axes[0],
        mat_em,
        labels,
        title=f"Emission Weight w_em\nEpisode {episode}",
    )
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = draw_heatmap(
        axes[1],
        mat_eff,
        labels,
        title=f"Efficiency Weight w_eff\nEpisode {episode}",
    )
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    out_path = out_dir / f"fig4_weight_heatmap_5x5_ep{episode:03d}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    BASE_DIR = Path(r"E:\DRL_project\network_marl_batch_v5\results")
    run_dir = BASE_DIR / "grid_5x5_20260428_112944"
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "结果分析" / "fig4_weight_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "tls_step_metrics.csv"
    df = pd.read_csv(csv_path)

    required = {"episode", "tls_id", "w_em", "w_eff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tls_step_metrics.csv 缺少字段: {sorted(missing)}")

    grid_size = 5
    episodes = list(range(5, 81, 5))  # 5,10,15,...,80
    default_multiple = 5

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["w_em"] = pd.to_numeric(df["w_em"], errors="coerce")
    df["w_eff"] = pd.to_numeric(df["w_eff"], errors="coerce")

    df = df.dropna(subset=["episode", "tls_id", "w_em", "w_eff"]).copy()
    df["episode"] = df["episode"].astype(int)

    if df.empty:
        raise ValueError("tls_step_metrics.csv 中没有有效 w_em / w_eff 数据。")

    all_episodes = sorted(df["episode"].unique().tolist())

    episodes_arg = episodes
    target_episodes = select_target_episodes(
        all_episodes=all_episodes,
        episodes_arg=episodes_arg,
        default_multiple=int(default_multiple),
    )

    print(f"[INFO] run_dir = {run_dir}")
    print(f"[INFO] csv_path = {csv_path}")
    print(f"[INFO] available episodes: {all_episodes[0]} ~ {all_episodes[-1]}")
    print(f"[INFO] target episodes = {target_episodes}")
    print(f"[INFO] output dir = {out_dir}")

    saved_paths = []

    for ep in target_episodes:
        out_path = plot_one_episode(
            df=df,
            episode=int(ep),
            out_dir=out_dir,
            grid_size=int(grid_size),
        )
        saved_paths.append(out_path)
        print(f"[OK] saved: {out_path}")

    print(f"[DONE] 共输出 {len(saved_paths)} 张热力图。")


if __name__ == "__main__":
    main()