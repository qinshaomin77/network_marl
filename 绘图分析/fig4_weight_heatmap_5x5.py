#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def resolve_run_dir(run_dir: str | None) -> Path:
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

def parse_episodes_arg(episodes_str: str | None) -> list[int] | None:
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

    multiple = max(int(default_multiple), 1)
    target = [ep for ep in all_episodes if ep % multiple == 0]

    if not target:
        print(
            f"[WARN] 没有找到 {multiple} 的倍数 episode，"
            f"默认绘制最后一个 episode: {all_episodes[-1]}"
        )
        target = [all_episodes[-1]]

    return target

def extract_grid_pos(tls_id: str) -> tuple[int, int] | None:
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

def build_matrices(
    df_ep: pd.DataFrame,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    if len(valid_positions) >= max(1, len(tls_ids) // 2):
        rows = [r for r, _ in valid_positions]
        cols = [c for _, c in valid_positions]

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

    n = min(grid_size * grid_size, len(g))
    for idx in range(n):
        r = idx // grid_size
        c = idx % grid_size

        mat_em[r, c] = float(g.loc[idx, "w_em"])
        mat_eff[r, c] = float(g.loc[idx, "w_eff"])
        labels[r, c] = str(g.loc[idx, "tls_id"])

    return mat_em, mat_eff, labels

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
