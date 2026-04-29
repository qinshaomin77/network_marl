#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


PROFILE_COLS = {
    "env_step_sec": "prof_env_step_sec",
    "lower_act_sec": "prof_lower_act_sec",
    "upper_act_sec": "prof_upper_act_sec",
    "lower_update_sec": "prof_lower_update_sec",
    "upper_update_sec": "prof_upper_update_sec",
    "logging_sec": "prof_logging_sec",
}


SUGGESTIONS = {
    "env_step_sec": "优先减少 TraCI 调用与日志采样频率，尽量走 subscription/cache，避免逐车高频 RPC。",
    "lower_act_sec": "检查下层网络前向是否存在 CPU/GPU 频繁拷贝；可尝试 batch 化、torch.compile、AMP。",
    "upper_act_sec": "上层前向可降低调用频率（调大 UPPER_K），并检查 edge 特征构造是否重复计算。",
    "lower_update_sec": "适度降低更新频率或降低每次更新计算量（如调 N_STEP、网络规模、梯度统计开销）。",
    "upper_update_sec": "可减少 UPPER_UPDATE_EPOCHS / batch 复杂度，或增加 rollout 减少触发频率。",
    "logging_sec": "降低每步日志记录频率，关闭高成本 raw 特征日志，仅采样关键 episode。",
}


def _to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarize(rows: List[Dict[str, str]]) -> Tuple[Dict[str, float], Dict[str, float], float, int]:
    totals = {k: 0.0 for k in PROFILE_COLS.keys()}
    episode_total = 0.0

    for r in rows:
        episode_total += _to_float(r.get("episode_elapsed_sec", "0"), 0.0)
        for k, col in PROFILE_COLS.items():
            totals[k] += _to_float(r.get(col, "0"), 0.0)

    measured_total = sum(totals.values())
    if measured_total <= 1e-12:
        shares = {k: 0.0 for k in totals.keys()}
    else:
        shares = {k: (v / measured_total) for k, v in totals.items()}
    return totals, shares, episode_total, len(rows)


def topk(shares: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(shares.items(), key=lambda x: x[1], reverse=True)[:k]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze training profile columns in episode_summary.csv")
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Path to episode_summary.csv. If empty, auto-find latest under results/*/logs/",
    )
    return p


def find_latest_csv(root: Path) -> Path:
    cands = list(root.glob("results/*/logs/episode_summary.csv"))
    if not cands:
        raise FileNotFoundError("No episode_summary.csv found under results/*/logs/")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else find_latest_csv(project_root)

    rows = load_rows(csv_path)
    totals, shares, episode_total, n_ep = summarize(rows)
    top3 = topk(shares, 3)

    print("=" * 72)
    print("Profile Analysis")
    print("=" * 72)
    print(f"source_csv: {csv_path}")
    print(f"episodes  : {n_ep}")
    print(f"sum_episode_elapsed_sec: {episode_total:.3f}")
    print("")

    print("Module Breakdown (by measured profile sum)")
    measured_total = sum(totals.values())
    for k in sorted(totals.keys()):
        sec = totals[k]
        pct = 100.0 * shares[k]
        print(f"- {k:16s} {sec:10.3f} sec   {pct:6.2f}%")
    print(f"- {'measured_total':16s} {measured_total:10.3f} sec")
    print("")

    print("Top 3 Bottlenecks")
    for i, (k, ratio) in enumerate(top3, start=1):
        print(f"{i}. {k} ({ratio*100:.2f}%)")
        print(f"   suggestion: {SUGGESTIONS.get(k, '无建议')}")

    print("=" * 72)


if __name__ == "__main__":
    main()

