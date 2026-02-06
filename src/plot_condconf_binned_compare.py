"""
Plot binned split-vs-condconf diagnostics for one experiment and one ICL length.

Inputs:
  1) Summary CSV from eval_icl_lr2x_condconf.py
     (contains split_cov_bin_XX / condconf_cov_bin_XX and mean widths per bin)
  2) Optional per-point details CSV from the same evaluator
     (contains split_width / condconf_width / diag_bin) for width boxplots.

Outputs:
  - <out-dir>/<stem>_binned_coverage_<exp>_L<icl>.png
  - <out-dir>/<stem>_binned_width_box_<exp>_L<icl>.png  (requires --details-csv)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _infer_exp_name(run_dir: str) -> str:
    s = (run_dir or "").replace("\\", "/").strip("/")
    if not s:
        return ""
    parts = [p for p in s.split("/") if p]
    if "results" in parts:
        i = parts.index("results")
        if i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def _safe_filename(s: str) -> str:
    s = s.strip()
    if not s:
        return "exp"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _finite_mean(values: List[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(v.mean())


def _pick_rows(summary_csv: Path, exp: str, icl_len: int, task: str) -> List[dict]:
    rows = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp_name = _infer_exp_name(r.get("run_dir", "") or "") or (r.get("exp", "") or "")
            if exp_name != exp:
                continue
            try:
                L = int(float(r.get("icl_len", "nan")))
            except Exception:
                continue
            if L != int(icl_len):
                continue
            if task and (r.get("task", "") != task):
                continue
            rows.append(r)
    return rows


def _resolve_task(rows: List[dict], task: str) -> str:
    if task:
        return task
    tasks = sorted(set((r.get("task", "") or "") for r in rows))
    if len(tasks) == 1:
        return tasks[0]
    raise SystemExit(f"Multiple tasks found {tasks}; please pass --task explicitly.")


def _extract_bin_count(row: dict) -> int:
    keys = [k for k in row.keys() if k.startswith("diag_bin_n_")]
    return len(keys)


def _aggregate_bins(rows: List[dict], b: int) -> Dict[str, np.ndarray]:
    out = {
        "n": np.zeros((b,), dtype=np.float64),
        "split_cov": np.full((b,), np.nan, dtype=np.float64),
        "cond_cov": np.full((b,), np.nan, dtype=np.float64),
        "split_w_mean": np.full((b,), np.nan, dtype=np.float64),
        "cond_w_mean": np.full((b,), np.nan, dtype=np.float64),
    }
    for i in range(b):
        out["n"][i] = _finite_mean([float(r.get(f"diag_bin_n_{i:02d}", "nan")) for r in rows])
        out["split_cov"][i] = _finite_mean([float(r.get(f"split_cov_bin_{i:02d}", "nan")) for r in rows])
        out["cond_cov"][i] = _finite_mean([float(r.get(f"condconf_cov_bin_{i:02d}", "nan")) for r in rows])
        out["split_w_mean"][i] = _finite_mean([float(r.get(f"split_avg_width_bin_{i:02d}", "nan")) for r in rows])
        out["cond_w_mean"][i] = _finite_mean([float(r.get(f"condconf_avg_width_bin_{i:02d}", "nan")) for r in rows])
    return out


def _load_detail_widths(details_csv: Path, exp: str, icl_len: int, task: str, b: int) -> tuple[List[np.ndarray], List[np.ndarray]]:
    split_per_bin: List[List[float]] = [[] for _ in range(b)]
    cond_per_bin: List[List[float]] = [[] for _ in range(b)]
    with details_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exp_name = _infer_exp_name(r.get("run_dir", "") or "") or (r.get("exp", "") or "")
            if exp_name != exp:
                continue
            try:
                L = int(float(r.get("icl_len", "nan")))
            except Exception:
                continue
            if L != int(icl_len):
                continue
            if task and (r.get("task", "") != task):
                continue
            try:
                bi = int(float(r.get("diag_bin", "nan")))
            except Exception:
                continue
            if bi < 0 or bi >= b:
                continue
            split_per_bin[bi].append(float(r["split_width"]))
            cond_per_bin[bi].append(float(r["condconf_width"]))
    split = [np.asarray(v, dtype=np.float64) for v in split_per_bin]
    cond = [np.asarray(v, dtype=np.float64) for v in cond_per_bin]
    return split, cond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, help="Summary CSV from eval_icl_lr2x_condconf.py")
    parser.add_argument("--details-csv", default="", help="Optional binned details CSV for width boxplots")
    parser.add_argument("--exp", required=True, help="Experiment name, e.g. S01_gpt2_w32_d6_lr")
    parser.add_argument("--icl-len", type=int, required=True, help="ICL length to visualize")
    parser.add_argument("--task", default="", help="Optional exact task string from CSV")
    parser.add_argument("--out-dir", default="../results/_icl_curves")
    parser.add_argument("--title", default="CondConf binned diagnostics")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _pick_rows(summary_csv, exp=args.exp, icl_len=int(args.icl_len), task=args.task)
    if not rows:
        raise SystemExit(f"No matching rows for exp={args.exp}, icl_len={args.icl_len} in {summary_csv}")

    task_name = _resolve_task(rows, args.task)
    rows = [r for r in rows if (r.get("task", "") == task_name)]
    if not rows:
        raise SystemExit(f"No rows left after task filter: {task_name}")

    b = _extract_bin_count(rows[0])
    if b <= 0:
        raise SystemExit("Summary CSV has no diagnostic bins. Re-run evaluator with --diagnostic-bins > 0.")

    bins = np.arange(b, dtype=np.float64)
    agg = _aggregate_bins(rows, b=b)
    alpha = float(rows[0].get("alpha", "0.05"))
    x_features = rows[0].get("x_features", "")
    target = 1.0 - alpha

    # Coverage bars
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    width = 0.38
    ax.bar(bins - width / 2, agg["split_cov"], width=width, label="Split CP", alpha=0.85)
    ax.bar(bins + width / 2, agg["cond_cov"], width=width, label="CondConf", alpha=0.85)
    ax.axhline(target, color="black", linestyle="--", linewidth=1, alpha=0.7, label=f"target={target:.2f}")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"Difficulty bin (easy→hard, quantiles on {x_features})")
    ax.set_ylabel("Coverage")
    ax.set_title(f"{args.title} | coverage | {args.exp} | L={args.icl_len}")
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{i}\n(n={int(round(agg['n'][i]))})" for i in range(b)])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    cov_path = out_dir / f"{summary_csv.stem}_binned_coverage_{_safe_filename(args.exp)}_L{int(args.icl_len)}.png"
    fig.savefig(cov_path, dpi=160)
    plt.close(fig)
    print("Saved:", cov_path)

    # Width boxplots (requires details CSV).
    if args.details_csv:
        details_csv = Path(args.details_csv).resolve()
        split_by_bin, cond_by_bin = _load_detail_widths(
            details_csv,
            exp=args.exp,
            icl_len=int(args.icl_len),
            task=task_name,
            b=b,
        )
        if all(v.size == 0 for v in split_by_bin) and all(v.size == 0 for v in cond_by_bin):
            raise SystemExit(f"No matching detail rows for exp={args.exp}, icl_len={args.icl_len} in {details_csv}")

        centers = np.arange(b, dtype=np.float64)
        pos_split = centers - 0.18
        pos_cond = centers + 0.18

        fig, ax = plt.subplots(figsize=(10.8, 5.0))
        bp_split = ax.boxplot(
            split_by_bin,
            positions=pos_split,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
        )
        bp_cond = ax.boxplot(
            cond_by_bin,
            positions=pos_cond,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
        )
        for box in bp_split["boxes"]:
            box.set(facecolor="#4C78A8", alpha=0.45, edgecolor="#4C78A8")
        for box in bp_cond["boxes"]:
            box.set(facecolor="#F58518", alpha=0.45, edgecolor="#F58518")
        for key in ("medians", "whiskers", "caps"):
            for art in bp_split[key]:
                art.set(color="#4C78A8")
            for art in bp_cond[key]:
                art.set(color="#F58518")

        ax.set_xlabel(f"Difficulty bin (easy→hard, quantiles on {x_features})")
        ax.set_ylabel("Width (2*q(x))")
        ax.set_title(f"{args.title} | width boxplot | {args.exp} | L={args.icl_len}")
        ax.set_xticks(centers)
        ax.set_xticklabels([str(i) for i in range(b)])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(
            handles=[
                Patch(facecolor="#4C78A8", edgecolor="#4C78A8", alpha=0.45, label="Split CP"),
                Patch(facecolor="#F58518", edgecolor="#F58518", alpha=0.45, label="CondConf"),
            ]
        )
        fig.tight_layout()
        width_path = out_dir / f"{summary_csv.stem}_binned_width_box_{_safe_filename(args.exp)}_L{int(args.icl_len)}.png"
        fig.savefig(width_path, dpi=160)
        plt.close(fig)
        print("Saved:", width_path)
    else:
        print("[INFO] --details-csv not provided; skipped width boxplot.")


if __name__ == "__main__":
    main()

