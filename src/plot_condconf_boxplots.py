"""
Plot conditional conformal interval width boxplots from a CSV produced by eval_icl_lr2x_condconf.py.

This script uses summary statistics computed per ICL length (per experiment) and does *not* need
raw per-example widths.

Expected columns (produced by eval_icl_lr2x_condconf.py):
  - icl_len
  - avg_width
  - width_p25, width_p50, width_p75
  - width_whislo, width_whishi

Outputs (per experiment):
  - <out-dir>/<stem>_width_box_<exp>.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _infer_exp_name(run_dir: str) -> str:
    """
    Try to infer the experiment name (e.g. S01_gpt2_...) from a run_dir.

    Supports both:
      - .../results/<EXP>/<RUN_ID>
      - ...\\results\\<EXP>\\<RUN_ID>
    Fallback: parent directory name.
    """
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


def _sort_key(name: str) -> Tuple[int, str]:
    # Prefer Sxx numeric ordering when present.
    if len(name) >= 3 and name[0].upper() == "S" and name[1:3].isdigit():
        return (int(name[1:3]), name)
    return (10_000, name)


def _safe_filename(s: str) -> str:
    s = s.strip()
    if not s:
        return "exp"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def load_box_stats(csv_path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    per_exp: Dict[str, Dict[int, Dict[str, float]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            run_dir = r.get("run_dir", "") or ""
            exp = _infer_exp_name(run_dir) or (r.get("exp", "") or "")
            if not exp:
                continue
            if "width_p25" not in r or "width_p50" not in r or "width_p75" not in r:
                continue
            if "width_whislo" not in r or "width_whishi" not in r:
                continue

            L = int(float(r["icl_len"]))
            stats = {
                "med": float(r["width_p50"]),
                "q1": float(r["width_p25"]),
                "q3": float(r["width_p75"]),
                "whislo": float(r["width_whislo"]),
                "whishi": float(r["width_whishi"]),
                "mean": float(r.get("avg_width", "nan")),
            }

            per_exp.setdefault(exp, {})
            # If duplicates exist for the same (exp, L), keep the first row.
            per_exp[exp].setdefault(L, stats)

    return per_exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="compare_lr2x_condconf_summary_s_series_alpha0p05.csv",
        help="Path to condconf summary CSV (as produced by eval_icl_lr2x_condconf.py).",
    )
    parser.add_argument("--out-dir", default="../results/_icl_curves")
    parser.add_argument("--title", default="Conditional conformal")
    parser.add_argument("--exp", default="", help="If set, plot only a single experiment name (exact match).")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_exp = load_box_stats(csv_path)
    if not per_exp:
        raise SystemExit(f"No boxplot stats found in {csv_path}")

    exps: List[str]
    if args.exp:
        if args.exp not in per_exp:
            raise SystemExit(f"Experiment '{args.exp}' not found in {csv_path}")
        exps = [args.exp]
    else:
        exps = sorted(per_exp.keys(), key=_sort_key)

    stem = csv_path.stem
    for exp in exps:
        by_L = per_exp[exp]
        lens = sorted(by_L.keys())
        if not lens:
            continue

        bxp_stats = []
        for L in lens:
            s = by_L[L]
            bxp_stats.append(
                {
                    "label": str(L),
                    "med": s["med"],
                    "q1": s["q1"],
                    "q3": s["q3"],
                    "whislo": s["whislo"],
                    "whishi": s["whishi"],
                    "fliers": [],
                }
            )

        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        ax.bxp(bxp_stats, positions=lens, showfliers=False, widths=0.6)
        ax.set_title(f"{args.title} | width boxplot | {exp}")
        ax.set_xlabel("ICL length")
        ax.set_ylabel("width (2*q(x))")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"{stem}_width_box_{_safe_filename(exp)}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

