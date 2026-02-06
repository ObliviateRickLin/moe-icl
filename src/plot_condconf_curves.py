"""
Plot conditional conformal curves from a CSV produced by eval_icl_lr2x_condconf.py.

This is the analogue of the existing ICL curve/conformal plotting scripts, but it
reads precomputed rows from CSV (useful when evaluation is run via a loop).

Outputs:
  - <out-dir>/<stem>_coverage.png
  - <out-dir>/<stem>_width.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Curve:
    icl_len: np.ndarray
    coverage: np.ndarray
    width: np.ndarray
    coverage_ci_low: Optional[np.ndarray] = None
    coverage_ci_high: Optional[np.ndarray] = None
    width_ci_low: Optional[np.ndarray] = None
    width_ci_high: Optional[np.ndarray] = None


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


def _finite_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def load_curves(csv_path: Path) -> Dict[str, Curve]:
    per_exp: Dict[str, Dict[int, List[float]]] = {}
    per_exp_cov: Dict[str, Dict[int, List[float]]] = {}
    per_exp_cov_ci_low: Dict[str, Dict[int, List[float]]] = {}
    per_exp_cov_ci_high: Dict[str, Dict[int, List[float]]] = {}
    per_exp_w_ci_low: Dict[str, Dict[int, List[float]]] = {}
    per_exp_w_ci_high: Dict[str, Dict[int, List[float]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            run_dir = r.get("run_dir", "") or ""
            exp = _infer_exp_name(run_dir) or (r.get("exp", "") or "")
            if not exp:
                continue
            L = int(float(r["icl_len"]))
            cov = float(r["coverage"])
            width = float(r.get("avg_width", r.get("interval_width", "nan")))
            cov_ci_low = float(r.get("coverage_bootstrap_low", "nan"))
            cov_ci_high = float(r.get("coverage_bootstrap_high", "nan"))
            w_ci_low = float(r.get("width_bootstrap_low", "nan"))
            w_ci_high = float(r.get("width_bootstrap_high", "nan"))

            per_exp.setdefault(exp, {}).setdefault(L, []).append(width)
            per_exp_cov.setdefault(exp, {}).setdefault(L, []).append(cov)
            per_exp_cov_ci_low.setdefault(exp, {}).setdefault(L, []).append(cov_ci_low)
            per_exp_cov_ci_high.setdefault(exp, {}).setdefault(L, []).append(cov_ci_high)
            per_exp_w_ci_low.setdefault(exp, {}).setdefault(L, []).append(w_ci_low)
            per_exp_w_ci_high.setdefault(exp, {}).setdefault(L, []).append(w_ci_high)

    curves: Dict[str, Curve] = {}
    for exp in per_exp.keys():
        lens = sorted(set(per_exp[exp].keys()) & set(per_exp_cov[exp].keys()))
        if not lens:
            continue
        widths = np.array([float(np.mean(per_exp[exp][L])) for L in lens], dtype=np.float64)
        covs = np.array([float(np.mean(per_exp_cov[exp][L])) for L in lens], dtype=np.float64)
        cov_ci_low = np.array([_finite_mean(per_exp_cov_ci_low[exp][L]) for L in lens], dtype=np.float64)
        cov_ci_high = np.array([_finite_mean(per_exp_cov_ci_high[exp][L]) for L in lens], dtype=np.float64)
        w_ci_low = np.array([_finite_mean(per_exp_w_ci_low[exp][L]) for L in lens], dtype=np.float64)
        w_ci_high = np.array([_finite_mean(per_exp_w_ci_high[exp][L]) for L in lens], dtype=np.float64)

        have_cov_ci = bool(np.isfinite(cov_ci_low).any() and np.isfinite(cov_ci_high).any())
        have_w_ci = bool(np.isfinite(w_ci_low).any() and np.isfinite(w_ci_high).any())
        curves[exp] = Curve(
            icl_len=np.asarray(lens, dtype=np.int64),
            coverage=covs,
            width=widths,
            coverage_ci_low=cov_ci_low if have_cov_ci else None,
            coverage_ci_high=cov_ci_high if have_cov_ci else None,
            width_ci_low=w_ci_low if have_w_ci else None,
            width_ci_high=w_ci_high if have_w_ci else None,
        )
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="compare_lr2x_condconf_summary_s_series_alpha0p05.csv",
        help="Path to condconf summary CSV (as produced by eval_icl_lr2x_condconf.py).",
    )
    parser.add_argument("--out-dir", default="../results/_icl_curves")
    parser.add_argument("--title", default="Conditional conformal")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--width-mode",
        choices=["avg", "median"],
        default="avg",
        help="Which width statistic to plot. "
        "'avg' uses avg_width (mean width). 'median' uses width_p50 (median width). "
        "Note: bootstrap CIs in the CSV are for the mean and are only shown for width-mode=avg.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(csv_path)
    if not curves:
        raise SystemExit(f"No curves found in {csv_path}")

    exp_names = sorted(curves.keys(), key=_sort_key)
    target = 1.0 - float(args.alpha)
    stem = csv_path.stem

    # Coverage
    plt.figure(figsize=(8.5, 4.8))
    for exp in exp_names:
        c = curves[exp]
        plt.plot(c.icl_len, c.coverage, marker="o", linewidth=2, label=exp)
        if c.coverage_ci_low is not None and c.coverage_ci_high is not None:
            plt.fill_between(c.icl_len, c.coverage_ci_low, c.coverage_ci_high, alpha=0.15)
    plt.axhline(target, color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"target {target:.2f}")
    plt.title(f"{args.title} | coverage")
    plt.xlabel("ICL length")
    plt.ylabel("coverage")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_cov = out_dir / f"{stem}_coverage.png"
    plt.savefig(out_cov, dpi=160)
    plt.close()

    # Width
    plt.figure(figsize=(8.5, 4.8))
    for exp in exp_names:
        c = curves[exp]
        if args.width_mode == "median":
            # For median, prefer width_p50 if present; otherwise fallback to c.width (avg_width).
            # We re-load per-row widths to avoid changing the Curve dataclass structure.
            widths_by_L = {}
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    run_dir = r.get("run_dir", "") or ""
                    exp_name = _infer_exp_name(run_dir) or (r.get("exp", "") or "")
                    if exp_name != exp:
                        continue
                    L = int(float(r.get("icl_len", "nan")))
                    w = float(r.get("width_p50", "nan"))
                    if np.isfinite(w):
                        widths_by_L.setdefault(L, []).append(w)
            if widths_by_L:
                lens = c.icl_len
                med = np.array([float(np.mean(widths_by_L.get(int(L), [np.nan]))) for L in lens], dtype=np.float64)
                plt.plot(lens, med, marker="o", linewidth=2, label=exp)
            else:
                plt.plot(c.icl_len, c.width, marker="o", linewidth=2, label=exp)
        else:
            plt.plot(c.icl_len, c.width, marker="o", linewidth=2, label=exp)
            if c.width_ci_low is not None and c.width_ci_high is not None:
                plt.fill_between(c.icl_len, c.width_ci_low, c.width_ci_high, alpha=0.15)
    width_title = "avg interval width" if args.width_mode == "avg" else "median interval width"
    plt.title(f"{args.title} | {width_title}")
    plt.xlabel("ICL length")
    plt.ylabel("width (2*q(x))")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_w = out_dir / f"{stem}_width.png"
    plt.savefig(out_w, dpi=160)
    plt.close()

    print("Saved:", out_cov)
    print("Saved:", out_w)


if __name__ == "__main__":
    main()
