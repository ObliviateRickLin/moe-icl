"""
Beautiful plotting for SpeedCP results:
- S32 (mixnoise4, 4 noise tasks)
- S20, S37, S38, S39 (single-noise): L=80 box+coverage, and curves coverage/avg_width vs L

Improvements:
- Using a clean, professional style.
- Distinct color palette for noise levels.
- Better labels, grid, and formatting.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Try to use seaborn for better aesthetics if available
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    HAS_SEABORN = True
except ImportError:
    plt.style.use('ggplot')
    HAS_SEABORN = False

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "_icl_curves"
OUT_DIR = RESULTS_DIR

# Define a nice color palette for 4 conditions (noise levels)
COLORS = ["#5DADE2", "#48C9B0", "#F4D03F", "#EB984E"] # Blue, Green, Yellow, Orange

def load_row(path: Path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    return r[0] if r else None

def parse_noise_from_task(task_str: str) -> float:
    import re
    m = re.search(r"noise_std=([\d.]+)", task_str)
    return float(m.group(1)) if m else 0.0

def box_stats_from_row(row: dict, label: str):
    return {
        "label": label,
        "q1": float(row["width_p25"]),
        "med": float(row["width_p50"]),
        "q3": float(row["width_p75"]),
        "whislo": float(row["width_whislo"]),
        "whishi": float(row["width_whishi"]),
    }

def setup_ax(ax, title, ylabel):
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_ylabel(ylabel, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_s32():
    rows = []
    for i in range(4):
        p = RESULTS_DIR / f"compare_lr2x_speedcp_summary_S32_gpt2_w64_d12_mixnoise4_80x40_task{i}_alpha0p05.csv"
        row = load_row(p)
        if row:
            row["_noise"] = parse_noise_from_task(row["task"])
            rows.append(row)

    if not rows: return
    rows.sort(key=lambda r: r["_noise"])
    
    labels = [f"σ={r['_noise']}" for r in rows]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Box plot
    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
    for median in res['medians']:
        median.set_color('firebrick')
        median.set_linewidth(2)
    
    setup_ax(ax1, "S32 (MixNoise-4) Interval Widths", "Interval half-width (q)")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 2. Coverage
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor='black', width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S32 (MixNoise-4) Coverage Performance", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc='lower right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S32_box_coverage.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_single_noise():
    paths = [
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_s_nlr80_series_alpha0p05.csv", "S20 (σ=0.1)"),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S37_gpt2_w64_d12_nlr80x40_noise025_alpha0p05.csv", "S37 (σ=0.25)"),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S38_gpt2_w64_d12_nlr80x40_noise05_alpha0p05.csv", "S38 (σ=0.5)"),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S39_gpt2_w64_d12_nlr80x40_noise10_alpha0p05.csv", "S39 (σ=1.0)"),
    ]

    rows = []
    for p, lab in paths:
        if not p.exists(): continue
        with p.open(newline="", encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        if not r: continue
        row = next((rr for rr in r if int(rr.get("icl_len", 80)) == 80), r[0])
        row["_label"] = lab
        rows.append(row)

    if not rows: return
    
    labels = [r["_label"] for r in rows]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Box plot
    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
    for median in res['medians']:
        median.set_color('firebrick')
        median.set_linewidth(2)
        
    setup_ax(ax1, "Single-Noise Experiments (L=80) Widths", "Interval half-width (q)")
    ax1.set_xticklabels(labels, rotation=15)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 2. Coverage
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor='black', width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "Single-Noise Experiments (L=80) Coverage", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc='lower right')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S20S37S38S39_box_coverage.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_s52_s55_box_coverage():
    """
    Plot (in one figure) width boxplots + coverage bars for the latest S52-S55 runs.

    Expected inputs (produced by eval_icl_lr2x_speedcp.py):
      - compare_lr2x_speedcp_summary_S52_..._alpha0p05.csv
      - compare_lr2x_speedcp_summary_S53_..._alpha0p05.csv
      - compare_lr2x_speedcp_summary_S54_..._alpha0p05.csv
      - compare_lr2x_speedcp_summary_S55_..._alpha0p05.csv
    """
    paths = [
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S52_gpt2_w256_d12_nlr80x40_noise01_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S53_gpt2_w256_d12_nlr80x40_noise025_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S54_gpt2_w256_d12_nlr80x40_noise05_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S55_gpt2_w256_d12_nlr80x40_noise10_alpha0p05.csv",
    ]

    rows = []
    for p in paths:
        row = load_row(p)
        if not row:
            continue
        row["_noise"] = parse_noise_from_task(row.get("task", ""))
        row["_exp"] = row.get("exp", p.stem)
        rows.append(row)

    if not rows:
        return

    # Sort by noise so colors/labels are consistent.
    rows.sort(key=lambda r: r["_noise"])

    labels = [f"S{r['_exp'][1:3]} (σ={r['_noise']})" if str(r["_exp"]).startswith("S") else f"σ={r['_noise']}" for r in rows]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Box plot
    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for median in res["medians"]:
        median.set_color("firebrick")
        median.set_linewidth(2)

    setup_ax(ax1, "S52–S55 (L=80) Interval Widths", "Interval half-width (q)")
    ax1.set_xticklabels(labels, rotation=15)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    # 2. Coverage
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor="black", width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S52–S55 (L=80) Coverage", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc="lower right")

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S52S53S54S55_box_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_s60_s63_box_coverage():
    """
    Plot (in one figure) width boxplots + coverage bars for S60-S63 (2nn, L=200).

    Expected inputs (produced by eval_icl_lr2x_speedcp.py):
      - compare_lr2x_speedcp_summary_S60_..._noise01_alpha0p05.csv
      - compare_lr2x_speedcp_summary_S61_..._noise025_alpha0p05.csv
      - compare_lr2x_speedcp_summary_S62_..._noise05_alpha0p05.csv
      - compare_lr2x_speedcp_summary_S63_..._noise10_alpha0p05.csv
    """
    paths = [
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S60_gpt2_w256_d12_n2nn200x40_noise01_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S61_gpt2_w256_d12_n2nn200x40_noise025_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S62_gpt2_w256_d12_n2nn200x40_noise05_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S63_gpt2_w256_d12_n2nn200x40_noise10_alpha0p05.csv",
    ]

    rows = []
    for p in paths:
        row = load_row(p)
        if not row:
            continue
        row["_noise"] = parse_noise_from_task(row.get("task", ""))
        row["_exp"] = row.get("exp", p.stem)
        rows.append(row)

    if not rows:
        return

    rows.sort(key=lambda r: r["_noise"])

    labels = [f"S{r['_exp'][1:3]} (σ={r['_noise']})" if str(r["_exp"]).startswith("S") else f"σ={r['_noise']}" for r in rows]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for median in res["medians"]:
        median.set_color("firebrick")
        median.set_linewidth(2)

    setup_ax(ax1, "S60–S63 (2nn, L=200) Interval Widths", "Interval half-width (q)")
    ax1.set_xticklabels(labels, rotation=15)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor="black", width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S60–S63 (2nn, L=200) Coverage", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc="lower right")

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S60S61S62S63_box_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_s56_s59_box_coverage():
    """
    Plot (in one figure) width boxplots + coverage bars for S56-S59 (nqr, L=200).
    """
    paths = [
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S56_gpt2_w256_d12_nqr200x40_noise01_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S57_gpt2_w256_d12_nqr200x40_noise025_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S58_gpt2_w256_d12_nqr200x40_noise05_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S59_gpt2_w256_d12_nqr200x40_noise10_alpha0p05.csv",
    ]

    rows = []
    for p in paths:
        row = load_row(p)
        if not row:
            continue
        row["_noise"] = parse_noise_from_task(row.get("task", ""))
        row["_exp"] = row.get("exp", p.stem)
        rows.append(row)

    if not rows:
        return

    rows.sort(key=lambda r: r["_noise"])

    labels = [f"S{r['_exp'][1:3]} (σ={r['_noise']})" if str(r["_exp"]).startswith("S") else f"σ={r['_noise']}" for r in rows]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for median in res["medians"]:
        median.set_color("firebrick")
        median.set_linewidth(2)

    setup_ax(ax1, "S56–S59 (nqr, L=200) Interval Widths", "Interval half-width (q)")
    ax1.set_xticklabels(labels, rotation=15)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor="black", width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S56–S59 (nqr, L=200) Coverage", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc="lower right")

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S56S57S58S59_box_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_s64_s67_box_coverage():
    """
    Plot (in one figure) width boxplots + coverage bars for S64-S67 (ndt, L=200).
    """
    paths = [
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S64_gpt2_w256_d12_ndt200x40_noise01_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S65_gpt2_w256_d12_ndt200x40_noise025_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S66_gpt2_w256_d12_ndt200x40_noise05_alpha0p05.csv",
        RESULTS_DIR / "compare_lr2x_speedcp_summary_S67_gpt2_w256_d12_ndt200x40_noise10_alpha0p05.csv",
    ]

    rows = []
    for p in paths:
        row = load_row(p)
        if not row:
            continue
        row["_noise"] = parse_noise_from_task(row.get("task", ""))
        row["_exp"] = row.get("exp", p.stem)
        rows.append(row)

    if not rows:
        return

    rows.sort(key=lambda r: r["_noise"])

    labels = [f"S{r['_exp'][1:3]} (σ={r['_noise']})" if str(r["_exp"]).startswith("S") else f"σ={r['_noise']}" for r in rows]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for median in res["medians"]:
        median.set_color("firebrick")
        median.set_linewidth(2)

    setup_ax(ax1, "S64–S67 (ndt, L=200) Interval Widths", "Interval half-width (q)")
    ax1.set_xticklabels(labels, rotation=15)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor="black", width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S64–S67 (ndt, L=200) Coverage", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc="lower right")

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S64S65S66S67_box_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_s69_s73_s74_s75_box_coverage():
    """Plot (left) width boxplot + (right) coverage for S69/S73/S74/S75 maxdim L=200 SpeedCP."""
    paths = [
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S73_gpt2_w512_d12_nlr201x10_maxdim_L200_ex1600_v2_alpha0p05.csv", "S73 (d=10)"),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S74_gpt2_w512_d12_nlr201x20_maxdim_L200_ex1600_v2_alpha0p05.csv", "S74 (d=20)"),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S75_gpt2_w512_d12_nlr201x40_maxdim_L200_ex1600_v2_alpha0p05.csv", "S75 (d=40)"),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S69_gpt2_w512_d12_nlr201x100_maxdim_L200_ex1600_v2_alpha0p05.csv", "S69 (d=100)"),
    ]
    rows = []
    labels = []
    for p, lab in paths:
        row = load_row(p)
        if row:
            rows.append(row)
            labels.append(lab)
    if not rows:
        return
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for median in res["medians"]:
        median.set_color("firebrick")
        median.set_linewidth(2)
    setup_ax(ax1, "S69/S73/S74/S75 Interval Widths (L=200)", "Interval half-width (q)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor="black", width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S69/S73/S74/S75 Coverage (L=200)", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc="lower right")
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S69S73S74S75_box_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT_DIR / "speedcp_S69S73S74S75_box_coverage.png")


def plot_s76_s78_s79_s80_box_coverage():
    """Plot (left) width boxplot + (right) coverage for S76/S78/S79/S80 from synthetic CSV (2NN σ=0.1, L=200)."""
    csv_path = RESULTS_DIR / "compare_lr2x_speedcp_summary_S76S78S79S80_2nn_noise01_synthetic_alpha0p05.csv"
    if not csv_path.exists():
        print("[SKIP] Synthetic CSV not found:", csv_path)
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 4:
        print("[SKIP] Synthetic CSV has < 4 rows:", csv_path)
        return
    # Use first 4 rows in order S76, S78, S79, S80 (as in CSV)
    rows = rows[:4]
    labels = [r.get("exp", "").replace("_gpt2_w256_d12_n2nn201x40_noise01", "") for r in rows]
    if not all(labels):
        labels = ["S76", "S78", "S79", "S80"]
    coverages = [float(r["coverage"]) * 100 for r in rows]
    bxp_stats = [box_stats_from_row(r, labels[i]) for i, r in enumerate(rows)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    res = ax1.bxp(bxp_stats, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(res["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
    for median in res["medians"]:
        median.set_color("firebrick")
        median.set_linewidth(2)
    setup_ax(ax1, "S76/S78/S79/S80 (2NN σ=0.1, L=200) Interval Widths", "Interval half-width (q)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, coverages, color=COLORS, alpha=0.8, edgecolor="black", width=0.6)
    ax2.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, label="Target 95%")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 105)
    setup_ax(ax2, "S76/S78/S79/S80 (2NN σ=0.1, L=200) Coverage", "Empirical Coverage (%)")
    ax2.legend(frameon=True, loc="lower right")
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "speedcp_S76S78S79S80_box_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT_DIR / "speedcp_S76S78S79S80_box_coverage.png")


def plot_single_noise_curves():
    """Plot coverage vs L and avg_width vs L for S20, S37, S38, S39 from curve CSVs."""
    paths = [
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_s_nlr80_series_curve_alpha0p05.csv", "S20 (σ=0.1)", COLORS[0]),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S37_gpt2_w64_d12_nlr80x40_noise025_curve_alpha0p05.csv", "S37 (σ=0.25)", COLORS[1]),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S38_gpt2_w64_d12_nlr80x40_noise05_curve_alpha0p05.csv", "S38 (σ=0.5)", COLORS[2]),
        (RESULTS_DIR / "compare_lr2x_speedcp_summary_S39_gpt2_w64_d12_nlr80x40_noise10_curve_alpha0p05.csv", "S39 (σ=1.0)", COLORS[3]),
    ]
    curves = []
    for p, label, color in paths:
        if not p.exists():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        lens = [int(float(r["icl_len"])) for r in rows]
        cov = [float(r["coverage"]) * 100 for r in rows]
        aw = [float(r["avg_width"]) for r in rows]
        curves.append((label, lens, cov, aw, color))

    if not curves:
        return

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    for label, lens, cov, aw, color in curves:
        ax1.plot(lens, cov, "o-", color=color, label=label, linewidth=2, markersize=6)
        ax2.plot(lens, aw, "o-", color=color, label=label, linewidth=2, markersize=6)

    ax1.axhline(y=95, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.8)
    ax1.set_xlabel("Number of examples (L)", labelpad=10)
    ax1.set_ylabel("Empirical Coverage (%)", labelpad=10)
    ax1.set_title("SpeedCP S20/S37/S38/S39 Coverage vs L", pad=15, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_ylim(0, 105)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "speedcp_S20S37S38S39_coverage_vs_L.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    ax2.set_xlabel("Number of examples (L)", labelpad=10)
    ax2.set_ylabel("Average interval width", labelpad=10)
    ax2.set_title("SpeedCP S20/S37/S38/S39 Average Width vs L", pad=15, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.7)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "speedcp_S20S37S38S39_avgwidth_vs_L.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    plot_s32()
    plot_single_noise()
    plot_s52_s55_box_coverage()
    plot_s60_s63_box_coverage()
    plot_s56_s59_box_coverage()
    plot_s64_s67_box_coverage()
    plot_s69_s73_s74_s75_box_coverage()
    plot_s76_s78_s79_s80_box_coverage()
    plot_single_noise_curves()
    print("Beautiful plots saved to results/_icl_curves/")
