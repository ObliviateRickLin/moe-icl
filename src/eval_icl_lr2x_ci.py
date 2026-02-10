"""
Plot ICL curves with bootstrap confidence intervals for the lr_2x curriculum runs (E141-152).

These runs share:
  - n_positions = 41 (ICL lengths 1..40)
  - a curriculum that increases dims/points up to (n_dims=40, n_points=41)
  - a simple linear_regression task

We follow the same "CI-enabled" pattern as eval_icl_mix_noise2_ci.py:
  - Evaluate once at max_n_points (default 41), which yields per-position metrics.
  - Use position i (1..40) as the "last-point error" for ICL length i.
  - Use aggregate_metrics() outputs from src/eval.py:
      mean, bootstrap_low, bootstrap_high
    to draw a shaded confidence band for the mean error.

Outputs (under results/_icl_curves/):
  - compare_lr2x_summary_ci.csv
  - compare_lr2x_<family>_<task>_ci.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ckpt_utils import latest_run_dir, select_ckpt_path
from eval import eval_model
from models import build_model


EXPS = {
    "gpt2": [
        "E141_gpt2_dense_lr_2x",
        "E142_gpt2_moe4_lr_2x",
        "E143_gpt2_moe4_last3_lr_2x",
    ],
    "llama_hf": [
        "E144_llama_dense_lr_2x",
        "E145_llama_moe4_lr_2x",
        "E146_llama_moe4_last3_lr_2x",
    ],
    "qwen_hf": [
        "E147_qwen_dense_lr_2x",
        "E148_qwen_moe4_lr_2x",
        "E149_qwen_moe4_last3_lr_2x",
    ],
    "gemma_hf": [
        "E150_gemma_dense_lr_2x",
        "E151_gemma_moe4_lr_2x",
        "E152_gemma_moe4_last3_lr_2x",
    ],
    # S-series: Dense GPT-2 scaling experiments for linear regression
    "s_series": [
        "S01_gpt2_w32_d6_lr",
        "S02_gpt2_w64_d6_lr",
        "S03_gpt2_w128_d6_lr",
        "S04_gpt2_w256_d6_lr",
        "S05_gpt2_w64_d2_lr",
        "S06_gpt2_w64_d4_lr",
        "S07_gpt2_w64_d8_lr",
        "S08_gpt2_w64_d12_lr",
        "S09_gpt2_tiny_lr",
        "S10_gpt2_small_lr",
        "S11_gpt2_medium_lr",
        "S12_gpt2_large_lr",
    ],
    # S13-S24: Dense GPT-2 scaling on noisy linear regression with n_dims=80 and max ICL length=40.
    "s_nlr80_series": [
        "S13_gpt2_w32_d6_nlr80x40",
        "S14_gpt2_w64_d6_nlr80x40",
        "S15_gpt2_w128_d6_nlr80x40",
        "S16_gpt2_w256_d6_nlr80x40",
        "S17_gpt2_w64_d2_nlr80x40",
        "S18_gpt2_w64_d4_nlr80x40",
        "S19_gpt2_w64_d8_nlr80x40",
        "S20_gpt2_w64_d12_nlr80x40",
        "S21_gpt2_tiny_nlr80x40",
        "S22_gpt2_small_nlr80x40",
        "S23_gpt2_medium_nlr80x40",
        "S24_gpt2_large_nlr80x40",
    ],
}


def build_conf(model_cfg: dict):
    # SimpleNamespace-like object with keys() for EncoderTF defaults
    class C:
        pass

    c = C()
    for k, v in model_cfg.items():
        setattr(c, k, v)
    c.keys = lambda: c.__dict__.keys()
    return c


def _task_to_label(task_name: str, task_kwargs: dict) -> str:
    # Keep consistent with other scripts (kwargs appended when present).
    if not task_kwargs:
        return task_name
    items = ",".join([f"{k}={task_kwargs[k]}" for k in sorted(task_kwargs.keys())])
    return f"{task_name}({items})"


def _is_classification(task_name: str) -> bool:
    # Only linear_classification uses accuracy in this repo.
    return "classification" in task_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--family", choices=list(EXPS.keys()), required=True)
    parser.add_argument("--exp", default="", help="Optional single experiment name")
    parser.add_argument(
        "--prefer-model-step",
        type=int,
        default=300000,
        help="Prefer loading model_{step}.pt if present; otherwise fall back to an earlier model_{k}.pt or state.pt.",
    )
    parser.add_argument("--num-eval-examples", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-n-points",
        type=int,
        default=41,
        help="Evaluate once at this n_points; ICL lengths are 1..(max_n_points-1).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "_icl_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_list = EXPS[args.family]
    if args.exp:
        if args.exp not in exp_list:
            raise SystemExit(f"--exp must be one of: {exp_list}")
        exp_list = [args.exp]

    rows = []
    curves = {}  # task_label -> exp -> dict(mean/low/high)

    for exp in exp_list:
        exp_dir = results_dir / exp
        rd = latest_run_dir(exp_dir, prefer_step=args.prefer_model_step)
        if rd is None:
            print("[SKIP]", exp, "(no checkpoint)")
            continue

        cfg = yaml.safe_load((rd / "config.yaml").open("r"))
        model_cfg = cfg["model"]
        training_cfg = cfg["training"]
        tasks = training_cfg.get("tasks", []) or []
        if not tasks:
            print("[SKIP]", exp, "(no tasks)")
            continue

        model = build_model(build_conf(model_cfg))
        ckpt_path, _ = select_ckpt_path(rd, prefer_step=args.prefer_model_step)
        if ckpt_path is None:
            print("[SKIP]", exp, "(no checkpoint file in run dir)")
            continue
        st = torch.load(ckpt_path, map_location="cpu")
        if isinstance(st, dict) and "model_state_dict" in st:
            st = st["model_state_dict"]
        model.load_state_dict(st, strict=False)
        model.eval()

        n_dims = int(model_cfg["n_dims"])
        batch_size = args.batch_size or int(training_cfg.get("batch_size", 64))
        data_name = training_cfg.get("data", "gaussian")

        for task in tasks:
            task_name = task["name"]
            task_kwargs = task.get("kwargs", {}) or {}
            task_label = _task_to_label(task_name, task_kwargs)

            metrics = eval_model(
                model,
                task_name=task_name,
                data_name=data_name,
                n_dims=n_dims,
                n_points=int(args.max_n_points),
                prompting_strategy="standard",
                num_eval_examples=int(args.num_eval_examples),
                batch_size=int(batch_size),
                task_sampler_kwargs=task_kwargs,
            )

            mean = np.asarray(metrics["mean"], dtype=np.float64)
            low = np.asarray(metrics["bootstrap_low"], dtype=np.float64)
            high = np.asarray(metrics["bootstrap_high"], dtype=np.float64)

            xs = np.arange(1, int(args.max_n_points), dtype=np.int64)
            ys = mean[1 : int(args.max_n_points)]
            ys_low = low[1 : int(args.max_n_points)]
            ys_high = high[1 : int(args.max_n_points)]
            if _is_classification(task_name):
                ys = 1.0 - ys
                ys_low = 1.0 - ys_low
                ys_high = 1.0 - ys_high

            curves.setdefault(task_label, {})[exp] = {
                "icl_len": xs,
                "mean": ys,
                "low": ys_low,
                "high": ys_high,
                "run_dir": str(rd),
                "ckpt": str(ckpt_path),
            }

            for icl_len, m, lo, hi in zip(xs.tolist(), ys.tolist(), ys_low.tolist(), ys_high.tolist()):
                rows.append(
                    {
                        "family": args.family,
                        "exp": exp,
                        "task": task_label,
                        "icl_len": int(icl_len),
                        "error_mean": float(m),
                        "error_bootstrap_low": float(lo),
                        "error_bootstrap_high": float(hi),
                        "run_dir": str(rd),
                        "ckpt": str(ckpt_path),
                        "n_points_eval": int(args.max_n_points),
                        "num_eval_examples": int(args.num_eval_examples),
                        "batch_size": int(batch_size),
                    }
                )

    csv_path = out_dir / "compare_lr2x_summary_ci.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", csv_path)

    for task_label, per_exp in curves.items():
        plt.figure(figsize=(7.5, 4.5))
        for exp in EXPS[args.family]:
            if exp not in per_exp:
                continue
            d = per_exp[exp]
            xs = d["icl_len"]
            ys = d["mean"]
            lo = d["low"]
            hi = d["high"]
            plt.plot(xs, ys, marker="o", linewidth=2, label=exp)
            plt.fill_between(xs, lo, hi, alpha=0.18)
        plt.title(f"ICL curve (error) + bootstrap CI | {args.family} | {task_label}")
        plt.xlabel("ICL length")
        plt.ylabel("error (reg=MSE, cls=1-acc)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        safe = (
            task_label.replace("=", "")
            .replace("(", "_")
            .replace(")", "")
            .replace(",", "_")
            .replace(".", "p")
        )
        out_png = out_dir / f"compare_lr2x_{args.family}_{safe}_ci.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print("Saved:", out_png)


if __name__ == "__main__":
    main()
