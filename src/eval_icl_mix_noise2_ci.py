"""
Plot ICL curves with bootstrap confidence intervals for mix_noise2 experiments.

This script is a CI-enabled companion to eval_icl_mix_noise2.py.

Key idea (decoder-only families):
  - Evaluate once at max context (n_points=21), which yields per-position metrics.
  - Use position i (1..20) as the "last-point error" for ICL length i.
  - Use aggregate_metrics() outputs from src/eval.py:
      mean, bootstrap_low, bootstrap_high
    to draw a shaded confidence band for the mean error.

Outputs:
  - results/_icl_curves/compare_mix_noise2_summary_ci.csv
  - results/_icl_curves/compare_mix_noise2_<family>_noisy_linear_regression_noise_std0.1_ci.png
  - results/_icl_curves/compare_mix_noise2_<family>_noisy_linear_regression_noise_std0.5_ci.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from eval import eval_model
from models import build_model
from ckpt_utils import latest_run_dir, select_ckpt_path


EXPS = {
    "gpt2": [
        "E81_gpt2_dense_mix_noise2",
        "E82_gpt2_moe4_mix_noise2",
        "E83_gpt2_moe4_last3_mix_noise2",
    ],
    "llama_hf": [
        "E87_llama_dense_mix_noise2",
        "E88_llama_moe4_mix_noise2",
        "E89_llama_moe4_last3_mix_noise2",
    ],
    "qwen_hf": [
        "E93_qwen_dense_mix_noise2",
        "E94_qwen_moe4_mix_noise2",
        "E95_qwen_moe4_last3_mix_noise2",
    ],
    "gemma_hf": [
        "E99_gemma_dense_mix_noise2",
        "E100_gemma_moe4_mix_noise2",
        "E101_gemma_moe4_last3_mix_noise2",
    ],
}


def build_conf(model_cfg):
    class C:
        pass

    c = C()
    for k, v in model_cfg.items():
        setattr(c, k, v)
    c.keys = lambda: c.__dict__.keys()
    return c


def _task_to_label(task_name: str, task_kwargs: dict) -> str:
    noise_std = task_kwargs.get("noise_std", None)
    if noise_std is None:
        return task_name
    return f"{task_name}(noise_std={noise_std})"


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
        default=21,
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
    curves = {}  # (task_label) -> exp -> dict(mean/low/high)

    for exp in exp_list:
        exp_dir = results_dir / exp
        rd = latest_run_dir(exp_dir, prefer_step=args.prefer_model_step)
        if rd is None:
            print("skip", exp, "no checkpoint")
            continue

        cfg = yaml.safe_load((rd / "config.yaml").open("r"))
        model_cfg = cfg["model"]
        training_cfg = cfg["training"]
        tasks = training_cfg.get("tasks", [])
        if not tasks:
            print("skip", exp, "no tasks")
            continue

        model = build_model(build_conf(model_cfg))
        ckpt_path, _ = select_ckpt_path(rd, prefer_step=args.prefer_model_step)
        if ckpt_path is None:
            print("skip", exp, "no checkpoint file in", rd)
            continue
        st = torch.load(ckpt_path, map_location="cpu")
        if isinstance(st, dict) and "model_state_dict" in st:
            st = st["model_state_dict"]
        model.load_state_dict(st, strict=False)
        model.eval()

        n_dims = model_cfg["n_dims"]
        batch_size = args.batch_size or training_cfg.get("batch_size", 64)
        data_name = training_cfg.get("data", "gaussian")

        for task in tasks:
            task_name = task["name"]
            task_kwargs = task.get("kwargs", {}) or {}
            task_label = _task_to_label(task_name, task_kwargs)

            # Evaluate once at max context; use per-position outputs.
            metrics = eval_model(
                model,
                task_name=task_name,
                data_name=data_name,
                n_dims=n_dims,
                n_points=args.max_n_points,
                prompting_strategy="standard",
                num_eval_examples=args.num_eval_examples,
                batch_size=batch_size,
                task_sampler_kwargs=task_kwargs,
            )

            mean = np.asarray(metrics["mean"], dtype=np.float64)
            low = np.asarray(metrics["bootstrap_low"], dtype=np.float64)
            high = np.asarray(metrics["bootstrap_high"], dtype=np.float64)

            # ICL length i corresponds to position i (1..max_n_points-1).
            xs = np.arange(1, args.max_n_points, dtype=np.int64)
            ys = mean[1: args.max_n_points]
            ys_low = low[1: args.max_n_points]
            ys_high = high[1: args.max_n_points]

            curves.setdefault(task_label, {})[exp] = {
                "icl_len": xs,
                "mean": ys,
                "low": ys_low,
                "high": ys_high,
                "run_dir": str(rd),
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
                        "n_points_eval": int(args.max_n_points),
                        "num_eval_examples": int(args.num_eval_examples),
                    }
                )

    # Save CSV
    csv_path = out_dir / "compare_mix_noise2_summary_ci.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "family",
                    "exp",
                    "task",
                    "icl_len",
                    "error_mean",
                    "error_bootstrap_low",
                    "error_bootstrap_high",
                    "run_dir",
                    "n_points_eval",
                    "num_eval_examples",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", csv_path)

    # Plot per noise level (task label)
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
        plt.ylabel("error (MSE)")
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
        out = out_dir / f"compare_mix_noise2_{args.family}_{safe}_ci.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print("Saved:", out)


if __name__ == "__main__":
    main()
