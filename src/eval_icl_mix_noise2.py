"""
Evaluate ICL curves for mix_noise2 experiments (noise=0.1/0.5).

Outputs:
  - results/_icl_curves/compare_mix_noise2_summary.csv
  - results/_icl_curves/compare_mix_noise2_<family>_noise0p1.png
  - results/_icl_curves/compare_mix_noise2_<family>_noise0p5.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

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
    "s_mixnoise4_80x40_series": [
        "S25_gpt2_w32_d6_mixnoise4_80x40",
        "S26_gpt2_w64_d6_mixnoise4_80x40",
        "S27_gpt2_w128_d6_mixnoise4_80x40",
        "S28_gpt2_w256_d6_mixnoise4_80x40",
        "S29_gpt2_w64_d2_mixnoise4_80x40",
        "S30_gpt2_w64_d4_mixnoise4_80x40",
        "S31_gpt2_w64_d8_mixnoise4_80x40",
        "S32_gpt2_w64_d12_mixnoise4_80x40",
        "S33_gpt2_tiny_mixnoise4_80x40",
        "S34_gpt2_small_mixnoise4_80x40",
        "S35_gpt2_medium_mixnoise4_80x40",
        "S36_gpt2_large_mixnoise4_80x40",
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


def parse_n_points(n_points_str):
    # support "2-21" or "2,4,8,12"
    pts = []
    for part in n_points_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo = int(lo); hi = int(hi)
            pts.extend(list(range(lo, hi + 1)))
        else:
            pts.append(int(part))
    return sorted(set(pts))


def load_existing_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--family", choices=list(EXPS.keys()), required=True)
    parser.add_argument("--exp", default="", help="Optional single experiment name")
    # ICL length 1..20 => n_points 2..21
    parser.add_argument("--n-points", default="2-21")
    parser.add_argument(
        "--prefer-model-step",
        type=int,
        default=300000,
        help="Prefer loading model_{step}.pt if present; otherwise fall back to an earlier model_{k}.pt or state.pt.",
    )
    parser.add_argument("--num-eval-examples", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "_icl_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_points_list = parse_n_points(args.n_points)
    csv_path = out_dir / "compare_mix_noise2_summary.csv"

    rows = []

    exp_list = EXPS[args.family]
    if args.exp:
        if args.exp not in exp_list:
            raise SystemExit(f"--exp must be one of: {exp_list}")
        exp_list = [args.exp]

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

        # Fast path: evaluate once at max_n_points, then slice the per-position mean curve.
        max_n_points = max(n_points_list) if n_points_list else model_cfg.get("n_positions", 21)

        for task in tasks:
            task_name = task["name"]
            task_kwargs = task.get("kwargs", {}) or {}
            noise_std = task_kwargs.get("noise_std", None)
            task_label = f"{task_name}(noise_std={noise_std})"

            metrics = eval_model(
                model,
                task_name=task_name,
                data_name=data_name,
                n_dims=n_dims,
                n_points=max_n_points,
                prompting_strategy="standard",
                num_eval_examples=args.num_eval_examples,
                batch_size=batch_size,
                task_sampler_kwargs=task_kwargs,
            )
            mean_curve = metrics["mean"]

            for n_points in n_points_list:
                icl_len = int(n_points) - 1
                if icl_len < 0 or icl_len >= len(mean_curve):
                    continue
                err = mean_curve[icl_len]
                rows.append({
                    "family": args.family,
                    "exp": exp,
                    "task": task_label,
                    "icl_len": icl_len,
                    "error": float(err),
                    "run_dir": str(rd),
                })

    # overwrite CSV with fresh results (keep other families if present)
    existing = load_existing_rows(csv_path)
    kept = [r for r in existing if r.get("family") != args.family]
    merged = kept + rows
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["family", "exp", "task", "icl_len", "error", "run_dir"]
        )
        writer.writeheader()
        writer.writerows(merged)

    # plot per noise level for this family
    data = [r for r in rows if r["family"] == args.family]
    if not data:
        return

    # group by task label
    tasks = sorted(set(r["task"] for r in data))
    for task_label in tasks:
        plt.figure(figsize=(7.5, 4.5))
        for exp in EXPS[args.family]:
            sub = [r for r in data if r["task"] == task_label and r["exp"] == exp]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: int(r["icl_len"]))
            xs = [int(r["icl_len"]) for r in sub]
            ys = [float(r["error"]) for r in sub]
            plt.plot(xs, ys, marker="o", label=exp)
        plt.title(f"ICL curve (error) | {args.family} | {task_label}")
        plt.xlabel("ICL length")
        plt.ylabel("error (MSE)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        safe = task_label.replace("=", "").replace("(", "_").replace(")", "").replace(",", "_")
        out = out_dir / f"compare_mix_noise2_{args.family}_{safe}.png"
        plt.savefig(out, dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
