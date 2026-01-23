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


def latest_run_dir_with_ckpt(exp_dir: Path, prefer_step: int):
    """
    Prefer a fixed-step checkpoint (model_{prefer_step}.pt) if present.
    Fall back to state.pt otherwise.
    """
    candidates = []
    for cfg in exp_dir.rglob("config.yaml"):
        rd = cfg.parent
        model_path = rd / f"model_{prefer_step}.pt"
        state_path = rd / "state.pt"

        if model_path.exists():
            candidates.append((1, model_path.stat().st_mtime, rd))
        elif state_path.exists():
            candidates.append((0, state_path.stat().st_mtime, rd))

    if not candidates:
        return None
    # Prefer runs that have the fixed-step checkpoint; within that, take the latest.
    return max(candidates, key=lambda t: (t[0], t[1]))[2]


def select_ckpt_path(run_dir: Path, prefer_step: int) -> Path:
    ckpt = run_dir / f"model_{prefer_step}.pt"
    if ckpt.exists():
        return ckpt
    return run_dir / "state.pt"


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
        default=10000,
        help="Prefer loading model_{step}.pt if present; otherwise fall back to state.pt.",
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
        rd = latest_run_dir_with_ckpt(exp_dir, args.prefer_model_step)
        if rd is None:
            print("skip", exp, "no state")
            continue

        cfg = yaml.safe_load((rd / "config.yaml").open("r"))
        model_cfg = cfg["model"]
        training_cfg = cfg["training"]
        tasks = training_cfg.get("tasks", [])
        if not tasks:
            print("skip", exp, "no tasks")
            continue

        model = build_model(build_conf(model_cfg))
        ckpt_path = select_ckpt_path(rd, args.prefer_model_step)
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
            noise_std = task_kwargs.get("noise_std", None)
            task_label = f"{task_name}(noise_std={noise_std})"

            for n_points in n_points_list:
                icl_len = n_points - 1
                metrics = eval_model(
                    model,
                    task_name=task_name,
                    data_name=data_name,
                    n_dims=n_dims,
                    n_points=n_points,
                    prompting_strategy="standard",
                    num_eval_examples=args.num_eval_examples,
                    batch_size=batch_size,
                    task_sampler_kwargs=task_kwargs,
                )
                last_val = metrics["mean"][-1]
                err = last_val
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
