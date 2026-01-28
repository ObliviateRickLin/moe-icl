"""
Conformal prediction diagnostics for the lr_2x curriculum runs (E141-152).

This script produces two conformal curves vs. ICL length:
  1) Coverage(L): empirical coverage of a symmetric interval around y_hat
  2) HalfWidth(L): conformal half-width q_L (so interval is [y_hat - q_L, y_hat + q_L])

We use split conformal for each ICL length L independently:
  - Evaluate once at max_n_points (default 41) to get per-position predictions.
  - For each L=1..(max_n_points-1):
      scores = |y - y_hat| at position L over calibration examples
      q_L = k-th order statistic with k = ceil((n_cal + 1) * (1 - alpha))
      coverage(L) measured on the held-out test examples

Outputs (under results/_icl_curves/):
  - compare_lr2x_conformal_<family>_<task>_alpha0p05.png
  - compare_lr2x_conformal_summary_<family>_alpha0p05.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ckpt_utils import latest_run_dir, select_ckpt_path
from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler


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
    if not task_kwargs:
        return task_name
    items = ",".join([f"{k}={task_kwargs[k]}" for k in sorted(task_kwargs.keys())])
    return f"{task_name}({items})"


def _conformal_q(scores: torch.Tensor, alpha: float) -> float:
    """
    Split conformal quantile with finite-sample correction.
    scores: 1D tensor of nonconformity scores on calibration set.
    Returns q such that interval [yhat-q, yhat+q] targets marginal coverage 1-alpha.
    """
    scores = scores.detach().flatten().to(torch.float64)
    n = int(scores.numel())
    if n <= 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    vals, _ = torch.sort(scores)
    return float(vals[k - 1].item())


@torch.no_grad()
def eval_abs_error_matrix(
    model,
    task_name: str,
    task_kwargs: dict,
    data_name: str,
    n_dims: int,
    n_points: int,
    num_eval_examples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Returns abs errors with shape (num_eval_examples, n_points) on the given device's output,
    but stored on CPU (float32).
    """
    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims=n_dims)
    task_sampler = get_task_sampler(task_name, n_dims, batch_size, **task_kwargs)

    model = model.to(device).eval()

    out = []
    for _ in range(num_eval_examples // batch_size):
        xs = data_sampler.sample_xs(n_points, batch_size, n_dims, device=device)
        task = task_sampler(device=device)
        ys = task.evaluate(xs)
        pred = model(xs, ys)
        abs_err = (ys - pred).abs()
        out.append(abs_err.detach().cpu().to(torch.float32))
    return torch.cat(out, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--family", choices=list(EXPS.keys()), required=True)
    parser.add_argument("--exp", default="", help="Optional single experiment name")
    parser.add_argument("--prefer-model-step", type=int, default=300000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-examples", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-n-points", type=int, default=41)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if not (0.0 < args.alpha < 1.0):
        raise SystemExit("--alpha must be in (0,1)")
    if not (0.0 < args.calib_frac < 1.0):
        raise SystemExit("--calib-frac must be in (0,1)")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device=cuda requested but CUDA not available; falling back to cpu.")
        device = "cpu"

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "_icl_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_list = EXPS[args.family]
    if args.exp:
        if args.exp not in exp_list:
            raise SystemExit(f"--exp must be one of: {exp_list}")
        exp_list = [args.exp]

    rows: list[dict] = []
    plots: dict[str, dict[str, dict[str, np.ndarray]]] = {}  # task -> exp -> series dict

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
        data_name = training_cfg.get("data", "gaussian")
        batch_size = args.batch_size or int(training_cfg.get("batch_size", 64))

        for task in tasks:
            task_name = task["name"]
            task_kwargs = task.get("kwargs", {}) or {}
            task_label = _task_to_label(task_name, task_kwargs)

            abs_err = eval_abs_error_matrix(
                model=model,
                task_name=task_name,
                task_kwargs=task_kwargs,
                data_name=data_name,
                n_dims=n_dims,
                n_points=int(args.max_n_points),
                num_eval_examples=int(args.num_eval_examples),
                batch_size=int(batch_size),
                device=device,
            )

            # Split calibration/test along examples.
            g = torch.Generator()
            g.manual_seed(int(args.seed))
            n_total = int(abs_err.shape[0])
            perm = torch.randperm(n_total, generator=g)
            n_cal = int(round(float(args.calib_frac) * n_total))
            n_cal = max(1, min(n_cal, n_total - 1))
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal:]

            abs_cal = abs_err[cal_idx]
            abs_test = abs_err[test_idx]

            icl_lens = np.arange(1, int(args.max_n_points), dtype=np.int64)
            cov = np.zeros_like(icl_lens, dtype=np.float64)
            q = np.zeros_like(icl_lens, dtype=np.float64)

            for j, L in enumerate(icl_lens.tolist()):
                q_L = _conformal_q(abs_cal[:, L], alpha=float(args.alpha))
                q[j] = q_L
                cov[j] = float((abs_test[:, L] <= q_L).to(torch.float32).mean().item())

                rows.append(
                    {
                        "family": args.family,
                        "exp": exp,
                        "task": task_label,
                        "icl_len": int(L),
                        "alpha": float(args.alpha),
                        "q_half_width": float(q_L),
                        "interval_width": float(2.0 * q_L),
                        "coverage": float(cov[j]),
                        "n_total": int(n_total),
                        "n_cal": int(n_cal),
                        "n_test": int(n_total - n_cal),
                        "run_dir": str(rd),
                        "ckpt": str(ckpt_path),
                        "n_points_eval": int(args.max_n_points),
                        "num_eval_examples": int(args.num_eval_examples),
                        "batch_size": int(batch_size),
                        "seed": int(args.seed),
                        "calib_frac": float(args.calib_frac),
                        "device": device,
                    }
                )

            plots.setdefault(task_label, {})[exp] = {
                "icl_len": icl_lens,
                "coverage": cov,
                "q": q,
            }

    alpha_tag = str(args.alpha).replace(".", "p")
    csv_path = out_dir / f"compare_lr2x_conformal_summary_{args.family}_alpha{alpha_tag}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", csv_path)

    target = 1.0 - float(args.alpha)
    for task_label, per_exp in plots.items():
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)

        ax = axes[0]
        for exp in EXPS[args.family]:
            if exp not in per_exp:
                continue
            d = per_exp[exp]
            ax.plot(d["icl_len"], d["coverage"], marker="o", linewidth=2, label=exp)
        ax.axhline(target, color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"target {target:.2f}")
        ax.set_ylabel("coverage")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax2 = axes[1]
        for exp in EXPS[args.family]:
            if exp not in per_exp:
                continue
            d = per_exp[exp]
            ax2.plot(d["icl_len"], d["q"], marker="o", linewidth=2, label=exp)
        ax2.set_xlabel("ICL length")
        ax2.set_ylabel("half-width q (|y - y_hat| quantile)")
        ax2.grid(alpha=0.3)

        fig.suptitle(f"Conformal (split) | {args.family} | {task_label} | alpha={args.alpha}")
        fig.tight_layout()

        safe = (
            task_label.replace("=", "")
            .replace("(", "_")
            .replace(")", "")
            .replace(",", "_")
            .replace(".", "p")
        )
        out_png = out_dir / f"compare_lr2x_conformal_{args.family}_{safe}_alpha{alpha_tag}.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print("Saved:", out_png)


if __name__ == "__main__":
    main()

