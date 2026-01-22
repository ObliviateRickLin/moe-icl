"""
Evaluate scaling vs num_experts using existing checkpoints (CPU).

Outputs:
  - results/_scaling_eval.csv
"""

import argparse
import json
from pathlib import Path

import yaml
import torch
import numpy as np

from models import build_model
from types import SimpleNamespace
from tasks import get_task_sampler
from samplers import get_data_sampler


def load_config(run_dir: Path):
    with (run_dir / "config.yaml").open("r", encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f)


def latest_run(results_dir: Path, exp_name: str):
    exp_dir = results_dir / exp_name
    if not exp_dir.exists():
        return None
    runs = []
    for cfg in exp_dir.rglob("config.yaml"):
        state = cfg.parent / "state.pt"
        if state.exists():
            runs.append((state.stat().st_mtime, cfg.parent))
    if not runs:
        return None
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1]


def eval_run(run_dir: Path, num_eval_examples: int):
    cfg = load_config(run_dir)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    tasks = training_cfg.get("tasks", []) or []

    # build model
    conf = SimpleNamespace(**model_cfg)
    conf.keys = lambda: conf.__dict__.keys()
    model = build_model(conf)
    state = torch.load(run_dir / "state.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    n_dims = model_cfg["n_dims"]
    n_positions = model_cfg["n_positions"]
    batch_size = training_cfg.get("batch_size", 64)
    num_batches = max(1, num_eval_examples // batch_size)

    data_sampler = get_data_sampler(training_cfg.get("data", "gaussian"), n_dims=n_dims)

    task_results = []
    with torch.no_grad():
        for t in tasks:
            task_name = t["name"]
            kwargs = t.get("kwargs", {}) or {}
            task_sampler = get_task_sampler(task_name, n_dims, batch_size, **kwargs)
            loss_sum = 0.0
            for _ in range(num_batches):
                xs = data_sampler.sample_xs(n_positions, batch_size, n_dims)
                task_obj = task_sampler()
                ys = task_obj.evaluate(xs)
                pred = model(xs, ys)
                loss = ((pred[:, -1] - ys[:, -1]) ** 2).mean().item()
                loss_sum += loss
            final_loss = loss_sum / num_batches
            task_results.append(
                {
                    "task": task_name,
                    "kwargs": kwargs,
                    "final_loss": float(final_loss),
                }
            )

    avg_loss = float(np.mean([r["final_loss"] for r in task_results])) if task_results else 0.0
    return avg_loss, task_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--num-eval-examples", type=int, default=512)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()

    # Expert scaling set (noisy linear regression, 4-noise)
    scaling_exps = [
        "E03_moe2_4noise",
        "E04_moe4_4noise",
        "E05_moe8_4noise",
        "E06_moe16_4noise",
        "E64_moe32",
        "E43_moe64_10k",
        "E46_moe64_100k",
        "E49_moe64_1m",
    ]

    rows = []
    for exp in scaling_exps:
        rd = latest_run(results_dir, exp)
        if rd is None:
            print(f"[SKIP] {exp} (missing)")
            continue
        cfg = load_config(rd)
        model_cfg = cfg["model"]
        avg_loss, task_results = eval_run(rd, args.num_eval_examples)
        rows.append(
            {
                "exp": exp,
                "run_dir": str(rd),
                "num_experts": model_cfg.get("num_experts", None),
                "top_k": model_cfg.get("top_k", None),
                "seq_level_routing": model_cfg.get("seq_level_routing", None),
                "aux_loss_coef": model_cfg.get("aux_loss_coef", None),
                "router_noise": model_cfg.get("router_noise", None),
                "avg_final_loss": avg_loss,
                "task_results": json.dumps(task_results),
            }
        )
        print(f"[OK] {exp}: avg_final_loss={avg_loss:.4f}")

    # save
    import csv

    out = results_dir / "_scaling_eval.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print("Saved", out)


if __name__ == "__main__":
    main()
