"""
Evaluate ICL curves (error vs. ICL length) for each experiment in results/.

For each experiment with a state.pt + config.yaml, we:
  - load the model checkpoint
  - evaluate each training task at multiple n_points
  - record last-point error (ICL length = n_points - 1)
  - write CSV + JSON and save per-experiment plots
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from eval import eval_model
from models import build_model


def _task_label(task):
    name = task.get("name", "unknown")
    kwargs = task.get("kwargs", {}) or {}
    if kwargs:
        items = ",".join([f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())])
        return f"{name}({items})"
    return name


def _is_classification(task_name: str):
    # Only linear_classification uses accuracy in this repo
    return "classification" in task_name


def _latest_run_dir(exp_dir: Path):
    run_dirs = []
    for cfg in exp_dir.rglob("config.yaml"):
        rd = cfg.parent
        if (rd / "state.pt").exists():
            run_dirs.append(rd)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: (p / "state.pt").stat().st_mtime)


def _build_conf(model_cfg: dict):
    # SimpleNamespace with keys() for EncoderTF defaults
    class _C:
        pass
    conf = _C()
    for k, v in model_cfg.items():
        setattr(conf, k, v)
    conf.keys = lambda: conf.__dict__.keys()
    return conf


def parse_n_points(n_points_str, max_points):
    if n_points_str is None or n_points_str.strip() == "":
        # default: even lengths + max
        pts = list(range(2, max_points + 1, 2))
        if max_points not in pts:
            pts.append(max_points)
        return pts
    pts = []
    for part in n_points_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo = int(lo); hi = int(hi)
            pts.extend(list(range(lo, hi + 1)))
        else:
            pts.append(int(part))
    # unique + sorted + clamp
    pts = sorted(set([p for p in pts if p >= 2]))
    return [p for p in pts if p <= max_points]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--exps", default="", help="comma list, e.g. E70,E71; empty=all")
    parser.add_argument("--n-points", default="", help="e.g. 2-21 or 2,4,6,8")
    parser.add_argument("--num-eval-examples", type=int, default=6400)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-dir", default="../results/_icl_curves")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.exps:
        exps = [e.strip() for e in args.exps.split(",") if e.strip()]
    else:
        exps = sorted([p.name for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("E")])

    rows = []
    details = []

    for exp in exps:
        exp_dir = results_dir / exp
        rd = _latest_run_dir(exp_dir)
        if rd is None:
            rows.append({"exp": exp, "status": "no_state"})
            continue

        cfg = yaml.safe_load((rd / "config.yaml").open("r"))
        model_cfg = cfg.get("model", {})
        training_cfg = cfg.get("training", {})
        tasks = training_cfg.get("tasks", [])
        if not tasks:
            rows.append({"exp": exp, "status": "no_tasks"})
            continue

        n_dims = model_cfg.get("n_dims")
        n_positions = model_cfg.get("n_positions")
        batch_size = training_cfg.get("batch_size", 64)
        data_name = training_cfg.get("data", "gaussian")

        n_points_list = parse_n_points(args.n_points, n_positions)
        if not n_points_list:
            rows.append({"exp": exp, "status": "no_n_points"})
            continue

        # build model
        model = build_model(_build_conf(model_cfg))
        st = torch.load(rd / "state.pt", map_location="cpu")
        if isinstance(st, dict) and "model_state_dict" in st:
            st = st["model_state_dict"]
        model.load_state_dict(st, strict=False)
        model.eval()
        if args.device == "cuda" and torch.cuda.is_available():
            model.cuda()

        # evaluate
        exp_detail = {
            "exp": exp,
            "tasks": [_task_label(t) for t in tasks],
            "n_points_list": n_points_list,
            "curves": {},
        }

        for task in tasks:
            task_name = task.get("name")
            task_kwargs = task.get("kwargs", {}) or {}
            label = _task_label(task)

            curve = []
            for n_points in n_points_list:
                with torch.no_grad():
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
                # last-point performance
                last_val = metrics["mean"][-1]
                if _is_classification(task_name):
                    err = 1.0 - last_val
                else:
                    err = last_val
                curve.append({"n_points": n_points, "icl_len": n_points - 1, "error": err, "metric_last": last_val})

                rows.append({
                    "exp": exp,
                    "task": label,
                    "task_name": task_name,
                    "n_points": n_points,
                    "icl_len": n_points - 1,
                    "metric_last": last_val,
                    "error": err,
                    "num_eval_examples": args.num_eval_examples,
                })

            exp_detail["curves"][label] = curve

        # plot per experiment
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for task_label, curve in exp_detail["curves"].items():
            xs = [c["icl_len"] for c in curve]
            ys = [c["error"] for c in curve]
            ax.plot(xs, ys, marker="o", label=task_label)
        ax.set_title(f"{exp} ICL curve (last-point error)")
        ax.set_xlabel("ICL length (n_points - 1)")
        ax.set_ylabel("error (reg=MSE, cls=1-acc)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out_png = save_dir / f"{exp}_icl_curve.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

        details.append(exp_detail)

    # save outputs
    csv_path = save_dir / "_icl_curve_summary.csv"
    json_path = save_dir / "_icl_curve_details.json"
    import csv
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Plots in: {save_dir}")


if __name__ == "__main__":
    main()
