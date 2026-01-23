"""
Empirical delta-risk check on existing checkpoints (CPU).

For each experiment:
  - Collect routing distributions per task and per layer.
  - Pick the layer with highest JS divergence.
  - For each task, define delta_t = 1 - max_e P(e | task, layer*).
  - Compute per-task final-token MSE loss.
  - Report correlation between delta_t and loss_t.

Outputs:
  - results/_delta_risk_eval.csv
  - results/_delta_risk_eval_tasks.csv
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from models import build_model
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


def js_div(p, q, eps=1e-12):
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def _spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_exp(run_dir: Path, batches_per_task: int):
    cfg = load_config(run_dir)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    if not (model_cfg.get("use_moe", False) or model_cfg.get("moe_layers")) or model_cfg.get("family") != "EncoderTF":
        return None

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
    tasks = training_cfg.get("tasks", []) or []
    num_tasks = len(tasks)
    num_experts = model_cfg["num_experts"]

    data_sampler = get_data_sampler(training_cfg.get("data", "gaussian"), n_dims=n_dims)

    n_layers = len(model._mlps)
    counts = np.zeros((n_layers, num_tasks, num_experts), dtype=np.int64)
    loss_sums = np.zeros(num_tasks, dtype=np.float64)
    loss_counts = np.zeros(num_tasks, dtype=np.int64)

    def task_label(t):
        kwargs = t.get("kwargs", {}) or {}
        if "noise_std" in kwargs:
            return f"sigma={kwargs['noise_std']}"
        return t.get("name", "task")

    labels = [task_label(t) for t in tasks]

    with torch.no_grad():
        for t_idx, task in enumerate(tasks):
            task_sampler = get_task_sampler(
                task["name"], n_dims, batch_size, **(task.get("kwargs", {}) or {})
            )
            for _ in range(batches_per_task):
                xs = data_sampler.sample_xs(n_positions, batch_size, n_dims)
                task_obj = task_sampler()
                ys = task_obj.evaluate(xs)

                # Forward manually to access routing
                zs = model._combine(xs, ys)
                H = model._read_in(zs)
                for layer_idx, (q, k, v, ln1, mlp, ln2) in enumerate(
                    zip(
                        model._queries,
                        model._keys,
                        model._values,
                        model._lns_1,
                        model._mlps,
                        model._lns_2,
                    )
                ):
                    query = q(H)
                    key = k(H)
                    value = v(H)
                    attn = torch.relu(torch.einsum("bid,bjd->bij", query, key))
                    if model.normalize_attn:
                        attn = attn / ys.shape[1]
                    H = H + torch.einsum("bij,bjd->bid", attn, value)
                    if model.layernorm:
                        H = ln1(H)
                    # routing counts
                    _, _, routing = mlp(H, return_routing_info=True)
                    topk = routing["top_k_indices"].reshape(-1)
                    bc = torch.bincount(topk, minlength=num_experts)
                    counts[layer_idx, t_idx] += bc.cpu().numpy()
                    # forward
                    mlp_out, _, _ = mlp(H)
                    H = H + mlp_out
                    if model.layernorm:
                        H = ln2(H)

                # compute final-token loss
                pred = model._read_out(H)[:, -1, 0]
                loss = (pred - ys[:, -1]).pow(2).mean().item()
                loss_sums[t_idx] += loss
                loss_counts[t_idx] += 1

    # select top layer by JS divergence
    js_scores = []
    for l in range(n_layers):
        js = 0.0
        pairs = 0
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                js += js_div(counts[l, i], counts[l, j])
                pairs += 1
        js_scores.append(js / max(pairs, 1))
    top_layer = int(np.argmax(js_scores))

    # per-task delta
    deltas = []
    for t_idx in range(num_tasks):
        row = counts[top_layer, t_idx]
        row = row / (row.sum() + 1e-12)
        deltas.append(1.0 - row.max())

    loss_t = (loss_sums / (loss_counts + 1e-12)).tolist()

    # correlation
    if num_tasks >= 2:
        corr = float(np.corrcoef(deltas, loss_t)[0, 1])
        # z-score loss within experiment to reduce scale effects
        lt = np.asarray(loss_t, dtype=np.float64)
        lt_z = (lt - lt.mean()) / (lt.std() + 1e-12)
        corr_z = float(np.corrcoef(deltas, lt_z)[0, 1])
        corr_s = _spearman_corr(deltas, lt)
    else:
        corr = float("nan")
        corr_z = float("nan")
        corr_s = float("nan")

    return {
        "exp": run_dir.parent.name,
        "run_dir": str(run_dir),
        "top_layer": top_layer,
        "js_top_layer": js_scores[top_layer],
        "corr_delta_loss": float(corr),
        "corr_delta_loss_z": float(corr_z),
        "corr_delta_loss_spearman": float(corr_s),
        "task_labels": json.dumps(labels),
        "deltas": json.dumps(deltas),
        "losses": json.dumps(loss_t),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--batches-per-task", type=int, default=10)
    parser.add_argument("--all-moe", action="store_true")
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--max-exps", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if args.all_moe:
        overview = results_dir / "_exp_overview.json"
        if overview.exists():
            rows = json.loads(overview.read_text(encoding="utf-8"))
            exps = []
            for r in rows:
                if not (r.get("use_moe") or r.get("moe_layers")):
                    continue
                tasks = json.loads(r.get("tasks", "[]"))
                if len(tasks) < args.min_tasks:
                    continue
                exps.append(r["exp"])
        else:
            exps = []
            for exp_dir in results_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                rd = latest_run(results_dir, exp_dir.name)
                if rd is None:
                    continue
                cfg = load_config(rd)
                model_cfg = cfg.get("model", {})
                if not (model_cfg.get("use_moe", False) or model_cfg.get("moe_layers")):
                    continue
                tasks = cfg.get("training", {}).get("tasks", []) or []
                if len(tasks) < args.min_tasks:
                    continue
                exps.append(exp_dir.name)
        exps = sorted(set(exps))
    else:
        exps = [
            "E53_moe8_aux0",
            "E59_moe4_seqaux0",
            "E60_moe16_seqaux0",
            "E07_moe4_seq_routing",
            "E09_moe4_no_aux",
            "E12_moe8_8noise",
            "E54_moe8_aux10",
        ]
    if args.offset and args.offset > 0:
        exps = exps[args.offset :]
    if args.max_exps and args.max_exps > 0:
        exps = exps[: args.max_exps]

    rows = []
    task_rows = []
    for exp in exps:
        rd = latest_run(results_dir, exp)
        if rd is None:
            print(f"[SKIP] {exp}")
            continue
        out = analyze_exp(rd, args.batches_per_task)
        if out is None:
            print(f"[SKIP] {exp} (not MoE EncoderTF)")
            continue
        rows.append(out)
        # expand task rows
        labels = json.loads(out["task_labels"])
        deltas = json.loads(out["deltas"])
        losses = json.loads(out["losses"])
        # precompute z-score for per-task table
        lt = np.asarray(losses, dtype=np.float64)
        lt_z = (lt - lt.mean()) / (lt.std() + 1e-12)
        for l, d, loss, z in zip(labels, deltas, losses, lt_z.tolist()):
            task_rows.append(
                {
                    "exp": out["exp"],
                    "task": l,
                    "delta": d,
                    "loss": loss,
                    "loss_z": z,
                    "top_layer": out["top_layer"],
                }
            )
        print(f"[OK] {exp}: corr={out['corr_delta_loss']:.3f}")

    # save
    import csv

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out = results_dir / f"_delta_risk_eval{suffix}.csv"
    out_tasks = results_dir / f"_delta_risk_eval_tasks{suffix}.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if task_rows:
        with out_tasks.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            writer.writeheader()
            writer.writerows(task_rows)
    print("Saved delta-risk outputs to", out, out_tasks)


if __name__ == "__main__":
    main()
