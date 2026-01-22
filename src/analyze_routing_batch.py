"""
Batch routing/specialization analysis across many experiments.

Outputs:
  - results/_routing_summary.csv: one row per experiment
  - results/_routing_layers.csv: per-layer metrics
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

# Local imports
from models import build_model
from tasks import get_task_sampler
from samplers import get_data_sampler


def _safe_load_yaml(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f)


def _build_conf(model_cfg: dict):
    conf = SimpleNamespace(**model_cfg)
    conf.keys = lambda: conf.__dict__.keys()
    return conf


def _js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def _entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + eps)
    return -np.sum(p * np.log(p + eps))


def _mutual_info(counts_te, eps=1e-12):
    # counts_te: (tasks, experts)
    counts_te = np.asarray(counts_te, dtype=np.float64)
    total = counts_te.sum()
    if total <= 0:
        return 0.0
    p_te = counts_te / total
    p_t = p_te.sum(axis=1, keepdims=True)
    p_e = p_te.sum(axis=0, keepdims=True)
    denom = p_t @ p_e
    mi = np.sum(p_te * np.log((p_te + eps) / (denom + eps)))
    return float(mi)


def _task_label(task):
    name = task.get("name", "unknown")
    kwargs = task.get("kwargs", {}) or {}
    # Stable string with sorted keys
    items = ",".join([f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())])
    return f"{name}({items})"


def _collect_counts(
    model,
    tasks,
    data_sampler,
    n_dims,
    n_positions,
    batch_size,
    batches_per_task,
    device,
):
    n_layer = len(model._mlps)
    num_experts = model._mlps[0].num_experts
    counts = np.zeros((n_layer, len(tasks), num_experts), dtype=np.int64)
    totals = np.zeros((n_layer, len(tasks)), dtype=np.int64)

    for t_idx, task in enumerate(tasks):
        task_sampler = get_task_sampler(
            task["name"],
            n_dims,
            batch_size,
            **(task.get("kwargs", {}) or {}),
        )
        for _ in range(batches_per_task):
            xs = data_sampler.sample_xs(
                n_positions, batch_size, n_dims, device=device
            )
            task_obj = task_sampler(device=device)
            ys = task_obj.evaluate(xs)

            with torch.no_grad():
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

                    # MoE routing
                    mlp_out, _, routing = mlp(H, return_routing_info=True)
                    top_k = routing["top_k_indices"]  # (B, S, k)
                    flat = top_k.reshape(-1)
                    bc = torch.bincount(flat, minlength=num_experts)
                    counts[layer_idx, t_idx] += bc.detach().cpu().numpy()
                    totals[layer_idx, t_idx] += int(flat.numel())

                    H = H + mlp_out
                    if model.layernorm:
                        H = ln2(H)

    return counts, totals


def _weight_cosine_stats(model):
    stats = []
    for layer_idx, mlp in enumerate(model._mlps):
        num_experts = mlp.num_experts
        # Flatten expert weights
        w1 = mlp.w1.detach().cpu().reshape(num_experts, -1)
        w2 = mlp.w2.detach().cpu().reshape(num_experts, -1)
        vec = torch.cat([w1, w2], dim=1).numpy()
        # Normalize
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        vec_n = vec / norms
        cos = vec_n @ vec_n.T
        # Off-diagonal stats
        mask = ~np.eye(num_experts, dtype=bool)
        off = cos[mask]
        stats.append(
            {
                "layer": layer_idx,
                "weight_cos_mean": float(off.mean()) if off.size else 0.0,
                "weight_cos_min": float(off.min()) if off.size else 0.0,
                "weight_cos_max": float(off.max()) if off.size else 0.0,
            }
        )
    return stats


def analyze_experiment(run_dir: Path, batches_per_task: int, device: str):
    config_path = run_dir / "config.yaml"
    state_path = run_dir / "state.pt"
    cfg = _safe_load_yaml(config_path)

    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    if not model_cfg.get("use_moe", False):
        return None
    if model_cfg.get("family") != "EncoderTF":
        return None

    conf = _build_conf(model_cfg)
    model = build_model(conf)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()

    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    tasks = training_cfg.get("tasks", [])
    if not tasks:
        return None

    n_dims = model_cfg["n_dims"]
    n_positions = model_cfg["n_positions"]
    batch_size = training_cfg.get("batch_size", 64)
    data_sampler = get_data_sampler(training_cfg.get("data", "gaussian"), n_dims=n_dims)

    counts, totals = _collect_counts(
        model,
        tasks,
        data_sampler,
        n_dims,
        n_positions,
        batch_size,
        batches_per_task,
        device,
    )
    weight_stats = _weight_cosine_stats(model)

    num_experts = model_cfg["num_experts"]
    task_labels = [_task_label(t) for t in tasks]

    layer_rows = []
    entropy_vals = []
    js_vals = []
    mi_vals = []

    for l in range(counts.shape[0]):
        counts_l = counts[l]
        total_l = counts_l.sum()
        if total_l == 0:
            continue
        p_e = counts_l.sum(axis=0) / total_l
        ent = _entropy(p_e) / (np.log(num_experts) + 1e-12)
        entropy_vals.append(ent)

        # Task-conditioned specialization
        js = 0.0
        if len(tasks) > 1:
            pairs = 0
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    js += _js_divergence(counts_l[i], counts_l[j])
                    pairs += 1
            js = js / max(pairs, 1)
        js_vals.append(js)

        mi = _mutual_info(counts_l)
        mi_vals.append(mi)

        ws = weight_stats[l] if l < len(weight_stats) else {}
        layer_rows.append(
            {
                "exp": run_dir.parent.name,
                "run_dir": str(run_dir),
                "layer": l,
                "num_experts": num_experts,
                "top_k": model_cfg.get("top_k", 1),
                "seq_level_routing": model_cfg.get("seq_level_routing", False),
                "aux_loss_coef": model_cfg.get("aux_loss_coef", 0.0),
                "router_noise": model_cfg.get("router_noise", False),
                "entropy_norm": ent,
                "js_div": js,
                "mi": mi,
                "usage_frac": json.dumps(p_e.tolist()),
                "task_labels": json.dumps(task_labels),
                "weight_cos_mean": ws.get("weight_cos_mean", 0.0),
                "weight_cos_min": ws.get("weight_cos_min", 0.0),
                "weight_cos_max": ws.get("weight_cos_max", 0.0),
            }
        )

    summary = {
        "exp": run_dir.parent.name,
        "run_dir": str(run_dir),
        "num_experts": num_experts,
        "top_k": model_cfg.get("top_k", 1),
        "seq_level_routing": model_cfg.get("seq_level_routing", False),
        "aux_loss_coef": model_cfg.get("aux_loss_coef", 0.0),
        "router_noise": model_cfg.get("router_noise", False),
        "tasks": len(tasks),
        "batches_per_task": batches_per_task,
        "entropy_mean": float(np.mean(entropy_vals)) if entropy_vals else 0.0,
        "entropy_min": float(np.min(entropy_vals)) if entropy_vals else 0.0,
        "entropy_max": float(np.max(entropy_vals)) if entropy_vals else 0.0,
        "js_mean": float(np.mean(js_vals)) if js_vals else 0.0,
        "js_max": float(np.max(js_vals)) if js_vals else 0.0,
        "mi_mean": float(np.mean(mi_vals)) if mi_vals else 0.0,
        "mi_max": float(np.max(mi_vals)) if mi_vals else 0.0,
        "weight_cos_mean": float(np.mean([w["weight_cos_mean"] for w in weight_stats])) if weight_stats else 0.0,
        "weight_cos_min": float(np.min([w["weight_cos_min"] for w in weight_stats])) if weight_stats else 0.0,
        "weight_cos_max": float(np.max([w["weight_cos_max"] for w in weight_stats])) if weight_stats else 0.0,
    }

    return summary, layer_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--batches-per-task", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()

    # Find run dirs containing both config.yaml and state.pt
    run_dirs = []
    for cfg_path in results_dir.rglob("config.yaml"):
        run_dir = cfg_path.parent
        state_path = run_dir / "state.pt"
        if state_path.exists():
            run_dirs.append(run_dir)

    # Select the latest run per experiment (by state.pt mtime)
    latest = {}
    for rd in run_dirs:
        exp = rd.parent.name
        state_path = rd / "state.pt"
        mtime = state_path.stat().st_mtime
        if exp not in latest or mtime > latest[exp][0]:
            latest[exp] = (mtime, rd)

    summaries = []
    layer_rows_all = []
    for exp, (_, rd) in sorted(latest.items()):
        try:
            out = analyze_experiment(rd, args.batches_per_task, args.device)
            if out is None:
                continue
            summary, layer_rows = out
            summaries.append(summary)
            layer_rows_all.extend(layer_rows)
            print(f"[OK] {exp} ({rd})")
        except Exception as e:
            print(f"[SKIP] {exp} ({rd}) -> {e}")

    # Write CSVs
    import csv

    summary_path = results_dir / "_routing_summary.csv"
    layers_path = results_dir / "_routing_layers.csv"

    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    if layer_rows_all:
        with layers_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows_all[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows_all)

    print(f"Summary saved to {summary_path}")
    print(f"Layer metrics saved to {layers_path}")


if __name__ == "__main__":
    main()
