"""
Long-form routing analysis on CPU for key experiments (no training).

Outputs:
  - results/_routing_summary_long.csv
  - results/_routing_layers_long.csv
  - results/_routing_long_notes.md
  - results/_routing_figs/<exp>/... heatmaps
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

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
    if "noise_std" in kwargs:
        return f"sigma={kwargs['noise_std']}"
    items = ",".join([f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())])
    return f"{name}({items})"


def _collect_counts(model, tasks, data_sampler, n_dims, n_positions, batch_size, batches_per_task):
    n_layer = len(model._mlps)
    moe_indices = [i for i, mlp in enumerate(model._mlps) if hasattr(mlp, "router")]
    if not moe_indices:
        return None, None
    num_experts = model._mlps[moe_indices[0]].num_experts
    counts = np.zeros((n_layer, len(tasks), num_experts), dtype=np.int64)
    totals = np.zeros((n_layer, len(tasks)), dtype=np.int64)

    with torch.no_grad():
        for t_idx, task in enumerate(tasks):
            task_sampler = get_task_sampler(
                task["name"], n_dims, batch_size, **(task.get("kwargs", {}) or {})
            )
            for _ in range(batches_per_task):
                xs = data_sampler.sample_xs(n_positions, batch_size, n_dims)
                task_obj = task_sampler()
                ys = task_obj.evaluate(xs)
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

                    if hasattr(mlp, "router"):
                        _, _, routing = mlp(H, return_routing_info=True)
                        top_k = routing["top_k_indices"].reshape(-1)
                        bc = torch.bincount(top_k, minlength=num_experts)
                        counts[layer_idx, t_idx] += bc.cpu().numpy()
                        totals[layer_idx, t_idx] += int(top_k.numel())

                        mlp_out, _, _ = mlp(H)
                        H = H + mlp_out
                    else:
                        H = H + mlp(H)
                    if model.layernorm:
                        H = ln2(H)

    return counts, totals


def _weight_cos_stats(model):
    stats = []
    for layer_idx, mlp in enumerate(model._mlps):
        if not hasattr(mlp, "router"):
            stats.append(
                {
                    "layer": layer_idx,
                    "weight_cos_mean": 0.0,
                    "weight_cos_min": 0.0,
                    "weight_cos_max": 0.0,
                }
            )
            continue
        num_experts = mlp.num_experts
        w1 = mlp.w1.detach().cpu().reshape(num_experts, -1)
        w2 = mlp.w2.detach().cpu().reshape(num_experts, -1)
        vec = torch.cat([w1, w2], dim=1).numpy()
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        vec_n = vec / norms
        cos = vec_n @ vec_n.T
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


def analyze_one(run_dir: Path, batches_per_task: int, out_dir: Path):
    cfg = _safe_load_yaml(run_dir / "config.yaml")
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    if not (model_cfg.get("use_moe", False) or model_cfg.get("moe_layers")):
        return None
    if model_cfg.get("family") != "EncoderTF":
        return None

    conf = _build_conf(model_cfg)
    model = build_model(conf)
    state = torch.load(run_dir / "state.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    tasks = training_cfg.get("tasks", []) or []
    if not tasks:
        return None

    n_dims = model_cfg["n_dims"]
    n_positions = model_cfg["n_positions"]
    batch_size = training_cfg.get("batch_size", 64)
    data_sampler = get_data_sampler(training_cfg.get("data", "gaussian"), n_dims=n_dims)

    counts, totals = _collect_counts(
        model, tasks, data_sampler, n_dims, n_positions, batch_size, batches_per_task
    )
    if counts is None:
        return None
    num_experts = model_cfg["num_experts"]
    task_labels = [_task_label(t) for t in tasks]

    # per-layer metrics
    layer_rows = []
    entropy_vals, js_vals, mi_vals = [], [], []
    weight_stats = _weight_cos_stats(model)
    for l in range(counts.shape[0]):
        counts_l = counts[l]
        total_l = counts_l.sum()
        if total_l <= 0:
            continue
        p_e = counts_l.sum(axis=0) / (total_l + 1e-12)
        ent = _entropy(p_e) / (np.log(num_experts) + 1e-12)
        entropy_vals.append(ent)
        js = 0.0
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
                "usage_frac": json.dumps((p_e).tolist()),
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

    # heatmaps: top 3 layers by JS
    js_scores = [r["js_div"] for r in layer_rows]
    top_layers = sorted(range(len(js_scores)), key=lambda i: js_scores[i], reverse=True)[:3]
    fractions = counts / (counts.sum(axis=2, keepdims=True) + 1e-12)
    exp_out = out_dir / run_dir.parent.name
    exp_out.mkdir(parents=True, exist_ok=True)

    for l in top_layers:
        plt.figure(figsize=(6, 3))
        sns.heatmap(
            fractions[l],
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=True,
            xticklabels=[f"E{e}" for e in range(num_experts)],
            yticklabels=task_labels,
        )
        plt.title(f"{run_dir.parent.name} Layer {l} Task->Expert")
        plt.xlabel("Expert")
        plt.ylabel("Task")
        plt.tight_layout()
        plt.savefig(exp_out / f"layer{l:02d}.png", dpi=200)
        plt.close()

    agg = counts.sum(axis=0)
    agg_frac = agg / (agg.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(6, 3))
    sns.heatmap(
        agg_frac,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        xticklabels=[f"E{e}" for e in range(num_experts)],
        yticklabels=task_labels,
    )
    plt.title(f"{run_dir.parent.name} Aggregate Task->Expert")
    plt.xlabel("Expert")
    plt.ylabel("Task")
    plt.tight_layout()
    plt.savefig(exp_out / "aggregate.png", dpi=200)
    plt.close()

    return summary, layer_rows


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--batches-per-task", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "_routing_figs_long"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_exps = [
        # Strong specialization
        "E53_moe8_aux0",
        "E07_moe4_seq_routing",
        "E60_moe16_seqaux0",
        "E59_moe4_seqaux0",
        "E06_moe16_4noise",
        "E09_moe4_no_aux",
        # Baseline contrasts
        "E12_moe8_8noise",
        "E54_moe8_aux10",
        "E57_moe8_top4",
        "E58_moe8_top8",
    ]

    summaries = []
    layer_rows = []
    notes = []

    for exp in key_exps:
        rd = latest_run(results_dir, exp)
        if rd is None:
            notes.append(f"- {exp}: missing state/config")
            continue
        print(f"[RUN] {exp}")
        out = analyze_one(rd, args.batches_per_task, out_dir)
        if out is None:
            notes.append(f"- {exp}: skipped (not MoE EncoderTF)")
            continue
        summary, layers = out
        summaries.append(summary)
        layer_rows.extend(layers)
        notes.append(f"- {exp}: OK ({rd})")

    import csv
    if summaries:
        with (results_dir / "_routing_summary_long.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    if layer_rows:
        with (results_dir / "_routing_layers_long.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows)

    with (results_dir / "_routing_long_notes.md").open("w", encoding="utf-8") as f:
        f.write("Long routing analysis notes\n")
        f.write("===========================\n\n")
        f.write("Experiments:\n")
        for line in notes:
            f.write(line + "\n")

    print("Saved long sweep outputs to:", results_dir)
if __name__ == "__main__":
    main()
