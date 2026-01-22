"""
Multi-sparsity algorithm-selection analysis (CPU-friendly).

Outputs (under results/):
  - _multisp_summary.csv: per-experiment summary
  - _multisp_tasks.csv: per-task losses + (if MoE) expert routing at top JS layer
  - _multisp_layers.csv: per-layer JS divergence (MoE only)
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


def _task_label(task):
    kwargs = task.get("kwargs", {}) or {}
    if "sparsity" in kwargs:
        return f"sparsity={kwargs['sparsity']}"
    return task.get("name", "task")


def _forward_with_optional_routing(model, H, ys, want_routing):
    # Manual forward to access routing info if MoE
    routing_counts = None
    if want_routing:
        num_layers = len(model._mlps)
        num_experts = model._mlps[0].num_experts
        routing_counts = np.zeros((num_layers, num_experts), dtype=np.int64)

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

        if want_routing:
            mlp_out, _, routing = mlp(H, return_routing_info=True)
            top_k = routing["top_k_indices"].reshape(-1)
            bc = torch.bincount(top_k, minlength=routing_counts.shape[1])
            routing_counts[layer_idx] += bc.detach().cpu().numpy()
        else:
            mlp_out = mlp(H)

        H = H + mlp_out
        if model.layernorm:
            H = ln2(H)

    return H, routing_counts


def analyze_experiment(run_dir: Path, batches_per_task: int, device: str):
    cfg = _safe_load_yaml(run_dir / "config.yaml")
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    tasks = training_cfg.get("tasks", []) or []
    if not tasks:
        return None

    # Only keep sparse_linear_regression tasks
    if not all(t.get("name") == "sparse_linear_regression" for t in tasks):
        return None

    conf = _build_conf(model_cfg)
    model = build_model(conf)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()

    state = torch.load(run_dir / "state.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    n_dims = model_cfg["n_dims"]
    n_positions = model_cfg["n_positions"]
    batch_size = training_cfg.get("batch_size", 64)
    data_sampler = get_data_sampler(training_cfg.get("data", "gaussian"), n_dims=n_dims)

    use_moe = bool(model_cfg.get("use_moe", False))
    num_layers = len(model._mlps)
    num_experts = model_cfg.get("num_experts", 0)

    # Storage
    loss_sums = np.zeros(len(tasks), dtype=np.float64)
    loss_sums2 = np.zeros(len(tasks), dtype=np.float64)
    loss_counts = np.zeros(len(tasks), dtype=np.int64)

    counts = None
    if use_moe:
        counts = np.zeros((num_layers, len(tasks), num_experts), dtype=np.int64)

    with torch.no_grad():
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

                zs = model._combine(xs, ys)
                H = model._read_in(zs)
                H, routing_counts = _forward_with_optional_routing(
                    model, H, ys, want_routing=use_moe
                )
                if use_moe:
                    counts[:, t_idx] += routing_counts

                pred = model._read_out(H)[:, -1, 0]
                loss = (pred - ys[:, -1]).pow(2).mean().item()
                loss_sums[t_idx] += loss
                loss_sums2[t_idx] += loss * loss
                loss_counts[t_idx] += 1

    loss_means = (loss_sums / (loss_counts + 1e-12)).tolist()
    loss_vars = (loss_sums2 / (loss_counts + 1e-12)) - np.square(
        loss_sums / (loss_counts + 1e-12)
    )
    loss_stds = np.sqrt(np.maximum(loss_vars, 0.0)).tolist()

    # Select top JS layer if MoE
    top_layer = None
    js_scores = None
    if use_moe:
        js_scores = []
        for l in range(num_layers):
            js = 0.0
            pairs = 0
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    js += _js_divergence(counts[l, i], counts[l, j])
                    pairs += 1
            js_scores.append(js / max(pairs, 1))
        top_layer = int(np.argmax(js_scores)) if js_scores else None

    return {
        "exp": run_dir.parent.name,
        "run_dir": str(run_dir),
        "use_moe": use_moe,
        "num_experts": num_experts,
        "top_k": model_cfg.get("top_k", 1),
        "seq_level_routing": model_cfg.get("seq_level_routing", False),
        "aux_loss_coef": model_cfg.get("aux_loss_coef", 0.0),
        "router_noise": model_cfg.get("router_noise", False),
        "tasks": len(tasks),
        "batches_per_task": batches_per_task,
        "task_labels": json.dumps([_task_label(t) for t in tasks]),
        "loss_means": json.dumps(loss_means),
        "loss_stds": json.dumps(loss_stds),
        "top_layer": top_layer if top_layer is not None else "",
        "js_top_layer": float(js_scores[top_layer]) if (use_moe and js_scores) else "",
        "counts": counts,
        "js_scores": js_scores,
    }


def _select_exps(results_dir: Path):
    overview = results_dir / "_exp_overview.json"
    if not overview.exists():
        return []
    rows = json.loads(overview.read_text(encoding="utf-8"))
    exps = []
    for r in rows:
        tasks = json.loads(r.get("tasks", "[]"))
        if not tasks:
            continue
        if any(t != "sparse_linear_regression" for t in tasks):
            continue
        # multi-sparsity: keep 4+ tasks to avoid 2-task degeneracy
        if len(tasks) < 4:
            continue
        exps.append(r["exp"])
    return sorted(set(exps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--batches-per-task", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--exps", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if args.exps:
        exps = [e.strip() for e in args.exps.split(",") if e.strip()]
    else:
        exps = _select_exps(results_dir)

    summaries = []
    task_rows = []
    layer_rows = []

    for exp in exps:
        # pick latest run by state.pt mtime
        exp_dir = results_dir / exp
        if not exp_dir.exists():
            print(f"[SKIP] {exp} (missing)")
            continue
        runs = []
        for cfg in exp_dir.rglob("config.yaml"):
            state = cfg.parent / "state.pt"
            if state.exists():
                runs.append((state.stat().st_mtime, cfg.parent))
        if not runs:
            print(f"[SKIP] {exp} (no state)")
            continue
        runs.sort(key=lambda x: x[0], reverse=True)
        run_dir = runs[0][1]

        try:
            out = analyze_experiment(run_dir, args.batches_per_task, args.device)
            if out is None:
                print(f"[SKIP] {exp} (not multi-sparsity)")
                continue
        except Exception as e:
            print(f"[SKIP] {exp} -> {e}")
            continue

        summaries.append({k: v for k, v in out.items() if k not in {"counts", "js_scores"}})

        # Per-task rows
        task_labels = json.loads(out["task_labels"])
        loss_means = json.loads(out["loss_means"])
        loss_stds = json.loads(out["loss_stds"])
        use_moe = out["use_moe"]
        top_layer = out["top_layer"] if out["top_layer"] != "" else None

        for idx, (lbl, lm, ls) in enumerate(zip(task_labels, loss_means, loss_stds)):
            row = {
                "exp": out["exp"],
                "task": lbl,
                "loss_mean": lm,
                "loss_std": ls,
                "top_layer": top_layer if top_layer is not None else "",
            }
            if use_moe and top_layer is not None:
                counts = out["counts"][top_layer, idx]
                probs = counts / (counts.sum() + 1e-12)
                top_e = int(np.argmax(probs)) if probs.size else -1
                row.update(
                    {
                        "expert_probs": json.dumps(probs.tolist()),
                        "top_expert": top_e,
                        "top_prob": float(probs[top_e]) if top_e >= 0 else 0.0,
                    }
                )
            else:
                row.update({"expert_probs": "", "top_expert": "", "top_prob": ""})
            task_rows.append(row)

        # Per-layer JS (MoE only)
        if use_moe and out["js_scores"]:
            for l, js in enumerate(out["js_scores"]):
                layer_rows.append(
                    {
                        "exp": out["exp"],
                        "layer": l,
                        "js_div": float(js),
                    }
                )

        print(f"[OK] {exp}")

    # Save CSVs
    import csv

    if summaries:
        with (results_dir / "_multisp_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    if task_rows:
        with (results_dir / "_multisp_tasks.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
            writer.writeheader()
            writer.writerows(task_rows)

    if layer_rows:
        with (results_dir / "_multisp_layers.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows)

    print("Saved _multisp_summary.csv / _multisp_tasks.csv / _multisp_layers.csv")


if __name__ == "__main__":
    main()
