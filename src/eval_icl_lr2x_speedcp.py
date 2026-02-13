"""
SpeedCP-based conditional conformal diagnostics for ICL linear regression runs.

This mirrors `eval_icl_lr2x_condconf.py`, but uses SpeedCP (lambda-path + S-path)
to compute conditional conformal cutoffs much faster than the per-test CVXPY solve.

We conformalize absolute residuals r = |y - yhat| and produce symmetric intervals:
  [yhat - q(x), yhat + q(x)].

Outputs (under results/_icl_curves/):
  - compare_lr2x_speedcp_summary_<family>_alpha0p05.csv
"""

from __future__ import annotations

# Default to 1 BLAS thread per process to avoid oversubscription when using workers.
import os as _os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_k, "1")

import argparse
import csv
from pathlib import Path
import random
import re
import warnings

import numpy as np
import torch
import yaml

from ckpt_utils import latest_run_dir, select_ckpt_path
from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler
from uq.conditional_conformal import (
    InterceptPhi,
    LinearPhi,
    LinearRFFPhi,
    NormBinsPhi,
    split_indices,
)
from uq.speedcp_conformal import SpeedCPSymmetricAbs

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable=None, *args, **kwargs):
        return iterable


EXPS = {
    "s_nlr80_series": [
        "S13_gpt2_w32_d6_nlr80x40",
        "S14_gpt2_w64_d6_nlr80x40",
        "S15_gpt2_w128_d6_nlr80x40",
        "S16_gpt2_w256_d6_nlr80x40",
        "S17_gpt2_w64_d2_nlr80x40",
        "S18_gpt2_w64_d4_nlr80x40",
        "S19_gpt2_w64_d8_nlr80x40",
        "S20_gpt2_w64_d12_nlr80x40",
        "S21_gpt2_tiny_nlr80x40",
        "S22_gpt2_small_nlr80x40",
        "S23_gpt2_medium_nlr80x40",
        "S24_gpt2_large_nlr80x40",
    ],
}


def build_conf(model_cfg: dict):
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


@torch.no_grad()
def collect_xyhat(
    model,
    *,
    task_name: str,
    task_kwargs: dict,
    data_name: str,
    n_dims: int,
    n_points: int,
    num_eval_examples: int,
    batch_size: int,
    device: str,
):
    """
    Returns (xs, ys, yhat) as torch tensors on CPU:
      xs: (N, n_points, n_dims)
      ys: (N, n_points)
      yhat: (N, n_points)
    """
    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims=n_dims)
    task_sampler = get_task_sampler(task_name, n_dims, batch_size, **task_kwargs)

    model = model.to(device).eval()
    xs_all, ys_all, yhat_all = [], [], []
    for _ in range(num_eval_examples // batch_size):
        xs = data_sampler.sample_xs(n_points, batch_size, n_dims, device=device)
        task = task_sampler(device=device)
        ys = task.evaluate(xs)
        yhat = model(xs, ys)
        xs_all.append(xs.detach().cpu())
        ys_all.append(ys.detach().cpu())
        yhat_all.append(yhat.detach().cpu())
    return torch.cat(xs_all, dim=0), torch.cat(ys_all, dim=0), torch.cat(yhat_all, dim=0)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _build_phi(
    *,
    name: str,
    num_bins: int,
    rff_dim: int,
    rff_kernel: str,
    gamma: float,
    seed: int,
) -> object:
    if name == "intercept":
        return InterceptPhi()
    if name == "linear":
        return LinearPhi(standardize=True)
    if name == "linear_rff":
        if int(rff_dim) <= 0:
            raise SystemExit("--rff-dim must be >0 when --phi=linear_rff")
        return LinearRFFPhi(n_rff=int(rff_dim), kernel=str(rff_kernel), gamma=float(gamma), seed=int(seed))
    if name == "norm_bins":
        return NormBinsPhi(num_bins=num_bins)
    raise SystemExit("--phi must be one of: intercept, linear, linear_rff, norm_bins")


def _conformal_q_abs(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile with finite-sample correction.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(scores.size)
    if n <= 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = max(1, min(k, n))
    return float(np.partition(scores, k - 1)[k - 1])


def _parse_int_csv(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(int(tok))
    return out


def _safe_tag(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "run"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_")
    return s or "run"


def _s_proj(xs_ctx: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """
    Compute s_proj = ||(I - P)x_query|| where P projects onto span(rows(xs_ctx)).

    xs_ctx: (N, L, d)
    x_query: (N, d)
    returns: (N,) nonnegative
    """
    xs_ctx = np.asarray(xs_ctx, dtype=np.float64)
    x_query = np.asarray(x_query, dtype=np.float64)
    if xs_ctx.ndim != 3:
        raise ValueError("xs_ctx must have shape (N, L, d)")
    if x_query.ndim != 2:
        raise ValueError("x_query must have shape (N, d)")
    if xs_ctx.shape[0] != x_query.shape[0]:
        raise ValueError("xs_ctx and x_query must have matching N")
    if xs_ctx.shape[2] != x_query.shape[1]:
        raise ValueError("xs_ctx and x_query must have matching d")

    n, L, _d = xs_ctx.shape
    if L == 0:
        return np.linalg.norm(x_query, axis=1)

    XXT = xs_ctx @ np.transpose(xs_ctx, (0, 2, 1))  # (N, L, L)
    v = np.einsum("nld,nd->nl", xs_ctx, x_query)  # (N, L)
    a_v = np.linalg.solve(XXT, v[..., None])[..., 0]  # (N, L)
    x_norm2 = np.einsum("nd,nd->n", x_query, x_query)
    proj_norm2 = np.einsum("nl,nl->n", v, a_v)
    s2 = np.maximum(x_norm2 - proj_norm2, 0.0)
    return np.sqrt(s2)


def _bootstrap_mean_ci_90(values: np.ndarray, *, bootstrap_idx: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if bootstrap_idx.size == 0:
        return (float("nan"), float("nan"))
    means = values[bootstrap_idx].mean(axis=1)
    means.sort()
    b = int(means.shape[0])
    lo_i = int(0.05 * b)
    hi_i = int(0.95 * b)
    lo_i = max(0, min(lo_i, b - 1))
    hi_i = max(0, min(hi_i, b - 1))
    return (float(means[lo_i]), float(means[hi_i]))


def _box_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"min": float("nan"), "q1": float("nan"), "med": float("nan"), "q3": float("nan"), "max": float("nan")}
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    p05, p95 = np.quantile(values, [0.05, 0.95])
    iqr = q3 - q1
    whislo = np.min(values[values >= (q1 - 1.5 * iqr)]) if iqr > 0 else float(np.min(values))
    whishi = np.max(values[values <= (q3 + 1.5 * iqr)]) if iqr > 0 else float(np.max(values))
    return {
        "min": float(np.min(values)),
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "max": float(np.max(values)),
        "p05": float(p05),
        "p95": float(p95),
        "whislo": float(whislo),
        "whishi": float(whishi),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"cvxpy\.atoms\.affine\.vec")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Optional explicit run directory (contains config.yaml + model_*.pt/state.pt). "
        "If set, --family/--exp are ignored.",
    )
    parser.add_argument("--family", choices=list(EXPS.keys()), default="s_nlr80_series")
    parser.add_argument("--exp", default="", help="Optional single experiment name (within --family)")
    parser.add_argument("--prefer-model-step", type=int, default=500000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=-1,
        help="Seed for sampling evaluation episodes (xs/tasks/noise). If negative, defaults to --seed.",
    )
    parser.add_argument("--num-eval-examples", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-n-points", type=int, default=81)
    parser.add_argument("--icl-start", type=int, default=4)
    parser.add_argument("--icl-step", type=int, default=4)
    parser.add_argument(
        "--icl-lens",
        default="",
        help="Optional comma-separated ICL lengths to evaluate (overrides --icl-start/--icl-step).",
    )
    parser.add_argument("--bootstrap-trials", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--x-features",
        default="query_x",
        choices=["query_x", "s_proj"],
        help="Which covariates to condition on for SpeedCP.",
    )
    parser.add_argument("--phi", default="intercept", choices=["intercept", "linear", "linear_rff", "norm_bins"])
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--rff-dim", type=int, default=0)
    parser.add_argument("--rff-kernel", default="rbf", choices=["rbf", "laplacian"])
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "laplacian"])
    parser.add_argument("--gamma", type=float, default=0.025)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1)
    parser.add_argument("--speedcp-test-workers", type=int, default=1)
    parser.add_argument("--speedcp-worker-backend", choices=["thread", "process"], default="process")
    parser.add_argument("--speedcp-randomize", action="store_true", help="Enable randomized alpha0 (more variance).")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--task-index", type=int, default=None, metavar="I")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    if not (0.0 < float(args.alpha) < 1.0):
        raise SystemExit("--alpha must be in (0,1)")
    if not (0.0 < float(args.calib_frac) < 1.0):
        raise SystemExit("--calib-frac must be in (0,1)")
    if int(args.icl_step) <= 0:
        raise SystemExit("--icl-step must be > 0")
    if int(args.speedcp_test_workers) <= 0:
        raise SystemExit("--speedcp-test-workers must be >= 1")
    if float(args.gamma) <= 0:
        raise SystemExit("--gamma must be > 0")
    if float(args.lam) <= 0:
        raise SystemExit("--lambda must be > 0")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device=cuda requested but CUDA not available; falling back to cpu.")
        device = "cpu"

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "_icl_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if explicit_run_dir is not None:
        exp_list = ["__explicit_run_dir__"]
    else:
        exp_list = EXPS[args.family]
        if args.exp:
            if args.exp not in exp_list:
                raise SystemExit(f"--exp must be one of: {exp_list}")
            exp_list = [args.exp]

    rows = []
    for exp in tqdm(exp_list, desc="Runs", disable=bool(args.disable_tqdm)):
        if explicit_run_dir is not None:
            rd = explicit_run_dir
            exp_name = explicit_run_dir.parent.name or explicit_run_dir.name
        else:
            exp_dir = results_dir / exp
            rd = latest_run_dir(exp_dir, prefer_step=args.prefer_model_step)
            exp_name = exp
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
        if args.task_index is not None:
            if args.task_index < 0 or args.task_index >= len(tasks):
                print("[SKIP]", exp, f"(--task-index {args.task_index} out of range [0, {len(tasks)-1}])")
                continue
            tasks = [tasks[args.task_index]]

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

        for task in tqdm(tasks, desc=f"{exp_name} tasks", leave=False, disable=bool(args.disable_tqdm)):
            task_name = task["name"]
            task_kwargs = task.get("kwargs", {}) or {}
            task_label = _task_to_label(task_name, task_kwargs)

            data_seed = int(args.data_seed) if int(args.data_seed) >= 0 else int(args.seed)
            _seed_everything(data_seed)
            xs, ys, yhat = collect_xyhat(
                model,
                task_name=task_name,
                task_kwargs=task_kwargs,
                data_name=data_name,
                n_dims=n_dims,
                n_points=int(args.max_n_points),
                num_eval_examples=int(args.num_eval_examples),
                batch_size=int(batch_size),
                device=device,
            )

            split = split_indices(int(xs.shape[0]), calib_frac=float(args.calib_frac), seed=int(args.seed))
            cal_idx = torch.from_numpy(split.calib_idx).long()
            test_idx = torch.from_numpy(split.test_idx).long()

            bootstrap_idx = np.empty((0, 0), dtype=np.int64)
            if int(args.bootstrap_trials) > 0:
                rng = np.random.default_rng(int(args.seed))
                n_test = int(len(split.test_idx))
                bootstrap_idx = rng.integers(0, n_test, size=(int(args.bootstrap_trials), n_test), dtype=np.int64)

            if str(args.icl_lens).strip():
                icl_lens = np.array(sorted(_parse_int_csv(str(args.icl_lens))), dtype=np.int64)
            else:
                max_pt = int(args.max_n_points)
                icl_lens = np.arange(int(args.icl_start), max_pt, int(args.icl_step), dtype=np.int64)
                last_l = max_pt - 1
                if last_l >= 1 and last_l not in icl_lens:
                    icl_lens = np.unique(np.concatenate([icl_lens, [last_l]]))
            icl_lens = icl_lens[(icl_lens >= 1) & (icl_lens < int(args.max_n_points))]

            for L in tqdm(icl_lens.tolist(), desc=f"{exp_name}:{task_name} L", leave=False, disable=bool(args.disable_tqdm)):
                if args.x_features == "query_x":
                    x_cal = xs[cal_idx, L, :].numpy()
                    x_te = xs[test_idx, L, :].numpy()
                    diag_feat_te = np.linalg.norm(x_te, axis=1)
                else:
                    xs_cal_ctx = xs[cal_idx, :L, :].numpy()
                    xs_te_ctx = xs[test_idx, :L, :].numpy()
                    xq_cal = xs[cal_idx, L, :].numpy()
                    xq_te = xs[test_idx, L, :].numpy()
                    s_cal = _s_proj(xs_cal_ctx, xq_cal)
                    s_te = _s_proj(xs_te_ctx, xq_te)
                    x_cal = s_cal.reshape(-1, 1)
                    x_te = s_te.reshape(-1, 1)
                    diag_feat_te = s_te

                y_cal = ys[cal_idx, L].numpy()
                yhat_cal = yhat[cal_idx, L].numpy()
                y_te = ys[test_idx, L].numpy()
                yhat_te = yhat[test_idx, L].numpy()

                # Split conformal baseline (marginal).
                q_split = _conformal_q_abs(np.abs(y_cal - yhat_cal), alpha=float(args.alpha))
                lo_split = yhat_te - q_split
                hi_split = yhat_te + q_split
                covered_split = (y_te >= lo_split) & (y_te <= hi_split)

                phi = _build_phi(
                    name=str(args.phi),
                    num_bins=int(args.num_bins),
                    rff_dim=int(args.rff_dim),
                    rff_kernel=str(args.rff_kernel),
                    gamma=float(args.gamma),
                    seed=int(args.seed),
                )
                calib = SpeedCPSymmetricAbs(
                    phi=phi,
                    kernel=str(args.kernel),
                    gamma=float(args.gamma),
                    lam=float(args.lam),
                    seed=int(args.seed),
                    worker_backend=str(args.speedcp_worker_backend),
                )
                calib.fit(x_cal, yhat_cal, y_cal)
                q_te = calib.cutoff(
                    x_te,
                    yhat_te,
                    alpha=float(args.alpha),
                    randomize=bool(args.speedcp_randomize),
                    workers=int(args.speedcp_test_workers),
                )

                lo = yhat_te - q_te
                hi = yhat_te + q_te
                covered = (y_te >= lo) & (y_te <= hi)

                w_te = 2.0 * np.asarray(q_te, dtype=np.float64).reshape(-1)
                box = _box_stats(w_te)

                cov_ci_low, cov_ci_high = (float("nan"), float("nan"))
                w_ci_low, w_ci_high = (float("nan"), float("nan"))
                if bootstrap_idx.size:
                    cov_ci_low, cov_ci_high = _bootstrap_mean_ci_90(
                        np.asarray(covered, dtype=np.float64).reshape(-1), bootstrap_idx=bootstrap_idx
                    )
                    w_ci_low, w_ci_high = _bootstrap_mean_ci_90(w_te, bootstrap_idx=bootstrap_idx)

                fit_info = calib.last_fit_info
                row = {
                    "family": args.family,
                    "exp": exp_name,
                    "task": task_label,
                    "icl_len": int(L),
                    "alpha": float(args.alpha),
                    "x_features": str(args.x_features),
                    "phi": str(args.phi),
                    "num_bins": int(args.num_bins),
                    "kernel": str(args.kernel),
                    "gamma": float(args.gamma),
                    "lambda": float(args.lam),
                    "speedcp_randomize": int(bool(args.speedcp_randomize)),
                    "speedcp_workers": int(args.speedcp_test_workers),
                    "speedcp_backend": str(args.speedcp_worker_backend),
                    "coverage": float(np.mean(covered)),
                    "avg_width": float(np.mean(w_te)),
                    "split_coverage": float(np.mean(covered_split)),
                    "split_avg_width": float(2.0 * q_split),
                    "width_bootstrap_low": float(w_ci_low),
                    "width_bootstrap_high": float(w_ci_high),
                    "cov_bootstrap_low": float(cov_ci_low),
                    "cov_bootstrap_high": float(cov_ci_high),
                    "width_p05": float(box["p05"]),
                    "width_p25": float(box["q1"]),
                    "width_p50": float(box["med"]),
                    "width_p75": float(box["q3"]),
                    "width_p95": float(box["p95"]),
                    "width_whislo": float(box["whislo"]),
                    "width_whishi": float(box["whishi"]),
                    "width_min": float(box["min"]),
                    "width_max": float(box["max"]),
                    "n_total": int(xs.shape[0]),
                    "n_cal": int(len(split.calib_idx)),
                    "n_test": int(len(split.test_idx)),
                    "run_dir": str(rd),
                    "ckpt": str(ckpt_path),
                    "n_points_eval": int(args.max_n_points),
                    "num_eval_examples": int(args.num_eval_examples),
                    "batch_size": int(batch_size),
                    "seed": int(args.seed),
                    "data_seed": int(data_seed),
                    "calib_frac": float(args.calib_frac),
                    "bootstrap_trials": int(args.bootstrap_trials),
                    "device": device,
                    "speedcp_tune_seconds": float(fit_info.tune_seconds) if fit_info else float("nan"),
                    "speedcp_stage2_seconds": float(fit_info.stage2_seconds) if fit_info else float("nan"),
                    "diag_feat_mean": float(np.mean(diag_feat_te)),
                }
                rows.append(row)
                print(
                    (
                        "[METRIC]"
                        f" exp={exp_name}"
                        f" task={task_label}"
                        f" L={int(L)}"
                        f" cov={float(np.mean(covered)):.6f}"
                        f" width={float(np.mean(w_te)):.6f}"
                        f" split_cov={float(np.mean(covered_split)):.6f}"
                        f" split_width={float(2.0 * q_split):.6f}"
                    ),
                    flush=True,
                )

    alpha_tag = str(args.alpha).replace(".", "p")
    group_base = args.family if args.family else "run"
    if explicit_run_dir is not None:
        group_base = explicit_run_dir.parent.name or explicit_run_dir.name
    if args.output_suffix:
        group_base = f"{group_base}_{args.output_suffix}"
    group_tag = _safe_tag(group_base)
    csv_path = out_dir / f"compare_lr2x_speedcp_summary_{group_tag}_alpha{alpha_tag}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", csv_path)
    else:
        print("[WARN] no rows produced")


if __name__ == "__main__":
    main()

