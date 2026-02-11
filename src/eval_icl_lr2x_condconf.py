"""
Conditional conformal diagnostics for ICL linear regression runs.

This extends the existing split-conformal script by using the method from:
  Chen, Cherian, Candès (2023) - "Conformal Prediction with Conditional Guarantees"
via the `conditionalconformal` package (jjcherian/conditional-conformal).

For each ICL length L we:
  - sample evaluation episodes (xs, ys)
  - compute model predictions yhat at each position
  - fit conditional conformal on calibration examples using x_features at position L
  - compute empirical test coverage and average interval width

Default x_features: the query x at position L (shape n_dims).
Default Phi: norm-binned one-hot (group conditional guarantees on ||x||).

Optional alternative covariate: s_proj
  s_proj(Z) = ||(I - P_L) x_query|| where P_L is the projection onto the span of the
  L context inputs (rows of X_ctx). This is the natural "hardness" scale for noiseless
  underdetermined linear regression (L < d) under a min-norm least-squares baseline.

Optional RKHS component:
  Pass --kernel (e.g. rbf) to enable the RKHS variant in jjcherian/conditional-conformal.
  Note: the upstream implementation does not support --exact when kernel is enabled.

Outputs (under results/_icl_curves/):
  - compare_lr2x_condconf_summary_<family>_alpha0p05.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import warnings

import numpy as np
import torch
import yaml

from ckpt_utils import latest_run_dir, select_ckpt_path
from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler
from uq.conditional_conformal import CondConfSymmetricAbs, InterceptPhi, LinearPhi, NormBinsPhi, split_indices

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


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


def _build_phi(name: str, num_bins: int) -> object:
    if name == "intercept":
        return InterceptPhi()
    if name == "linear":
        return LinearPhi(standardize=True)
    if name == "norm_bins":
        return NormBinsPhi(num_bins=num_bins)
    raise SystemExit("--phi must be one of: intercept, linear, norm_bins")


def _conformal_q_abs(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile with finite-sample correction, matching eval_icl_lr2x_conformal.py.
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

    For L <= d and full row rank, we use:
      s^2 = ||x||^2 - v^T (X X^T)^{-1} v,  v = X x
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
    """
    Returns a 90% bootstrap CI for the mean, matching src/eval.py convention:
      - resample indices shape: (B, n)
      - compute bootstrap means
      - take the 5th and 95th percentiles via order statistics
    """
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
    """
    Boxplot stats for interval widths (Tukey style whiskers, 1.5*IQR).
    Returns a dict compatible with matplotlib.axes.Axes.bxp when keys are mapped.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {
            "p05": float("nan"),
            "q1": float("nan"),
            "med": float("nan"),
            "q3": float("nan"),
            "p95": float("nan"),
            "whislo": float("nan"),
            "whishi": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    p05, q1, med, q3, p95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
    iqr = float(q3 - q1)
    lo_fence = float(q1 - 1.5 * iqr)
    hi_fence = float(q3 + 1.5 * iqr)
    v_min = float(np.min(v))
    v_max = float(np.max(v))

    in_lo = v[v >= lo_fence]
    in_hi = v[v <= hi_fence]
    whislo = float(np.min(in_lo)) if in_lo.size else v_min
    whishi = float(np.max(in_hi)) if in_hi.size else v_max

    return {
        "p05": float(p05),
        "q1": float(q1),
        "med": float(med),
        "q3": float(q3),
        "p95": float(p95),
        "whislo": float(whislo),
        "whishi": float(whishi),
        "min": float(v_min),
        "max": float(v_max),
    }


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"cvxpy\.atoms\.affine\.vec")

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument(
        "--run-dir",
        default="",
        help="Optional explicit run directory (contains config.yaml + model_*.pt/state.pt). "
        "If set, --family/--exp are ignored.",
    )
    parser.add_argument("--family", choices=list(EXPS.keys()), default="")
    parser.add_argument("--exp", default="", help="Optional single experiment name (within --family)")
    parser.add_argument("--prefer-model-step", type=int, default=300000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calib-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-eval-examples", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-n-points", type=int, default=41)
    parser.add_argument(
        "--icl-start",
        type=int,
        default=1,
        help="First ICL length L to evaluate (inclusive). Only used if --icl-lens is empty.",
    )
    parser.add_argument(
        "--icl-step",
        type=int,
        default=1,
        help="ICL length step size. Only used if --icl-lens is empty.",
    )
    parser.add_argument(
        "--icl-lens",
        default="",
        help="Optional comma-separated ICL lengths to evaluate (overrides --icl-start/--icl-step). "
        "Example: --icl-lens 4,8,12,16,20,40,80",
    )
    parser.add_argument(
        "--bootstrap-trials",
        type=int,
        default=0,
        help="If >0, compute 90%% bootstrap CIs (5%%-95%%) for mean coverage and mean width over test examples.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--x-features",
        default="query_x",
        choices=["query_x", "s_proj"],
        help="Which covariates to feed into conditional conformal. "
        "query_x uses the raw query x at position L; s_proj uses ||(I-P_L)x_query||.",
    )
    parser.add_argument("--phi", default="norm_bins", choices=["intercept", "linear", "norm_bins"])
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument(
        "--diagnostic-bins",
        type=int,
        default=0,
        help="If >0, report split vs condconf coverage/avg_width across quantile bins of the 1D diagnostic feature "
        "(s_proj if --x-features=s_proj; ||x|| if --x-features=query_x). Adds columns to the output CSV.",
    )
    parser.add_argument(
        "--diagnostic-detail-icl-lens",
        default="",
        help="Optional comma-separated ICL lengths for which to export per-test-point binned diagnostics "
        "(bin id, split/cond coverage and widths). Example: --diagnostic-detail-icl-lens 1,5,10,20",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output CSV filename to avoid collisions across parallel runs "
        "(e.g. --output-suffix S01).",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument("--exact", action="store_true", help="Use exact cutoff computation (finite Phi only).")
    parser.add_argument(
        "--kernel",
        default="",
        help="Enable RKHS conditional conformal by specifying a sklearn-compatible kernel "
        "(e.g. rbf, linear, poly, sigmoid). Empty means no RKHS.",
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Kernel gamma (passed to sklearn pairwise_kernels).")
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=1.0,
        help="RKHS regularization lambda (package uses key 'lambda').",
    )
    args = parser.parse_args()

    if not (0.0 < args.alpha < 1.0):
        raise SystemExit("--alpha must be in (0,1)")
    if not (0.0 < args.calib_frac < 1.0):
        raise SystemExit("--calib-frac must be in (0,1)")
    if int(args.icl_step) <= 0:
        raise SystemExit("--icl-step must be a positive integer")
    if int(args.icl_start) < 0:
        raise SystemExit("--icl-start must be >= 0")

    infinite_params = {}
    if args.kernel:
        infinite_params = {"kernel": str(args.kernel), "gamma": float(args.gamma), "lambda": float(args.lam)}
        if args.exact:
            print("[WARN] --exact is not supported with --kernel; disabling exact mode.")
            args.exact = False

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
        if not args.family:
            raise SystemExit("Provide either --run-dir or --family.")
        exp_list = EXPS[args.family]
        if args.exp:
            if args.exp not in exp_list:
                raise SystemExit(f"--exp must be one of: {exp_list}")
            exp_list = [args.exp]

    rows = []
    detail_rows = []
    detail_lens = _parse_int_csv(args.diagnostic_detail_icl_lens)

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

        for task in tqdm(
            tasks,
            desc=f"{exp_name} tasks",
            leave=False,
            disable=bool(args.disable_tqdm),
        ):
            task_name = task["name"]
            task_kwargs = task.get("kwargs", {}) or {}
            task_label = _task_to_label(task_name, task_kwargs)

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

            # Split along examples.
            split = split_indices(int(xs.shape[0]), calib_frac=float(args.calib_frac), seed=int(args.seed))
            cal_idx = torch.from_numpy(split.calib_idx).long()
            test_idx = torch.from_numpy(split.test_idx).long()

            bootstrap_idx = np.empty((0, 0), dtype=np.int64)
            if int(args.bootstrap_trials) > 0:
                # Reuse bootstrap indices across ICL lengths (common random numbers for smoother curves).
                rng = np.random.default_rng(int(args.seed))
                n_test = int(len(split.test_idx))
                bootstrap_idx = rng.integers(0, n_test, size=(int(args.bootstrap_trials), n_test), dtype=np.int64)

            if str(args.icl_lens).strip():
                icl_lens = np.array(sorted(_parse_int_csv(str(args.icl_lens))), dtype=np.int64)
            else:
                icl_lens = np.arange(
                    int(args.icl_start),
                    int(args.max_n_points),
                    int(args.icl_step),
                    dtype=np.int64,
                )
            # By design we predict at position L using L context examples (0..L-1).
            # So L should be in [1, max_n_points-1].
            icl_lens = icl_lens[(icl_lens >= 1) & (icl_lens < int(args.max_n_points))]
            for L in tqdm(
                icl_lens.tolist(),
                desc=f"{exp_name}:{task_name} L",
                leave=False,
                disable=bool(args.disable_tqdm),
            ):
                # Build covariates at ICL length L.
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

                # Split conformal baseline (marginal), for comparison.
                q_split = _conformal_q_abs(np.abs(y_cal - yhat_cal), alpha=float(args.alpha))
                lo_split = yhat_te - q_split
                hi_split = yhat_te + q_split
                covered_split = (y_te >= lo_split) & (y_te <= hi_split)

                phi = _build_phi(args.phi, args.num_bins)
                calib = CondConfSymmetricAbs(phi=phi, seed=int(args.seed), infinite_params=infinite_params)
                calib.fit(x_cal, yhat_cal, y_cal)
                q_te = calib.cutoff(x_te, yhat_te, alpha=float(args.alpha), exact=bool(args.exact))
                lo, hi = calib.intervals(yhat_te, q_te)
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

                row = {
                        "family": args.family,
                        "exp": exp_name,
                        "task": task_label,
                        "icl_len": int(L),
                        "alpha": float(args.alpha),
                        "x_features": str(args.x_features),
                        "phi": str(args.phi),
                        "num_bins": int(args.num_bins),
                        "coverage": float(np.mean(covered)),
                        "coverage_bootstrap_low": float(cov_ci_low),
                        "coverage_bootstrap_high": float(cov_ci_high),
                        "avg_half_width": float(np.mean(q_te)),
                        "avg_width": float(2.0 * np.mean(q_te)),
                        "split_coverage": float(np.mean(covered_split)),
                        "split_avg_width": float(2.0 * q_split),
                        "width_bootstrap_low": float(w_ci_low),
                        "width_bootstrap_high": float(w_ci_high),
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
                        "calib_frac": float(args.calib_frac),
                        "bootstrap_trials": int(args.bootstrap_trials),
                        "device": device,
                        "exact": bool(args.exact),
                        "kernel": str(args.kernel),
                        "gamma": float(args.gamma),
                        "lambda": float(args.lam),
                }

                if int(args.diagnostic_bins) > 0:
                    b = int(args.diagnostic_bins)
                    qs = np.linspace(0.0, 1.0, b + 1)
                    edges = np.quantile(np.asarray(diag_feat_te, dtype=np.float64).reshape(-1), qs).astype(np.float64)
                    # Ensure non-decreasing edges for digitize; quantiles can be equal if feature is degenerate.
                    for i in range(1, edges.size):
                        if edges[i] < edges[i - 1]:
                            edges[i] = edges[i - 1]

                    w_split = np.full_like(w_te, 2.0 * q_split, dtype=np.float64)
                    bin_idx = np.full((diag_feat_te.shape[0],), -1, dtype=np.int64)
                    for i in range(b):
                        lo_e, hi_e = float(edges[i]), float(edges[i + 1])
                        if i == 0:
                            m = (diag_feat_te >= lo_e) & (diag_feat_te <= hi_e)
                        else:
                            m = (diag_feat_te > lo_e) & (diag_feat_te <= hi_e)
                        bin_idx[m] = i
                        n_bin = int(np.sum(m))
                        row[f"diag_bin_n_{i:02d}"] = int(n_bin)
                        if n_bin <= 0:
                            row[f"split_cov_bin_{i:02d}"] = float("nan")
                            row[f"condconf_cov_bin_{i:02d}"] = float("nan")
                            row[f"split_avg_width_bin_{i:02d}"] = float("nan")
                            row[f"condconf_avg_width_bin_{i:02d}"] = float("nan")
                        else:
                            row[f"split_cov_bin_{i:02d}"] = float(np.mean(covered_split[m]))
                            row[f"condconf_cov_bin_{i:02d}"] = float(np.mean(covered[m]))
                            row[f"split_avg_width_bin_{i:02d}"] = float(np.mean(w_split[m]))
                            row[f"condconf_avg_width_bin_{i:02d}"] = float(np.mean(w_te[m]))

                    if (not detail_lens) or (int(L) in detail_lens):
                        for j in range(diag_feat_te.shape[0]):
                            detail_rows.append(
                                {
                                    "family": args.family,
                                    "exp": exp_name,
                                    "task": task_label,
                                    "icl_len": int(L),
                                    "alpha": float(args.alpha),
                                    "x_features": str(args.x_features),
                                    "diagnostic_bins": int(b),
                                    "diag_feature_value": float(diag_feat_te[j]),
                                    "diag_bin": int(bin_idx[j]),
                                    "split_covered": int(bool(covered_split[j])),
                                    "condconf_covered": int(bool(covered[j])),
                                    "split_width": float(w_split[j]),
                                    "condconf_width": float(w_te[j]),
                                    "run_dir": str(rd),
                                    "ckpt": str(ckpt_path),
                                    "num_eval_examples": int(args.num_eval_examples),
                                    "batch_size": int(batch_size),
                                    "seed": int(args.seed),
                                    "calib_frac": float(args.calib_frac),
                                    "bootstrap_trials": int(args.bootstrap_trials),
                                    "device": device,
                                    "exact": bool(args.exact),
                                    "kernel": str(args.kernel),
                                    "gamma": float(args.gamma),
                                    "lambda": float(args.lam),
                                }
                            )

                rows.append(row)

    alpha_tag = str(args.alpha).replace(".", "p")
    if args.family:
        group_base = args.family
    elif explicit_run_dir is not None:
        group_base = explicit_run_dir.parent.name or explicit_run_dir.name
    else:
        group_base = "run"
    if args.output_suffix:
        group_base = f"{group_base}_{args.output_suffix}"
    group_tag = _safe_tag(group_base)
    csv_path = out_dir / f"compare_lr2x_condconf_summary_{group_tag}_alpha{alpha_tag}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved:", csv_path)
    else:
        print("[WARN] no rows produced")

    if detail_rows:
        detail_csv = out_dir / f"compare_lr2x_condconf_binned_details_{group_tag}_alpha{alpha_tag}.csv"
        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
        print("Saved:", detail_csv)


if __name__ == "__main__":
    main()
