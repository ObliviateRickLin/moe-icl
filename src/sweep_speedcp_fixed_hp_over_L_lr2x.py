"""
Sweep ICL example count (query position) L for SpeedCP conditional conformal with *fixed* (gamma, lambda).

This keeps the number of evaluation sequences fixed (e.g. N=1600 total, split into
800 calib / 800 test), and varies the number of in-context examples by choosing
different query positions L within a single sampled sequence of length `max_n_points`.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ckpt_utils import select_ckpt_path
from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler
from uq.conditional_conformal import InterceptPhi, split_indices
from uq.speedcp_conformal import SpeedCPSymmetricAbs


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _build_conf(model_cfg: dict):
    class C:
        pass

    c = C()
    for k, v in model_cfg.items():
        setattr(c, k, v)
    c.keys = lambda: c.__dict__.keys()
    return c


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--family", default="gpt")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--prefer-model-step", type=int, default=500000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--num-eval-examples", type=int, default=1600)
    p.add_argument("--calib-frac", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-n-points", type=int, default=81)
    p.add_argument("--icl-start", type=int, default=4)
    p.add_argument("--icl-step", type=int, default=4)

    p.add_argument("--kernel", choices=["rbf", "laplacian"], default="rbf")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--lambda", dest="lam", type=float, required=True)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-backend", choices=["process", "thread"], default="process")

    p.add_argument("--lambda-max-steps", type=int, default=200)
    p.add_argument("--lambda-tol", type=float, default=1e-6)
    p.add_argument("--lambda-ridge", type=float, default=1e-8)
    p.add_argument("--s-max-steps", type=int, default=200)
    p.add_argument("--s-eps", type=float, default=1e-3)
    p.add_argument("--s-tol", type=float, default=1e-6)
    p.add_argument("--s-ridge", type=float, default=1e-8)
    args = p.parse_args()

    # Limit BLAS threads only if the user will run many SpeedCP solves in parallel.
    # On CPU-only runs, leaving model inference multithreaded can matter a lot.
    if int(args.workers) > 1:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(k, "1")

    N = int(args.num_eval_examples)
    if N <= 0 or N % int(args.batch_size) != 0:
        raise SystemExit("--num-eval-examples must be positive and divisible by --batch-size")

    _seed_everything(int(args.seed))

    conf_path = Path(__file__).resolve().parent / "conf" / str(args.family) / f"{args.exp}.yaml"
    if not conf_path.exists():
        raise SystemExit(f"config not found: {conf_path}")
    cfg = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    tasks_cfg = train_cfg.get("tasks", [])
    if len(tasks_cfg) != 1:
        raise SystemExit("expected exactly 1 task in config")
    task_name = tasks_cfg[0]["name"]
    task_kwargs = tasks_cfg[0].get("kwargs", {}) or {}
    data_name = cfg.get("data") or cfg.get("data_name") or "gaussian"

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    ckpt_path, _ = select_ckpt_path(run_dir, prefer_step=int(args.prefer_model_step))
    if ckpt_path is None:
        raise SystemExit(f"no checkpoint found under: {run_dir}")

    st = torch.load(ckpt_path, map_location="cpu")
    model = build_model(_build_conf(model_cfg))
    model.load_state_dict(st["model_state_dict"] if isinstance(st, dict) and "model_state_dict" in st else st)

    n_dims = int(model_cfg["n_dims"])
    n_points = int(args.max_n_points)

    # Generate one batch of sequences (length n_points) for all L.
    t0 = time.time()
    xs, ys, yhat = collect_xyhat(
        model,
        task_name=str(task_name),
        task_kwargs=dict(task_kwargs),
        data_name=str(data_name),
        n_dims=n_dims,
        n_points=n_points,
        num_eval_examples=N,
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    gen_seconds = float(time.time() - t0)

    split = split_indices(N, calib_frac=float(args.calib_frac), seed=int(args.seed))
    calib_idx, test_idx = split.calib_idx, split.test_idx

    Ls = list(range(int(args.icl_start), int(args.max_n_points), int(args.icl_step)))
    Ls = [L for L in Ls if 1 <= L <= (n_points - 1)]
    if not Ls:
        raise SystemExit("empty L sweep; check --icl-start/--icl-step/--max-n-points")

    print(
        f"exp={args.exp} ckpt={ckpt_path} N={N} (n_cal={len(calib_idx)}, n_test={len(test_idx)}) "
        f"points={n_points} d={n_dims} alpha={float(args.alpha)} randomize={bool(args.randomize)} seed={int(args.seed)}"
    )
    print(f"fixed hp: kernel={args.kernel} gamma={float(args.gamma)} lambda={float(args.lam)}")
    print(f"generated once in {gen_seconds:.2f}s; sweeping L={Ls[0]}..{Ls[-1]} step={int(args.icl_step)}")
    print("")
    print("L\tn_cal\tn_test\tcoverage\tavg_width\tt_tune\tt_stage2\tt_total")

    for L in Ls:
        xq = xs[:, L, :].numpy()
        yq = ys[:, L].numpy()
        yhatq = yhat[:, L].numpy()

        x_cal = xq[calib_idx]
        y_cal = yq[calib_idx]
        yhat_cal = yhatq[calib_idx]
        x_test = xq[test_idx]
        y_test = yq[test_idx]
        yhat_test = yhatq[test_idx]

        phi = InterceptPhi()
        cp = SpeedCPSymmetricAbs(
            phi=phi,
            kernel=str(args.kernel),
            gamma=float(args.gamma),
            lam=float(args.lam),
            seed=int(args.seed),
            lambda_max_steps=int(args.lambda_max_steps),
            lambda_tol=float(args.lambda_tol),
            lambda_ridge=float(args.lambda_ridge),
            start_side="left",
            s_max_steps=int(args.s_max_steps),
            s_eps=float(args.s_eps),
            s_tol=float(args.s_tol),
            s_ridge=float(args.s_ridge),
            worker_backend=str(args.worker_backend),
            verbose=False,
        )

        t1 = time.time()
        cp.fit(x_cal, yhat_cal, y_cal)
        q = cp.cutoff(x_test, yhat_test, alpha=float(args.alpha), randomize=bool(args.randomize), workers=int(args.workers))
        total_s = float(time.time() - t1 + gen_seconds)

        covered = (np.abs(y_test - yhat_test) <= q).astype(np.float64)
        cov = float(covered.mean())
        width = float((2.0 * q).mean())
        tune_s = float(cp.last_fit_info.tune_seconds) if cp.last_fit_info is not None else float("nan")
        stage2_s = float(cp.last_fit_info.stage2_seconds) if cp.last_fit_info is not None else float("nan")
        print(
            f"{int(L)}\t{int(x_cal.shape[0])}\t{int(x_test.shape[0])}\t"
            f"{cov:.4f}\t{width:.4f}\t{tune_s:.2f}\t{stage2_s:.2f}\t{total_s:.2f}"
        )


if __name__ == "__main__":
    main()
