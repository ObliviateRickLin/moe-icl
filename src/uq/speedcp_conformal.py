"""
SpeedCP-based conditional conformal for symmetric intervals on regression residual magnitude.

This module wraps the SpeedCP implementation (Gibbs et al., 2023/2024 style S-path / lambda-path)
to produce per-test-point cutoffs q(x) for symmetric prediction intervals:
  [yhat(x) - q(x), yhat(x) + q(x)].

Key points:
  - We conformalize the absolute residual r = |y - yhat|.
  - The conditional cutoff function is modeled as:
        g(x) = Phi(x)^T eta + (K(x, X_cal) v) / lambda
    where (v, eta) are learned on calibration data for a *fixed* (gamma, lambda).
  - SpeedCP stage-2 (S_path) then computes S_opt for each test point, which we use as q(x).

This is intended to be a drop-in alternative to uq.conditional_conformal.CondConfSymmetricAbs
when the conditionalconformal kernel path is too slow.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np


def _import_speedcp():
    try:
        from speedcp.lambda_trace import lambda_path  # type: ignore
        from speedcp.S_trace import S_path  # type: ignore
        return lambda_path, S_path
    except Exception:
        # Fallback: repo-local checkout at <workspace>/_ext_speedcp2
        root = Path(__file__).resolve().parents[3]
        ext = root / "_ext_speedcp2"
        if ext.exists():
            sys.path.insert(0, str(ext))
            from speedcp.lambda_trace import lambda_path  # type: ignore
            from speedcp.S_trace import S_path  # type: ignore
            return lambda_path, S_path
        raise ImportError(
            "SpeedCP not importable. Install a `speedcp` package or clone the repo to `../_ext_speedcp2` "
            f"(looked for: {ext})."
        )


def _kernel_gram(X: np.ndarray, Y: Optional[np.ndarray], *, kernel: str, gamma: float) -> np.ndarray:
    from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel

    X = np.asarray(X, dtype=np.float64)
    Y_ = None if Y is None else np.asarray(Y, dtype=np.float64)
    if kernel == "rbf":
        return rbf_kernel(X, Y_, gamma=float(gamma))
    if kernel == "laplacian":
        return laplacian_kernel(X, Y_, gamma=float(gamma))
    raise ValueError(f"unsupported kernel: {kernel} (expected rbf or laplacian)")


def _kernel_self(*, kernel: str) -> float:
    # For rbf and laplacian, k(x,x)=1.
    if kernel in ("rbf", "laplacian"):
        return 1.0
    raise ValueError(f"unsupported kernel: {kernel}")


def _as_np_unique_sorted(a) -> np.ndarray:
    a = np.asarray(a, dtype=int).ravel()
    if a.size == 0:
        return a
    return np.unique(a)


def _solve_v_eta_given_sets(
    *,
    S_vec: np.ndarray,
    Phi: np.ndarray,
    K: np.ndarray,
    alpha: float,
    lam: float,
    indE: np.ndarray,
    indL: np.ndarray,
    indR: np.ndarray,
    ridge: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for (v, eta) at a fixed lambda, given active sets E/L/R.

    This matches SpeedCP's `lambda_trace.lambda_path` Step-4 update.
    """
    from cvxopt import matrix, solvers  # type: ignore

    S_vec = np.asarray(S_vec, dtype=np.float64).ravel()
    Phi = np.asarray(Phi, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    n = int(S_vec.size)
    d = int(Phi.shape[1])

    indE = _as_np_unique_sorted(indE)
    indL = _as_np_unique_sorted(indL)
    indR = _as_np_unique_sorted(indR)

    v = np.zeros(n, dtype=np.float64)
    if indL.size:
        v[indL] = -float(alpha)
    if indR.size:
        v[indR] = float(1.0 - alpha)

    E = indE.copy()
    m = int(E.size)
    if m <= 0:
        # Degenerate: no elbow. Fall back to pure-bounds v, and zero eta.
        return v, np.zeros((d,), dtype=np.float64)

    KEE = K[np.ix_(E, E)]
    KEL = K[np.ix_(E, indL)] if indL.size else np.zeros((m, 0), dtype=np.float64)
    KER = K[np.ix_(E, indR)] if indR.size else np.zeros((m, 0), dtype=np.float64)
    PhiE = Phi[E, :]

    active_cols = np.where(np.any(PhiE != 0.0, axis=0))[0]
    PhiE_act = PhiE[:, active_cols]
    d_act = int(active_cols.size)

    PhiL_act = Phi[np.ix_(indL, active_cols)] if indL.size else np.zeros((0, d_act), dtype=np.float64)
    PhiR_act = Phi[np.ix_(indR, active_cols)] if indR.size else np.zeros((0, d_act), dtype=np.float64)
    one_L = np.ones(int(indL.size), dtype=np.float64)
    one_R = np.ones(int(indR.size), dtype=np.float64)

    S_E = S_vec[E]
    if indL.size or indR.size:
        S_E_eff = S_E - (-float(alpha) * (KEL @ one_L) + float(1.0 - alpha) * (KER @ one_R)) / float(lam)
        b_bot = float(alpha) * (PhiL_act.T @ one_L) - float(1.0 - alpha) * (PhiR_act.T @ one_R)
    else:
        S_E_eff = S_E
        b_bot = np.zeros((d_act,), dtype=np.float64)

    top = np.hstack([PhiE_act, KEE / float(lam)])
    bot = np.hstack([np.zeros((d_act, d_act), dtype=np.float64), PhiE_act.T])
    H = np.vstack([top, bot])
    b = np.concatenate([S_E_eff, b_bot])

    P_np = H.T @ H + float(ridge) * np.eye(d_act + m, dtype=np.float64)
    q_np = -(H.T @ b)

    # Box constraints on v_E only.
    Zdm = np.zeros((m, d_act), dtype=np.float64)
    G_np = np.vstack(
        [
            np.hstack([Zdm, np.eye(m, dtype=np.float64)]),
            np.hstack([Zdm, -np.eye(m, dtype=np.float64)]),
        ]
    )
    h_np = np.hstack(
        [
            float(1.0 - alpha) * np.ones(m, dtype=np.float64),
            float(alpha) * np.ones(m, dtype=np.float64),
        ]
    )

    solvers.options["show_progress"] = False
    P = matrix(P_np, tc="d")
    q = matrix(q_np, tc="d")
    G = matrix(G_np, tc="d")
    h = matrix(h_np, tc="d")

    try:
        sol = solvers.qp(P, q, G, h)
        if sol["status"] != "optimal":
            z = np.linalg.lstsq(H, b, rcond=1e-12)[0]
        else:
            z = np.array(sol["x"], dtype=np.float64).reshape(-1)
    except Exception:
        z = np.linalg.lstsq(H, b, rcond=1e-12)[0]

    eta = np.zeros(d, dtype=np.float64)
    if d_act > 0:
        eta[active_cols] = z[:d_act]
    vE = z[d_act:]
    if vE.size != m:
        raise RuntimeError("unexpected vE size from QP solve")
    v[E] = np.clip(vE, -float(alpha), float(1.0 - alpha))

    # Numerical cleanup: snap near-bound values so set detection is stable.
    v[np.abs(v + float(alpha)) <= float(tol)] = -float(alpha)
    v[np.abs(v - float(1.0 - alpha)) <= float(tol)] = float(1.0 - alpha)
    return v, eta


@dataclass
class SpeedCPFitInfo:
    alpha: float
    gamma: float
    lam: float
    tune_seconds: float
    stage2_seconds: float


_WORKER_STATE = {}


def _speedcp_worker_init(state: dict) -> None:
    # Avoid extra BLAS oversubscription inside each worker.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)
    if _WORKER_STATE.get("S_path") is None:
        # Ensure SpeedCP is importable in the worker process.
        _lambda_path, S_path = _import_speedcp()
        _WORKER_STATE["S_path"] = S_path


def _speedcp_worker_solve(idxs: list[int]) -> list[tuple[int, float]]:
    _state = _WORKER_STATE
    S_path = _state["S_path"]
    S_cal = _state["S_cal"]
    Phi_cal = _state["Phi_cal"]
    Phi_test = _state["Phi_test"]
    K11 = _state["K11"]
    K12 = _state["K12"]
    k22 = float(_state["k22"])
    lam = float(_state["lam"])
    alpha = float(_state["alpha"])
    alpha0_const = float(_state["alpha0"])
    alpha0s = _state.get("alpha0s", None)
    best_v = _state["best_v"]
    best_eta = _state["best_eta"]
    start_side = _state["start_side"]
    max_steps = int(_state["max_steps"])
    eps = float(_state["eps"])
    tol = float(_state["tol"])
    ridge = float(_state["ridge"])

    n_cal = int(S_cal.shape[0])
    p = int(Phi_cal.shape[1])

    K_all = np.empty((n_cal + 1, n_cal + 1), dtype=np.float64)
    K_all[:-1, :-1] = K11
    Phi_all = np.empty((n_cal + 1, p), dtype=np.float64)
    Phi_all[:-1, :] = Phi_cal

    out = []
    for i in idxs:
        alpha0 = float(alpha0s[i]) if alpha0s is not None else alpha0_const
        k12 = K12[i, :]
        K_all[:-1, -1] = k12
        K_all[-1, :-1] = k12
        K_all[-1, -1] = k22
        Phi_all[-1, :] = Phi_test[i, :]

        res_S = S_path(
            S_cal,
            Phi_all,
            K_all,
            lam,
            alpha,
            alpha0=alpha0,
            best_v=best_v,
            best_eta=best_eta,
            start_side=start_side,
            max_steps=max_steps,
            eps=eps,
            tol=tol,
            ridge=ridge,
            verbose=False,
        )
        out.append((int(i), float(res_S["S_opt"])))
    return out


class SpeedCPSymmetricAbs:
    """
    SpeedCP conditional conformal for symmetric intervals via residual magnitudes.

    This wrapper assumes a *fixed* kernel hyperparameter gamma and RKHS lambda.
    It computes the calibration (v, eta) at the fixed lambda using lambda_path
    plus an exact set-conditioned solve.
    """

    def __init__(
        self,
        *,
        phi,
        kernel: Literal["rbf", "laplacian"] = "rbf",
        gamma: float = 1.0,
        lam: float = 1.0,
        seed: int = 0,
        # Stage-1 (lambda path) controls.
        lambda_max_steps: int = 500,
        lambda_tol: float = 1e-6,
        lambda_ridge: float = 1e-8,
        # Stage-2 (S path) controls.
        start_side: Literal["left", "right"] = "left",
        s_max_steps: int = 200,
        s_eps: float = 1e-3,
        s_tol: float = 1e-6,
        s_ridge: float = 1e-8,
        # Parallelism.
        worker_backend: Literal["thread", "process"] = "process",
        verbose: bool = False,
    ):
        self.phi = phi
        self.kernel = str(kernel)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.seed = int(seed)
        self.lambda_max_steps = int(lambda_max_steps)
        self.lambda_tol = float(lambda_tol)
        self.lambda_ridge = float(lambda_ridge)
        self.start_side = str(start_side)
        self.s_max_steps = int(s_max_steps)
        self.s_eps = float(s_eps)
        self.s_tol = float(s_tol)
        self.s_ridge = float(s_ridge)
        self.worker_backend = str(worker_backend)
        self.verbose = bool(verbose)

        self._x_cal: Optional[np.ndarray] = None
        self._r_cal: Optional[np.ndarray] = None
        self._Phi_cal: Optional[np.ndarray] = None
        self._K11: Optional[np.ndarray] = None
        self._stage1_alpha: Optional[float] = None
        self._best_v: Optional[np.ndarray] = None
        self._best_eta: Optional[np.ndarray] = None

        self.last_fit_info: Optional[SpeedCPFitInfo] = None

    def fit(self, x_features_calib: np.ndarray, yhat_calib: np.ndarray, y_calib: np.ndarray) -> "SpeedCPSymmetricAbs":
        x_features_calib = np.asarray(x_features_calib, dtype=np.float64)
        yhat_calib = np.asarray(yhat_calib, dtype=np.float64).reshape(-1)
        y_calib = np.asarray(y_calib, dtype=np.float64).reshape(-1)
        if x_features_calib.ndim != 2:
            raise ValueError("x_features_calib must be 2D")
        if yhat_calib.shape[0] != x_features_calib.shape[0] or y_calib.shape[0] != x_features_calib.shape[0]:
            raise ValueError("calibration x/y/yhat sizes must match")

        r_cal = np.abs(y_calib - yhat_calib).astype(np.float64)

        self.phi.fit(x_features_calib)
        Phi_cal = np.asarray(self.phi(x_features_calib), dtype=np.float64)

        K11 = _kernel_gram(x_features_calib, None, kernel=self.kernel, gamma=self.gamma)
        K11 = K11 + 1e-8 * np.eye(K11.shape[0], dtype=np.float64)

        self._x_cal = x_features_calib
        self._r_cal = r_cal
        self._Phi_cal = Phi_cal
        self._K11 = K11

        # Invalidate cached stage-1 solution (depends on alpha).
        self._stage1_alpha = None
        self._best_v = None
        self._best_eta = None
        self.last_fit_info = None
        return self

    def _ensure_stage1(self, *, alpha: float) -> None:
        if self._x_cal is None or self._r_cal is None or self._Phi_cal is None or self._K11 is None:
            raise RuntimeError("Call fit(...) before cutoff(...).")
        if self._stage1_alpha is not None and abs(self._stage1_alpha - float(alpha)) < 1e-12:
            return

        lambda_path, _S_path = _import_speedcp()

        t0 = time.time()
        # Ensure the lambda path runs far enough to cover lam (common values are <= 10).
        thres = max(10.0, float(self.lam) * 1.5)
        res = lambda_path(
            self._r_cal.ravel(),
            self._Phi_cal,
            self._K11,
            float(alpha),
            max_steps=int(self.lambda_max_steps),
            tol=float(self.lambda_tol),
            thres=float(thres),
            ridge=float(self.lambda_ridge),
            verbose=False,
        )
        lambdas = np.asarray(res["lambdas"], dtype=np.float64).ravel()
        if lambdas.size <= 0:
            raise RuntimeError("lambda_path returned empty path")

        # Pick the last step at or below the target lambda, assuming lambdas are monotone increasing.
        # If not monotone, we still get a reasonable base set via nearest.
        diffs = np.diff(lambdas)
        inc = bool(np.sum(diffs >= 0) >= np.sum(diffs < 0))
        if inc:
            base = int(np.searchsorted(lambdas, float(self.lam), side="right") - 1)
            base = max(0, min(base, int(lambdas.size - 1)))
        else:
            # monotone decreasing fallback
            base = int(np.searchsorted(-lambdas, -float(self.lam), side="right") - 1)
            base = max(0, min(base, int(lambdas.size - 1)))

        v_base = np.asarray(res["v_arr"][base, :], dtype=np.float64).ravel()
        eta_base = np.asarray(res["eta_arr"][base, :], dtype=np.float64).ravel()
        indE = np.asarray(res["Elbows"][base], dtype=int).ravel()

        # Detect L/R using a stable tolerance.
        at_lo = np.abs(v_base + float(alpha)) <= float(self.lambda_tol)
        at_hi = np.abs(v_base - float(1.0 - alpha)) <= float(self.lambda_tol)
        indL = np.where(at_lo)[0].astype(int)
        indR = np.where(at_hi)[0].astype(int)

        # Exact solve at the fixed lambda with the current active set.
        v_star, eta_star = _solve_v_eta_given_sets(
            S_vec=self._r_cal,
            Phi=self._Phi_cal,
            K=self._K11,
            alpha=float(alpha),
            lam=float(self.lam),
            indE=indE,
            indL=indL,
            indR=indR,
            ridge=float(self.lambda_ridge),
            tol=float(self.lambda_tol),
        )
        tune_seconds = float(time.time() - t0)

        # Keep eta_base only for debugging; eta_star is what we use.
        _ = eta_base  # unused

        self._stage1_alpha = float(alpha)
        self._best_v = v_star
        self._best_eta = eta_star
        self.last_fit_info = SpeedCPFitInfo(
            alpha=float(alpha),
            gamma=float(self.gamma),
            lam=float(self.lam),
            tune_seconds=tune_seconds,
            stage2_seconds=0.0,
        )

    def cutoff(
        self,
        x_features_test: np.ndarray,
        yhat_test: np.ndarray,
        *,
        alpha: float,
        randomize: bool = False,
        workers: int = 1,
    ) -> np.ndarray:
        """
        Returns q(x) (half-widths) for each test point.
        """
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0,1)")
        self._ensure_stage1(alpha=float(alpha))
        assert self._x_cal is not None and self._r_cal is not None and self._Phi_cal is not None and self._K11 is not None
        assert self._best_v is not None and self._best_eta is not None

        x_features_test = np.asarray(x_features_test, dtype=np.float64)
        _ = np.asarray(yhat_test, dtype=np.float64).reshape(-1)  # used only for shape checks
        if x_features_test.ndim != 2:
            raise ValueError("x_features_test must be 2D")
        if x_features_test.shape[0] != _.shape[0]:
            raise ValueError("x_features_test and yhat_test must have matching length")

        lambda_path, S_path = _import_speedcp()
        _ = lambda_path  # used in _ensure_stage1; keep import here for worker init.

        Phi_test = np.asarray(self.phi(x_features_test), dtype=np.float64)
        K12 = _kernel_gram(x_features_test, self._x_cal, kernel=self.kernel, gamma=self.gamma)
        k22 = _kernel_self(kernel=self.kernel)

        n_test = int(x_features_test.shape[0])
        out = np.empty((n_test,), dtype=np.float64)

        alpha0 = 1.0 - float(alpha)
        alpha0s: Optional[np.ndarray] = None
        if randomize:
            rng = np.random.default_rng(int(self.seed))
            alpha0s = rng.uniform(-float(alpha), 1.0 - float(alpha), size=(n_test,)).astype(np.float64)

        t0 = time.time()

        n_workers = int(max(1, workers))
        if n_workers <= 1 or n_test <= 1:
            # Fast sequential path (reuse one buffer).
            n_cal = int(self._r_cal.shape[0])
            p = int(self._Phi_cal.shape[1])
            K_all = np.empty((n_cal + 1, n_cal + 1), dtype=np.float64)
            K_all[:-1, :-1] = self._K11
            Phi_all = np.empty((n_cal + 1, p), dtype=np.float64)
            Phi_all[:-1, :] = self._Phi_cal
            for i in range(n_test):
                alpha0_i = float(alpha0s[i]) if alpha0s is not None else float(alpha0)
                k12 = K12[i, :]
                K_all[:-1, -1] = k12
                K_all[-1, :-1] = k12
                K_all[-1, -1] = k22
                Phi_all[-1, :] = Phi_test[i, :]
                res_S = S_path(
                    self._r_cal,
                    Phi_all,
                    K_all,
                    float(self.lam),
                    float(alpha),
                    alpha0=alpha0_i,
                    best_v=self._best_v,
                    best_eta=self._best_eta,
                    start_side=self.start_side,
                    max_steps=int(self.s_max_steps),
                    eps=float(self.s_eps),
                    tol=float(self.s_tol),
                    ridge=float(self.s_ridge),
                    verbose=False,
                )
                out[i] = float(res_S["S_opt"])
        else:
            idxs = np.arange(n_test, dtype=int)
            chunks = np.array_split(idxs, n_workers)
            # Ensure chunks are python lists so ProcessPool pickles less numpy metadata.
            chunk_lists = [c.astype(int).tolist() for c in chunks if c.size]
            if self.worker_backend == "thread":
                state = {
                    "S_path": S_path,
                    "S_cal": self._r_cal,
                    "Phi_cal": self._Phi_cal,
                    "Phi_test": Phi_test,
                    "K11": self._K11,
                    "K12": K12,
                    "k22": k22,
                    "lam": float(self.lam),
                    "alpha": float(alpha),
                    "alpha0": float(alpha0),
                    "alpha0s": alpha0s,
                    "best_v": self._best_v,
                    "best_eta": self._best_eta,
                    "start_side": self.start_side,
                    "max_steps": int(self.s_max_steps),
                    "eps": float(self.s_eps),
                    "tol": float(self.s_tol),
                    "ridge": float(self.s_ridge),
                }
                # Thread backend: set global state once in current process.
                _speedcp_worker_init(state)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    for pairs in ex.map(_speedcp_worker_solve, chunk_lists):
                        for i, q in pairs:
                            out[i] = float(q)
            else:
                state = {
                    # For process backend we set S_path=None and let the worker import SpeedCP.
                    "S_path": None,
                    "S_cal": self._r_cal,
                    "Phi_cal": self._Phi_cal,
                    "Phi_test": Phi_test,
                    "K11": self._K11,
                    "K12": K12,
                    "k22": k22,
                    "lam": float(self.lam),
                    "alpha": float(alpha),
                    "alpha0": float(alpha0),
                    "alpha0s": alpha0s,
                    "best_v": self._best_v,
                    "best_eta": self._best_eta,
                    "start_side": self.start_side,
                    "max_steps": int(self.s_max_steps),
                    "eps": float(self.s_eps),
                    "tol": float(self.s_tol),
                    "ridge": float(self.s_ridge),
                }
                # For process backend, pass a shallow state dict; each worker imports SpeedCP itself.
                with ProcessPoolExecutor(
                    max_workers=n_workers, initializer=_speedcp_worker_init, initargs=(state,)
                ) as ex:
                    for pairs in ex.map(_speedcp_worker_solve, chunk_lists):
                        for i, q in pairs:
                            out[i] = float(q)

        stage2_seconds = float(time.time() - t0)
        if self.last_fit_info is not None:
            self.last_fit_info = SpeedCPFitInfo(
                alpha=float(self.last_fit_info.alpha),
                gamma=float(self.last_fit_info.gamma),
                lam=float(self.last_fit_info.lam),
                tune_seconds=float(self.last_fit_info.tune_seconds),
                stage2_seconds=stage2_seconds,
            )

        return np.maximum(out, 0.0)
