from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import weakref


try:
    # Paper repo / PyPI package: jjcherian/conditional-conformal
    from conditionalconformal.condconf import CondConf as _CondConf
except Exception as e:  # pragma: no cover
    _CondConf = None
    _IMPORT_ERROR = e


def _require_conditionalconformal():
    if _CondConf is None:  # pragma: no cover
        raise ImportError(
            "conditionalconformal is required for conditional conformal UQ. "
            "Install with `pip install conditionalconformal cvxpy`."
        ) from _IMPORT_ERROR
    _patch_conditionalconformal_kernel_cache()


def _patch_conditionalconformal_solver(
    *,
    rkhs_solver: str = "auto",
    binary_search_tol: float = 1e-3,
):
    """
    Patch upstream condconf helper functions so we can control RKHS backend solver
    and binary-search tolerance from our CLI.
    """
    try:
        import conditionalconformal.condconf as cc  # type: ignore
        from scipy.optimize import linprog
    except Exception:  # pragma: no cover
        return

    cc._moe_rkhs_solver = str(rkhs_solver).lower()
    cc._moe_binary_search_tol = float(binary_search_tol)

    if not hasattr(cc, "_moe_orig_installed_solvers"):
        cc._moe_orig_installed_solvers = cc.cp.installed_solvers

    def _installed_solvers_moe():
        sols = list(cc._moe_orig_installed_solvers())
        solver_name = str(getattr(cc, "_moe_rkhs_solver", "auto")).lower()
        # When user explicitly selects non-auto solver, avoid MOSEK branch in upstream.
        if solver_name != "auto":
            sols = [s for s in sols if str(s).upper() != "MOSEK"]
        return sols

    cc.cp.installed_solvers = _installed_solvers_moe

    if not getattr(cc, "_moe_solver_patch_installed", False):
        import cvxpy as cp

        def _binary_search_moe(func, min_val, max_val, tol=1e-3):
            min_val, max_val = float(min_val), float(max_val)
            tol = float(getattr(cc, "_moe_binary_search_tol", tol))
            assert (max_val + tol) > max_val
            while (max_val - min_val) > tol:
                mid = (min_val + max_val) / 2.0
                if func(mid) > 0:
                    max_val = mid
                else:
                    min_val = mid
            return min_val, max_val

        # Cache MOSEK thread-limit params once (set threads=1 since problem is
        # too small for multi-threading to help, and it eliminates thread
        # explosion in parallel scenarios).
        # Use string key to avoid MOSEK enum deprecation warnings.
        _mosek_solve_params: dict = {"MSK_IPAR_NUM_THREADS": 1}

        def _solve_with_mosek(prob):
            """Solve a CVXPY problem with the best available solver."""
            solver_name = str(getattr(cc, "_moe_rkhs_solver", "auto")).lower()
            try:
                if "MOSEK" in cc.cp.installed_solvers():
                    prob.solve(solver="MOSEK", mosek_params=_mosek_solve_params)
                elif solver_name == "osqp" or solver_name == "auto":
                    prob.solve(solver="OSQP", warm_start=True, verbose=False)
                elif solver_name == "scs":
                    prob.solve(solver="SCS", warm_start=True, verbose=False)
                elif solver_name == "clarabel":
                    prob.solve(solver="CLARABEL", warm_start=True, verbose=False)
                else:
                    prob.solve(solver="OSQP", warm_start=True, verbose=False)
            except Exception:
                if "OSQP" in cc.cp.installed_solvers():
                    prob.solve(solver="OSQP", warm_start=True, verbose=False)
                elif "SCS" in cc.cp.installed_solvers():
                    prob.solve(solver="SCS", warm_start=True, verbose=False)
                else:
                    prob.solve()

        def _solve_dual_moe(S, gcc, x_test, quantiles, threshold=None):
            if gcc.infinite_params.get("kernel", None):
                x_test = np.asarray(x_test, dtype=float).reshape(1, -1)

                kernel = gcc.infinite_params.get("kernel", cc.FUNCTION_DEFAULTS["kernel"])
                gamma = float(gcc.infinite_params.get("gamma", cc.FUNCTION_DEFAULTS["gamma"]))
                radius = 1.0 / float(gcc.infinite_params.get("lambda", cc.FUNCTION_DEFAULTS["lambda"]))
                quantile_val = float(quantiles[-1][0])

                # Heavy test-point setup is identical across binary-search iterations for
                # a fixed test point. Cache it and only refresh scalar params above.
                test_setup_cache = getattr(gcc, "_moe_test_setup_cache", None)
                if test_setup_cache is None:
                    test_setup_cache = {}
                    setattr(gcc, "_moe_test_setup_cache", test_setup_cache)

                key = np.ascontiguousarray(x_test).view(np.uint8).tobytes()
                cached = test_setup_cache.get(key)
                if cached is None:
                    precomputed = getattr(gcc, "_moe_precomputed_k12", None)
                    if isinstance(precomputed, dict) and key in precomputed:
                        k12_col, k22_scalar = precomputed[key]
                    else:
                        K_12 = cc.pairwise_kernels(
                            X=np.concatenate([gcc.x_calib, x_test], axis=0),
                            Y=x_test,
                            metric=kernel,
                            gamma=gamma,
                        )
                        k12_col = K_12[:-1]
                        k22_scalar = float(K_12[-1, 0])

                    _, L_11 = cc._get_kernel_matrix(gcc.x_calib, kernel, gamma)
                    L_21 = np.linalg.solve(L_11, k12_col).T
                    L_22 = np.asarray([[k22_scalar]], dtype=float) - L_21 @ L_21.T
                    L_22[L_22 < 0] = 0.0
                    L_22 = np.sqrt(L_22)
                    l_21_22 = np.hstack([L_21, L_22])
                    cached = (k12_col, l_21_22)
                    test_setup_cache[key] = cached

                k12_col, l_21_22 = cached

                # --- DPP-compatible problem: only S varies within binary search ---
                # For each test point, L_full and Phi_full are fixed (Constants),
                # and only S_test changes (Parameter). This makes the problem DPP,
                # reducing CVXPY canonicalization overhead from ~189ms to ~64ms per solve.
                dpp_cache = getattr(gcc, "_moe_dpp_cache", None)
                if dpp_cache is None or dpp_cache[0] != key:
                    n_calib = len(gcc.scores_calib)
                    n = n_calib + 1
                    phi_test = gcc.Phi_fn(x_test).reshape(1, -1)

                    _, L_11 = cc._get_kernel_matrix(gcc.x_calib, kernel, gamma)
                    L_full_np = np.vstack([
                        np.hstack([L_11, np.zeros((L_11.shape[0], 1))]),
                        l_21_22,
                    ])
                    Phi_full_np = np.vstack([gcc.phi_calib, phi_test])

                    eta = cp.Variable(name="weights", shape=n)
                    S_param = cp.Parameter(name="S_dpp")
                    scores_base = cp.Constant(gcc.scores_calib.reshape(-1, 1))
                    S_vec = cp.vstack([scores_base, cp.reshape(S_param, (1, 1))])

                    L_c = cp.Constant(L_full_np)
                    Phi_c = cp.Constant(Phi_full_np)
                    C_val = float(radius) / n

                    constraints = [
                        (quantile_val - 1) <= eta,
                        quantile_val >= eta,
                        eta.T @ Phi_c == 0,
                    ]
                    dpp_prob = cp.Problem(
                        cp.Minimize(
                            0.5 * C_val * cp.sum_squares(L_c.T @ eta)
                            - cp.sum(cp.multiply(eta, cp.vec(S_vec)))
                        ),
                        constraints,
                    )
                    gcc._moe_dpp_cache = (key, dpp_prob)

                _, prob = gcc._moe_dpp_cache
                prob.param_dict["S_dpp"].value = float(S)
                _solve_with_mosek(prob)
                weights = prob.var_dict["weights"].value
            else:
                S = np.concatenate([gcc.scores_calib, [S]], dtype=float)
                Phi = np.concatenate([gcc.phi_calib, gcc.Phi_fn(x_test)], axis=0, dtype=float)
                zeros = np.zeros((Phi.shape[1],))
                bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
                res = linprog(
                    -1 * S,
                    A_eq=Phi.T,
                    b_eq=zeros,
                    bounds=bounds,
                    method="highs",
                    options={"presolve": False},
                )
                weights = res.x

            if threshold is None:
                if quantiles[-1] < 0.5:
                    threshold = quantiles[-1] - 1
                else:
                    threshold = quantiles[-1]
            return weights[-1] - threshold

        # --- Also patch _get_primal_solution on the CondConf class ---
        # After binary search, predict() calls _get_threshold → _get_primal_solution
        # which uses the original library's Parameter-based CVXPY problem. That
        # formulation sporadically triggers DCPError because CVXPY cannot always
        # verify `sum_squares(Parameter_matrix.T @ Variable)` as DCP when the
        # problem is non-DPP (solving chain rebuilt every call).
        # Fix: reconstruct a concrete problem with all Constants (no Parameters).
        _CondConf_cls = cc.CondConf

        def _patched_get_primal_solution(self, S, x, quantiles):
            if self.infinite_params.get("kernel", cc.FUNCTION_DEFAULTS.get("kernel", None)):
                x_test = np.asarray(x, dtype=float).reshape(1, -1)
                kernel = self.infinite_params.get("kernel", cc.FUNCTION_DEFAULTS["kernel"])
                gamma_val = float(self.infinite_params.get("gamma", cc.FUNCTION_DEFAULTS["gamma"]))
                radius = 1.0 / float(self.infinite_params.get("lambda", cc.FUNCTION_DEFAULTS["lambda"]))
                quantile_val = float(quantiles[-1][0])
                n_calib = len(self.scores_calib)
                n = n_calib + 1

                # Build kernel matrices as concrete numpy arrays
                K_12 = cc.pairwise_kernels(
                    X=np.concatenate([self.x_calib, x_test], axis=0),
                    Y=x_test,
                    metric=kernel,
                    gamma=gamma_val,
                )
                _, L_11 = cc._get_kernel_matrix(self.x_calib, kernel, gamma_val)
                L_21 = np.linalg.solve(L_11, K_12[:-1]).T  # (1, n_calib)
                L_22_sq = float(K_12[-1, 0]) - float(L_21 @ L_21.T)
                L_22_val = np.sqrt(max(0.0, L_22_sq))

                L_full = np.vstack([
                    np.hstack([L_11, np.zeros((n_calib, 1))]),
                    np.hstack([L_21.reshape(1, -1), np.array([[L_22_val]])]),
                ])

                phi_test = self.Phi_fn(x_test).reshape(1, -1)
                Phi_full = np.vstack([self.phi_calib, phi_test])
                S_vec = np.concatenate([self.scores_calib.reshape(-1), [float(S)]])

                # Concrete problem: only Variable is eta, everything else is Constant.
                eta = cp.Variable(name="weights", shape=n)
                C_val = float(radius) / n
                L_c = cp.Constant(L_full)
                Phi_c = cp.Constant(Phi_full)

                constraints = [
                    (quantile_val - 1) <= eta,
                    quantile_val >= eta,
                    eta.T @ Phi_c == 0,
                ]
                prob = cp.Problem(
                    cp.Minimize(
                        0.5 * C_val * cp.sum_squares(L_c.T @ eta)
                        - S_vec @ eta
                    ),
                    constraints,
                )
                _solve_with_mosek(prob)

                weights = eta.value
                beta = prob.constraints[-1].dual_value
                return beta, weights
            else:
                S_arr = np.concatenate([self.scores_calib, [float(S)]])
                Phi = np.concatenate([self.phi_calib, self.Phi_fn(x)], axis=0)
                zeros = np.zeros((Phi.shape[1],))
                bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
                res = linprog(
                    -1 * S_arr, A_eq=Phi.T, b_eq=zeros, bounds=bounds,
                    method="highs-ds", options={"presolve": False},
                )
                beta = -1 * res.eqlin.marginals
                weights = None
                return beta, weights

        _CondConf_cls._get_primal_solution = _patched_get_primal_solution

        cc.binary_search = _binary_search_moe
        cc._solve_dual = _solve_dual_moe
        cc._moe_solver_patch_installed = True


def _patch_conditionalconformal_kernel_cache():
    """
    conditionalconformal's RKHS path recomputes the calibration kernel Cholesky (O(n^3))
    inside each dual solve during binary search. Cache it per calibration set to make
    per-test-point solves practical at moderate n.
    """
    try:
        import conditionalconformal.condconf as cc  # type: ignore
    except Exception:  # pragma: no cover
        return

    if getattr(cc, "_moe_kernel_cache_patched", False):
        return
    if not hasattr(cc, "_get_kernel_matrix"):
        return

    orig_get_kernel_matrix = cc._get_kernel_matrix
    cache: dict[tuple[int, str, float], tuple[weakref.ref, tuple[np.ndarray, np.ndarray]]] = {}

    def _get_kernel_matrix_cached(x_calib, kernel, gamma):
        key = (id(x_calib), str(kernel), float(gamma))
        entry = cache.get(key)
        if entry is not None:
            ref, out = entry
            if ref() is x_calib:
                return out
            cache.pop(key, None)

        out = orig_get_kernel_matrix(x_calib, kernel, gamma)

        def _cleanup(_ref):
            cur = cache.get(key)
            if cur is not None and cur[0] is _ref:
                cache.pop(key, None)

        cache[key] = (weakref.ref(x_calib, _cleanup), out)
        return out

    cc._get_kernel_matrix = _get_kernel_matrix_cached  # type: ignore[attr-defined]
    cc._moe_kernel_cache_patched = True


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


class PhiBase:
    """
    Basis function Phi(x) used to define the conditional guarantees.

    Implementations may optionally be 'fit' on calibration x to set internal state.
    """

    def fit(self, x_calib: np.ndarray) -> "PhiBase":
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class InterceptPhi(PhiBase):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x)
        return np.ones((x.shape[0], 1), dtype=float)


class LinearPhi(PhiBase):
    """
    Phi(x) = [1, x] (optionally standardized).
    """

    def __init__(self, standardize: bool = True, eps: float = 1e-8):
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def fit(self, x_calib: np.ndarray) -> "LinearPhi":
        x_calib = _as_2d(x_calib).astype(float)
        if self.standardize:
            self._mu = x_calib.mean(axis=0, keepdims=True)
            self._sigma = x_calib.std(axis=0, keepdims=True) + self.eps
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x).astype(float)
        if self.standardize and self._mu is not None and self._sigma is not None:
            x = (x - self._mu) / self._sigma
        return np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)


class LinearRFFPhi(PhiBase):
    """
    Phi(x) = [1, x, z_rff(x)], where z_rff is a Random Fourier Features approximation
    of a shift-invariant kernel (RBF or Laplacian).

    This is a practical way to get "kernel-like" nonlinear flexibility while staying
    in the finite-dimensional CondConf path (so we can use --exact and avoid CVXPY).
    """

    def __init__(
        self,
        *,
        n_rff: int,
        kernel: Literal["rbf", "laplacian"] = "rbf",
        gamma: float = 1.0,
        seed: int = 0,
        standardize: bool = True,
        eps: float = 1e-8,
    ):
        self.n_rff = int(n_rff)
        self.kernel = str(kernel)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.standardize = bool(standardize)
        self.eps = float(eps)

        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None  # (d, n_rff)
        self._b: Optional[np.ndarray] = None  # (n_rff,)

    def fit(self, x_calib: np.ndarray) -> "LinearRFFPhi":
        if self.n_rff <= 0:
            raise ValueError("n_rff must be a positive integer")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be a positive finite float")
        if self.kernel not in ("rbf", "laplacian"):
            raise ValueError("kernel must be one of: rbf, laplacian")

        x_calib = _as_2d(x_calib).astype(float)
        d = int(x_calib.shape[1])

        if self.standardize:
            self._mu = x_calib.mean(axis=0, keepdims=True)
            self._sigma = x_calib.std(axis=0, keepdims=True) + self.eps

        rng = np.random.default_rng(int(self.seed))
        if self.kernel == "rbf":
            # sklearn's rbf: k(x,y)=exp(-gamma||x-y||^2). RFF uses w ~ N(0, 2*gamma I).
            scale = float(np.sqrt(2.0 * self.gamma))
            W = rng.normal(loc=0.0, scale=scale, size=(d, int(self.n_rff)))
        else:
            # Laplacian (L1) kernel: k(x,y)=exp(-gamma||x-y||_1) has w_i ~ Cauchy(0, gamma).
            W = rng.standard_cauchy(size=(d, int(self.n_rff))) * float(self.gamma)
        b = rng.uniform(low=0.0, high=2.0 * np.pi, size=(int(self.n_rff),))

        self._W = W.astype(float, copy=False)
        self._b = b.astype(float, copy=False)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x).astype(float)
        if self.standardize and self._mu is not None and self._sigma is not None:
            x = (x - self._mu) / self._sigma

        if self._W is None or self._b is None:
            # If user forgot to call fit(), fall back to a deterministic fit-like init.
            self.fit(x)
            assert self._W is not None and self._b is not None

        proj = x @ self._W + self._b.reshape(1, -1)
        z = np.sqrt(2.0 / float(self.n_rff)) * np.cos(proj)
        return np.concatenate([np.ones((x.shape[0], 1), dtype=float), x, z], axis=1)


class NormBinsPhi(PhiBase):
    """
    One-hot bins on ||x|| with an intercept: Phi(x) = [1, 1{bin=0}, ..., 1{bin=B-1}].

    Bin edges are learned from calibration x via quantiles.
    """

    def __init__(self, num_bins: int = 5):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self.num_bins = int(num_bins)
        self._edges: Optional[np.ndarray] = None

    def fit(self, x_calib: np.ndarray) -> "NormBinsPhi":
        x_calib = _as_2d(x_calib).astype(float)
        norms = np.linalg.norm(x_calib, axis=1)
        # Internal edges exclude endpoints; np.digitize expects monotone bins.
        qs = np.linspace(0, 1, self.num_bins + 1)[1:-1]
        edges = np.quantile(norms, qs).astype(float)
        # Ensure strictly increasing edges (avoid degenerate quantiles).
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8
        self._edges = edges
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x).astype(float)
        norms = np.linalg.norm(x, axis=1)
        if self._edges is None:
            qs = np.linspace(0, 1, self.num_bins + 1)[1:-1]
            edges = np.quantile(norms, qs).astype(float)
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + 1e-8
        else:
            edges = self._edges
        bin_ids = np.digitize(norms, edges, right=False)
        onehot = np.zeros((x.shape[0], self.num_bins), dtype=float)
        onehot[np.arange(x.shape[0]), bin_ids] = 1.0
        return np.concatenate([np.ones((x.shape[0], 1), dtype=float), onehot], axis=1)


@dataclass
class Split:
    calib_idx: np.ndarray
    test_idx: np.ndarray


def split_indices(n: int, calib_frac: float, seed: int = 0) -> Split:
    if not (0.0 < calib_frac < 1.0):
        raise ValueError("calib_frac must be in (0,1)")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    n_cal = int(round(float(calib_frac) * n))
    n_cal = max(1, min(n_cal, n - 1))
    return Split(calib_idx=perm[:n_cal], test_idx=perm[n_cal:])


class CondConfSymmetricAbs:
    """
    Conditional conformal for symmetric intervals [yhat - q(x), yhat + q(x)]
    with score S(x,y) = |y - yhat(x)|.

    Implementation note:
      - To keep the conditioning covariates as just x_features (e.g. query_x), we do NOT
        pass yhat into conditionalconformal's covariates.
      - Instead we set up conditionalconformal on residual magnitudes r = |y - yhat| with
        score_fn(x, r) = r and score_inv_fn(thr, x) = [0, thr]. The predicted cutoff is
        then q(x) = thr, which we use to form [yhat - q(x), yhat + q(x)].
    """

    def __init__(
        self,
        phi: PhiBase,
        *,
        infinite_params: Optional[dict] = None,
        seed: int = 0,
        rkhs_solver: str = "auto",
        binary_search_tol: float = 1e-3,
    ):
        _require_conditionalconformal()
        _patch_conditionalconformal_solver(
            rkhs_solver=str(rkhs_solver),
            binary_search_tol=float(binary_search_tol),
        )
        self.phi = phi
        self.infinite_params = infinite_params or {}
        self.seed = int(seed)
        self._x_features_calib: Optional[np.ndarray] = None
        self._r_calib: Optional[np.ndarray] = None

        def score_fn(x: np.ndarray, r: np.ndarray) -> np.ndarray:
            # Here y is the residual magnitude r = |y - yhat|, so the score is identity.
            r = np.asarray(r).reshape(-1).astype(float)
            return r

        def Phi_fn(x: np.ndarray) -> np.ndarray:
            x = _as_2d(x).astype(float)
            return self.phi(x)

        self._gcc = _CondConf(
            score_fn=score_fn,
            Phi_fn=Phi_fn,
            infinite_params=self.infinite_params,
            seed=self.seed,
        )

    def fit(self, x_features_calib: np.ndarray, yhat_calib: np.ndarray, y_calib: np.ndarray) -> "CondConfSymmetricAbs":
        x_features_calib = _as_2d(x_features_calib).astype(float)
        yhat_calib = np.asarray(yhat_calib).reshape(-1).astype(float)
        y_calib = np.asarray(y_calib).reshape(-1).astype(float)
        r_calib = np.abs(y_calib - yhat_calib)
        self.phi.fit(x_features_calib)
        self._x_features_calib = x_features_calib
        self._r_calib = r_calib
        self._gcc.setup_problem(x_features_calib, r_calib)
        return self

    def cutoff(
        self,
        x_features_test: np.ndarray,
        yhat_test: np.ndarray,
        *,
        alpha: float,
        exact: bool = True,
        randomize: bool = False,
        workers: int = 1,
    ) -> np.ndarray:
        """
        Returns q(x) (half-widths) for each test point.

        Note: this loops over test points because conditionalconformal's public API
        is per-point. Keep test size modest for now.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        x_features_test = _as_2d(x_features_test).astype(float)
        _yhat_test = np.asarray(yhat_test).reshape(-1).astype(float)

        quantile = 1.0 - float(alpha)
        precomputed_k12: Optional[dict[bytes, tuple[np.ndarray, float]]] = None
        kernel_name = self.infinite_params.get("kernel", None)
        if kernel_name is not None and self._x_features_calib is not None:
            try:
                import conditionalconformal.condconf as cc  # type: ignore
            except Exception:  # pragma: no cover
                cc = None
            if cc is not None:
                gamma = float(self.infinite_params.get("gamma", cc.FUNCTION_DEFAULTS["gamma"]))
                K_tc = cc.pairwise_kernels(
                    X=x_features_test,
                    Y=self._x_features_calib,
                    metric=kernel_name,
                    gamma=gamma,
                )
                precomputed_k12 = {}
                for i in range(x_features_test.shape[0]):
                    x_row = np.ascontiguousarray(x_features_test[i : i + 1])
                    key = x_row.view(np.uint8).tobytes()
                    k12_col = K_tc[i : i + 1, :].T
                    k22_scalar = float(
                        cc.pairwise_kernels(
                            X=x_row,
                            metric=kernel_name,
                            gamma=gamma,
                        )[0, 0]
                    )
                    precomputed_k12[key] = (k12_col, k22_scalar)
                setattr(self._gcc, "_moe_precomputed_k12", precomputed_k12)

        def score_inv_fn(thr: float, x_test_col: np.ndarray) -> Tuple[float, float]:
            t = float(thr)
            if not np.isfinite(t):
                return (float("nan"), float("nan"))
            # Residual magnitude r is nonnegative, so the prediction set is [0, t].
            return (0.0, max(0.0, t))

        out = np.zeros((x_features_test.shape[0],), dtype=float)

        def _predict_one(gcc_obj, x):
            try:
                interval = gcc_obj.predict(
                    quantile=quantile,
                    x_test=x,
                    score_inv_fn=score_inv_fn,
                    randomize=bool(randomize),
                    exact=bool(exact),
                )
            except np.linalg.LinAlgError:
                interval = gcc_obj.predict(
                    quantile=quantile,
                    x_test=x,
                    score_inv_fn=score_inv_fn,
                    randomize=bool(randomize),
                    exact=False,
                )
            except Exception:
                # Safety net for CVXPY DCPError or other solver errors.
                # Return NaN so the experiment continues instead of crashing.
                import warnings as _w
                _w.warn("_predict_one: solver error, returning NaN for this test point")
                return float("nan")
            return float(interval[1])

        n_workers = int(max(1, workers))
        if n_workers <= 1 or x_features_test.shape[0] <= 1:
            for i in range(x_features_test.shape[0]):
                out[i] = _predict_one(self._gcc, x_features_test[i : i + 1])
            return out

        if self._x_features_calib is None or self._r_calib is None:
            raise RuntimeError("Call fit() before cutoff(..., workers>1).")

        n_workers = min(n_workers, x_features_test.shape[0])
        chunks = np.array_split(np.arange(x_features_test.shape[0]), n_workers)

        def _solve_chunk(idxs: np.ndarray):
            if idxs.size == 0:
                return []
            gcc = _CondConf(
                score_fn=self._gcc.score_fn,
                Phi_fn=self._gcc.Phi_fn,
                infinite_params=self.infinite_params,
                seed=self.seed,
            )
            gcc.setup_problem(self._x_features_calib, self._r_calib)
            if precomputed_k12 is not None:
                setattr(gcc, "_moe_precomputed_k12", precomputed_k12)
            out_local = []
            for i in idxs.tolist():
                out_local.append((int(i), _predict_one(gcc, x_features_test[i : i + 1])))
            return out_local

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for pairs in ex.map(_solve_chunk, chunks):
                for i, q in pairs:
                    out[i] = q
        return out

    @staticmethod
    def intervals(yhat: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yhat = np.asarray(yhat).reshape(-1).astype(float)
        q = np.asarray(q).reshape(-1).astype(float)
        return yhat - q, yhat + q
