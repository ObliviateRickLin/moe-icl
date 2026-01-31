from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np


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
      - We pass x_aug = concat([x_features, yhat]) into conditionalconformal.
      - score_fn uses the last column as yhat, so it stays vectorized and stateless.
      - Phi_fn is evaluated on x_features only (excludes yhat).
    """

    def __init__(
        self,
        phi: PhiBase,
        *,
        infinite_params: Optional[dict] = None,
        seed: int = 0,
    ):
        _require_conditionalconformal()
        self.phi = phi
        self.infinite_params = infinite_params or {}
        self.seed = int(seed)

        def score_fn(x_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
            x_aug = _as_2d(x_aug).astype(float)
            y = np.asarray(y).reshape(-1)
            yhat = x_aug[:, -1]
            return np.abs(y - yhat)

        def Phi_fn(x_aug: np.ndarray) -> np.ndarray:
            x_aug = _as_2d(x_aug).astype(float)
            return self.phi(x_aug[:, :-1])

        self._gcc = _CondConf(
            score_fn=score_fn,
            Phi_fn=Phi_fn,
            infinite_params=self.infinite_params,
            seed=self.seed,
        )

    def fit(self, x_features_calib: np.ndarray, yhat_calib: np.ndarray, y_calib: np.ndarray) -> "CondConfSymmetricAbs":
        x_features_calib = _as_2d(x_features_calib).astype(float)
        yhat_calib = np.asarray(yhat_calib).reshape(-1, 1).astype(float)
        y_calib = np.asarray(y_calib).reshape(-1).astype(float)
        self.phi.fit(x_features_calib)
        x_aug = np.concatenate([x_features_calib, yhat_calib], axis=1)
        self._gcc.setup_problem(x_aug, y_calib)
        return self

    def cutoff(
        self,
        x_features_test: np.ndarray,
        yhat_test: np.ndarray,
        *,
        alpha: float,
        exact: bool = True,
        randomize: bool = False,
    ) -> np.ndarray:
        """
        Returns q(x) (half-widths) for each test point.

        Note: this loops over test points because conditionalconformal's public API
        is per-point. Keep test size modest for now.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        x_features_test = _as_2d(x_features_test).astype(float)
        yhat_test = np.asarray(yhat_test).reshape(-1, 1).astype(float)

        quantile = 1.0 - float(alpha)

        def score_inv_fn(thr: float, x_test_col: np.ndarray) -> Tuple[float, float]:
            # x_test_col comes in as (d, 1); last entry is yhat.
            yhat = float(x_test_col[-1, 0])
            t = float(thr)
            return (yhat - t, yhat + t)

        out = np.zeros((x_features_test.shape[0],), dtype=float)
        for i in range(x_features_test.shape[0]):
            x_aug = np.concatenate([x_features_test[i : i + 1], yhat_test[i : i + 1]], axis=1)
            try:
                interval = self._gcc.predict(
                    quantile=quantile,
                    x_test=x_aug,
                    score_inv_fn=score_inv_fn,
                    randomize=bool(randomize),
                    exact=bool(exact),
                )
            except np.linalg.LinAlgError:
                # The upstream exact path can fail when Phi induces singular bases.
                interval = self._gcc.predict(
                    quantile=quantile,
                    x_test=x_aug,
                    score_inv_fn=score_inv_fn,
                    randomize=bool(randomize),
                    exact=False,
                )
            # interval is (lo, hi) for our score_inv_fn
            out[i] = 0.5 * (float(interval[1]) - float(interval[0]))
        return out

    @staticmethod
    def intervals(yhat: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yhat = np.asarray(yhat).reshape(-1).astype(float)
        q = np.asarray(q).reshape(-1).astype(float)
        return yhat - q, yhat + q
