"""Rerandomization with a Mahalanobis-distance balance criterion.

Complete randomization balances covariates only on average; any single
draw can be badly unbalanced. Rerandomization (Morgan & Rubin 2012) keeps
drawing complete randomizations until the covariate means of the two arms
are close, as measured by the Mahalanobis distance

``M = (n_t n_c / n) (xbar_t - xbar_c)' S^{-1} (xbar_t - xbar_c)``

where ``S`` is the sample covariance of the ``k`` covariates. Under
complete randomization ``M`` is asymptotically ``chi2_k``, so the
acceptance threshold for an acceptance probability ``p_a`` is
``a = chi2_k.ppf(p_a)`` and about ``1 / p_a`` draws are needed.

Accepted assignments keep the difference-in-means estimator unbiased and
shrink the variance of every covariate mean difference (and of any linear
combination) by the factor

``v_a = P(chi2_{k+2} <= a) / P(chi2_k <= a)``

(Morgan & Rubin 2012, Theorem 3.1); the variance of the outcome estimator
falls by ``(1 - v_a) R^2`` where ``R^2`` is the squared multiple
correlation between outcome and covariates.

Inference must respect the design: :func:`rerandomization_test` runs a
Fisher randomization test whose reference distribution is drawn from the
*same* acceptance region, which keeps the test exact (a plain permutation
test would be conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "RerandomizationResult",
    "RerandomizationTestResult",
    "mahalanobis_balance",
    "rerandomization_threshold",
    "rerandomization_variance_reduction",
    "rerandomize",
    "rerandomization_test",
]


@dataclass(frozen=True)
class RerandomizationResult:
    """An accepted assignment and the design's acceptance criterion."""

    assignment: np.ndarray
    mahalanobis: float
    threshold: float
    acceptance_prob: float
    n_draws: int
    n_treated: int
    variance_reduction: float

    @property
    def variance_factor(self) -> float:
        """``v_a``: remaining fraction of each covariate mean-difference variance."""
        return 1.0 - self.variance_reduction


@dataclass(frozen=True)
class RerandomizationTestResult:
    """Fisher randomization test that conditions on the acceptance region."""

    estimate: float
    p_value: float
    n_permutations: int
    threshold: float
    alternative: str


def _covariates(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("X must be an (n, k) covariate matrix with n >= 3")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    return X


def _precision(X: np.ndarray) -> np.ndarray:
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    if np.any(np.diag(cov) <= 0):
        raise ValueError("every covariate must vary")
    # pseudo-inverse handles exactly collinear covariates (rank-deficient S)
    return np.linalg.pinv(cov)


def _rank(X: np.ndarray) -> int:
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    return int(np.linalg.matrix_rank(cov))


def _mahalanobis_many(X: np.ndarray, prec: np.ndarray, W: np.ndarray) -> np.ndarray:
    """M for each row of the 0/1 assignment matrix ``W`` (draws x n)."""
    n = X.shape[0]
    n_t = W.sum(axis=1)
    n_c = n - n_t
    mean_t = W @ X / n_t[:, None]
    mean_c = (1 - W) @ X / n_c[:, None]
    diff = mean_t - mean_c
    quad = np.einsum("ij,jk,ik->i", diff, prec, diff)
    return n_t * n_c / n * quad


def _check_assignment(assignment, n: int) -> np.ndarray:
    w = np.asarray(assignment)
    if w.shape != (n,):
        raise ValueError("assignment must have one entry per row of X")
    if not np.all(np.isin(w, (0, 1))):
        raise ValueError("assignment must be 0/1 (control/treatment)")
    w = w.astype(int)
    if w.sum() in (0, n):
        raise ValueError("assignment needs at least one unit in each arm")
    return w


def mahalanobis_balance(X, assignment) -> float:
    """Mahalanobis distance between treated and control covariate means."""
    X = _covariates(X)
    w = _check_assignment(assignment, X.shape[0])
    return float(_mahalanobis_many(X, _precision(X), w[None, :].astype(float))[0])


def _check_prob(acceptance_prob: float) -> float:
    p = float(acceptance_prob)
    if not 0.0 < p <= 1.0:
        raise ValueError("acceptance_prob must be in (0, 1]")
    return p


def rerandomization_threshold(k: int, acceptance_prob: float) -> float:
    """Acceptance threshold ``a`` with ``P(chi2_k <= a) = acceptance_prob``."""
    if int(k) < 1:
        raise ValueError("k must be a positive integer")
    p = _check_prob(acceptance_prob)
    return float("inf") if p == 1.0 else float(stats.chi2.ppf(p, int(k)))


def rerandomization_variance_reduction(k: int, threshold: float) -> float:
    """Percent reduction ``1 - v_a`` in each covariate mean-difference variance."""
    if int(k) < 1:
        raise ValueError("k must be a positive integer")
    a = float(threshold)
    if not a > 0:
        raise ValueError("threshold must be positive")
    if np.isinf(a):
        return 0.0
    v_a = stats.chi2.cdf(a, int(k) + 2) / stats.chi2.cdf(a, int(k))
    return float(1.0 - v_a)


def _draws(rng: np.random.Generator, n: int, n_treated: int, size: int) -> np.ndarray:
    keys = rng.random((size, n))
    order = np.argsort(keys, axis=1)
    W = np.zeros((size, n))
    np.put_along_axis(W, order[:, :n_treated], 1.0, axis=1)
    return W


def rerandomize(
    X,
    *,
    n_treated: int | None = None,
    acceptance_prob: float = 0.01,
    threshold: float | None = None,
    max_draws: int = 1_000_000,
    batch_size: int = 1024,
    seed: int | None = None,
) -> RerandomizationResult:
    """Draw complete randomizations until ``M <= threshold`` (Morgan & Rubin 2012).

    ``n_treated`` defaults to ``n // 2``. The threshold is
    ``chi2_k.ppf(acceptance_prob)`` unless given explicitly, where ``k`` is
    the rank of the covariate covariance. Raises ``RuntimeError`` if no
    draw is accepted within ``max_draws``.
    """
    X = _covariates(X)
    n = X.shape[0]
    n_t = n // 2 if n_treated is None else int(n_treated)
    if not 1 <= n_t <= n - 1:
        raise ValueError("n_treated must leave at least one unit in each arm")
    k = _rank(X)
    p = _check_prob(acceptance_prob)
    a = rerandomization_threshold(k, p) if threshold is None else float(threshold)
    if not a > 0:
        raise ValueError("threshold must be positive")
    if int(max_draws) < 1 or int(batch_size) < 1:
        raise ValueError("max_draws and batch_size must be positive")
    prec = _precision(X)
    rng = np.random.default_rng(seed)
    used = 0
    while used < int(max_draws):
        size = min(int(batch_size), int(max_draws) - used)
        W = _draws(rng, n, n_t, size)
        M = _mahalanobis_many(X, prec, W)
        hits = np.flatnonzero(M <= a)
        if hits.size:
            i = int(hits[0])
            return RerandomizationResult(
                assignment=W[i].astype(int),
                mahalanobis=float(M[i]),
                threshold=a,
                acceptance_prob=p if threshold is None else float(stats.chi2.cdf(a, k)),
                n_draws=used + i + 1,
                n_treated=n_t,
                variance_reduction=rerandomization_variance_reduction(k, a),
            )
        used += size
    raise RuntimeError("no acceptable assignment within max_draws; raise acceptance_prob or max_draws")


def rerandomization_test(
    y,
    X,
    assignment,
    *,
    threshold: float | None = None,
    acceptance_prob: float = 0.01,
    n_permutations: int = 999,
    alternative: str = "two-sided",
    max_draws: int = 10_000_000,
    seed: int | None = None,
) -> RerandomizationTestResult:
    """Fisher randomization test of the sharp null under rerandomization.

    The reference distribution of the difference in means is built from
    ``n_permutations`` assignments drawn with the same arm sizes and the same
    acceptance rule ``M <= threshold`` used in the design. The p-value uses
    the ``(b + 1) / (B + 1)`` correction, so it is valid in finite samples.
    """
    X = _covariates(X)
    n = X.shape[0]
    y = np.asarray(y, dtype=float).ravel()
    if y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError("y must be finite with one value per row of X")
    w = _check_assignment(assignment, n)
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be positive")
    k = _rank(X)
    a = rerandomization_threshold(k, acceptance_prob) if threshold is None else float(threshold)
    prec = _precision(X)
    n_t = int(w.sum())
    rng = np.random.default_rng(seed)
    observed = float(y[w == 1].mean() - y[w == 0].mean())
    stats_ref: list[np.ndarray] = []
    collected, used = 0, 0
    while collected < int(n_permutations):
        if used >= int(max_draws):
            raise RuntimeError("acceptance region too small: could not collect enough assignments")
        W = _draws(rng, n, n_t, 2048)
        used += 2048
        W = W[_mahalanobis_many(X, prec, W) <= a]
        if W.size:
            diff = W @ y / n_t - (1 - W) @ y / (n - n_t)
            stats_ref.append(diff)
            collected += diff.size
    ref = np.concatenate(stats_ref)[: int(n_permutations)]
    tol = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        b = int(np.sum(np.abs(ref) >= abs(observed) - tol))
    elif alternative == "greater":
        b = int(np.sum(ref >= observed - tol))
    else:
        b = int(np.sum(ref <= observed + tol))
    return RerandomizationTestResult(
        estimate=observed,
        p_value=(b + 1) / (ref.size + 1),
        n_permutations=int(ref.size),
        threshold=a,
        alternative=alternative,
    )
