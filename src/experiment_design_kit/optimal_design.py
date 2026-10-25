"""Space-filling (Latin hypercube) and D-optimal exact designs.

Classical factorial and response-surface designs assume a regular region
and a run budget that matches the design family. Two complementary tools
cover the other cases:

* **Space-filling designs** for computer experiments and surrogate
  modelling. :func:`latin_hypercube` stratifies every factor into ``n``
  equal bins with exactly one run per bin (McKay, Beckman & Conover 1979).
  :func:`maximin_latin_hypercube` improves a random LHS by element swaps
  within columns that minimise the Morris-Mitchell ``phi_p`` criterion
  (Morris & Mitchell 1995), which for large ``p`` is equivalent to
  maximising the smallest inter-point distance. :func:`centered_l2_discrepancy`
  (Hickernell 1998) and :func:`maximin_distance` score any design.
* **D-optimal exact designs** for an arbitrary run budget and candidate
  set. :func:`d_optimal_design` runs Fedorov's (1972) exchange algorithm
  with random restarts, maximising ``det(X'X)`` for a linear, interaction
  or full quadratic model. :func:`d_efficiency` reports
  ``100 * det(X'X / N)^(1/p)``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "latin_hypercube",
    "maximin_latin_hypercube",
    "maximin_distance",
    "phi_p",
    "centered_l2_discrepancy",
    "candidate_grid",
    "model_matrix",
    "d_efficiency",
    "OptimalDesign",
    "d_optimal_design",
]

MODELS = ("linear", "interaction", "quadratic")


def _check_sizes(n: int, k: int) -> None:
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")


def latin_hypercube(n: int, k: int, *, seed: Optional[int] = None, centered: bool = False) -> np.ndarray:
    """Return an ``(n, k)`` Latin hypercube sample on ``[0, 1)^k``.

    Each column is an independent permutation of the ``n`` strata
    ``[i/n, (i+1)/n)``. With ``centered=True`` points sit at stratum
    midpoints, otherwise uniformly inside them.
    """

    _check_sizes(n, k)
    rng = np.random.default_rng(seed)
    perms = np.column_stack([rng.permutation(n) for _ in range(k)])
    offset = 0.5 if centered else rng.random((n, k))
    return (perms + offset) / n


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def maximin_distance(x: Sequence[Sequence[float]]) -> float:
    """Smallest Euclidean distance between two distinct runs of ``x``."""

    pts = np.atleast_2d(np.asarray(x, dtype=float))
    if pts.shape[0] < 2:
        raise ValueError("at least two points are required")
    d = _pairwise_distances(pts)
    return float(d[np.triu_indices(pts.shape[0], k=1)].min())


def phi_p(x: Sequence[Sequence[float]], p: float = 15.0) -> float:
    """Morris-Mitchell criterion ``(sum_{i<j} d_ij^-p)^(1/p)`` (smaller is better)."""

    pts = np.atleast_2d(np.asarray(x, dtype=float))
    d = _pairwise_distances(pts)[np.triu_indices(pts.shape[0], k=1)]
    if np.any(d == 0):
        return float("inf")
    # scale by the minimum distance for numerical stability
    m = d.min()
    return float(np.sum((m / d) ** p) ** (1.0 / p) / m)


def maximin_latin_hypercube(
    n: int,
    k: int,
    *,
    seed: Optional[int] = None,
    iterations: int = 2000,
    p: float = 15.0,
    centered: bool = True,
) -> np.ndarray:
    """Return an LHS optimised for space filling by within-column swaps.

    Starting from :func:`latin_hypercube`, each iteration swaps two entries
    of one random column (which keeps the Latin property) and keeps the
    swap if it lowers :func:`phi_p`. This is the stochastic local search
    of Morris & Mitchell (1995) without annealing; with ``centered=True``
    the result lives on the stratum midpoints.
    """

    _check_sizes(n, k)
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = np.random.default_rng(seed)
    x = latin_hypercube(n, k, seed=int(rng.integers(2**31)), centered=centered)
    best = phi_p(x, p)
    for _ in range(int(iterations)):
        col = int(rng.integers(k))
        i, j = rng.choice(n, size=2, replace=False)
        x[[i, j], col] = x[[j, i], col]
        score = phi_p(x, p)
        if score < best:
            best = score
        else:
            x[[i, j], col] = x[[j, i], col]
    return x


def centered_l2_discrepancy(x: Sequence[Sequence[float]]) -> float:
    """Hickernell's centered L2 discrepancy (squared) of points in ``[0, 1]^k``.

    Matches ``scipy.stats.qmc.discrepancy(x, method="CD")``. Lower values
    mean the points are spread more uniformly.
    """

    pts = np.atleast_2d(np.asarray(x, dtype=float))
    if np.any(pts < 0) or np.any(pts > 1):
        raise ValueError("points must lie in the unit hypercube")
    n, k = pts.shape
    z = np.abs(pts - 0.5)
    term1 = (13.0 / 12.0) ** k
    term2 = np.sum(np.prod(1.0 + 0.5 * z - 0.5 * z**2, axis=1)) * 2.0 / n
    zi = z[:, None, :]
    zj = z[None, :, :]
    diff = np.abs(pts[:, None, :] - pts[None, :, :])
    prod = np.prod(1.0 + 0.5 * zi + 0.5 * zj - 0.5 * diff, axis=2)
    term3 = np.sum(prod) / n**2
    return float(term1 - term2 + term3)


def candidate_grid(k: int, levels: int = 3) -> np.ndarray:
    """Full grid of ``levels`` equally spaced coded levels in ``[-1, 1]`` per factor."""

    if k < 1 or levels < 2:
        raise ValueError("need k >= 1 and levels >= 2")
    axis = np.linspace(-1.0, 1.0, int(levels))
    return np.array(list(itertools.product(axis, repeat=int(k))), dtype=float)


def model_matrix(x: Sequence[Sequence[float]], model: str = "linear") -> np.ndarray:
    """Model matrix with intercept for ``linear``, ``interaction`` or ``quadratic`` models.

    Column order: ``1``, main effects, then (quadratic only) squares, then
    two-factor interactions — the same order as
    :func:`~experiment_design_kit.response_surface.quadratic_terms`.
    """

    pts = np.atleast_2d(np.asarray(x, dtype=float))
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    n, k = pts.shape
    cols = [np.ones(n)] + [pts[:, i] for i in range(k)]
    if model == "quadratic":
        cols += [pts[:, i] ** 2 for i in range(k)]
    if model in ("interaction", "quadratic"):
        cols += [pts[:, i] * pts[:, j] for i, j in itertools.combinations(range(k), 2)]
    return np.column_stack(cols)


def _logdet(xm: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(xm.T @ xm)
    return float(value) if sign > 0 else float("-inf")


def d_efficiency(x: Sequence[Sequence[float]], model: str = "linear") -> float:
    """D-efficiency ``100 * det(X'X / N)^(1/p)`` in percent.

    It is 100 for an orthogonal two-level design under the linear or
    interaction model on ``[-1, 1]``, and 0 when the model is not estimable.
    """

    xm = model_matrix(x, model)
    n, p = xm.shape
    ld = _logdet(xm)
    if not np.isfinite(ld):
        return 0.0
    return float(100.0 * np.exp(ld / p) / n)


@dataclass(frozen=True)
class OptimalDesign:
    """Result of :func:`d_optimal_design`."""

    matrix: np.ndarray
    indices: Tuple[int, ...]
    model: str
    log_det: float
    d_efficiency: float
    n_exchanges: int
    candidates: np.ndarray = field(repr=False)

    @property
    def n_runs(self) -> int:
        return int(self.matrix.shape[0])


def _fedorov(f: np.ndarray, start: np.ndarray, allow_repeats: bool, max_iter: int, tol: float):
    design = start.copy()
    ridge = 1e-8 * np.eye(f.shape[1])
    swaps = 0
    for _ in range(max_iter):
        xd = f[design]
        m_inv = np.linalg.inv(xd.T @ xd + ridge)
        g = f @ m_inv  # (C, p)
        d_all = np.einsum("ij,ij->i", g, f)  # variance function for every candidate
        d_i = d_all[design]  # (n,)
        d_ij = g[design] @ f.T  # (n, C)
        ratio = (1.0 - d_i)[:, None] * (1.0 + d_all)[None, :] + d_ij**2
        if not allow_repeats:
            ratio[:, design] = -np.inf
        ratio[np.arange(design.size), design] = -np.inf
        i, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        if ratio[i, j] <= 1.0 + tol:
            break
        design[i] = j
        swaps += 1
    return design, swaps


def d_optimal_design(
    n_runs: int,
    candidates: Optional[Sequence[Sequence[float]]] = None,
    *,
    k: Optional[int] = None,
    model: str = "linear",
    allow_repeats: bool = True,
    n_starts: int = 10,
    max_iter: int = 500,
    seed: Optional[int] = None,
    tol: float = 1e-9,
) -> OptimalDesign:
    """Find an exact D-optimal design of ``n_runs`` points by Fedorov exchange.

    ``candidates`` is an ``(C, k)`` array of allowed runs (constraints are
    imposed simply by leaving infeasible points out). When omitted, a
    three-level grid :func:`candidate_grid` in ``k`` factors is used. Each
    start draws a random subset of candidates; every iteration evaluates the
    determinant ratio of all point/candidate exchanges in closed form,

        det(M - x_i x_i' + x_j x_j') / det(M) = (1 - d_i)(1 + d_j) + d_ij^2,

    with ``d_ij = x_i' M^{-1} x_j``, and applies the best one until no
    exchange improves ``det(X'X)`` (Fedorov 1972; Atkinson, Donev & Tobias
    2007, ch. 12). The best of ``n_starts`` restarts is returned.
    ``allow_repeats=False`` forbids replicated candidate points.
    """

    if candidates is None:
        if k is None:
            raise ValueError("give either candidates or k")
        cand = candidate_grid(k, 3)
    else:
        cand = np.atleast_2d(np.asarray(candidates, dtype=float))
    f = model_matrix(cand, model)
    c, p = f.shape
    n_runs = int(n_runs)
    if n_runs < p:
        raise ValueError(f"n_runs must be at least the number of model terms ({p})")
    if not allow_repeats and n_runs > c:
        raise ValueError("n_runs exceeds the number of candidates without repeats")
    if np.linalg.matrix_rank(f) < p:
        raise ValueError("the model is not estimable on this candidate set")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    rng = np.random.default_rng(seed)
    best_design, best_ld, best_swaps = None, -np.inf, 0
    for _ in range(int(n_starts)):
        start = rng.choice(c, size=n_runs, replace=allow_repeats and n_runs > c)
        design, swaps = _fedorov(f, start, allow_repeats, max_iter, tol)
        ld = _logdet(f[design])
        if ld > best_ld + 1e-12:
            best_design, best_ld, best_swaps = design, ld, swaps
    if best_design is None:
        raise ValueError("no non-singular design was found; increase n_starts")
    order = np.sort(best_design)
    matrix = cand[order]
    return OptimalDesign(
        matrix=matrix,
        indices=tuple(int(i) for i in order),
        model=model,
        log_det=float(best_ld),
        d_efficiency=d_efficiency(matrix, model),
        n_exchanges=int(best_swaps),
        candidates=cand,
    )
