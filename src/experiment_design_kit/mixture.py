"""Mixture experiments: simplex designs, constrained regions and Scheffé models.

In a mixture experiment the factors are proportions ``x_1..x_q`` of a blend,
so ``x_i >= 0`` and ``sum(x) = 1``. The usual factorial/response-surface
machinery does not apply (the intercept is confounded with the constraint),
so this module provides the standard alternatives (Cornell, *Experiments with
Mixtures*, 3rd ed.):

* :func:`simplex_lattice` — the ``{q, m}`` simplex-lattice: every blend whose
  proportions are multiples of ``1/m`` (``C(q+m-1, m)`` points).
* :func:`simplex_centroid` — the ``2^q - 1`` centroids of all non-empty
  subsets of components (Scheffé 1963).
* ``augment=True`` on either adds the overall centroid and the ``q`` interior
  *axial check blends* ``x_i = (q+1)/(2q)``, ``x_j = 1/(2q)``.
* :func:`extreme_vertices` — McLean & Anderson (1966) vertices of the region
  cut out by lower/upper bounds ``L_i <= x_i <= U_i`` (plus optional edge and
  overall centroids); :func:`to_pseudocomponents` / :func:`from_pseudocomponents`
  map lower-bounded regions onto a full simplex.
* :func:`fit_scheffe` — least-squares Scheffé canonical polynomials
  (``linear``, ``quadratic``, ``special_cubic``, ``cubic``) with no intercept,
  R² computed about the mean.
* :func:`cox_effects` — response traces along Cox directions from a
  reference blend and each component's total effect.
* :func:`best_blend` — grid search of a fitted model over the (constrained)
  simplex.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MixtureDesign",
    "MixtureFit",
    "SCHEFFE_MODELS",
    "simplex_lattice",
    "simplex_centroid",
    "extreme_vertices",
    "to_pseudocomponents",
    "from_pseudocomponents",
    "scheffe_terms",
    "scheffe_model_matrix",
    "fit_scheffe",
    "cox_effects",
    "best_blend",
]

SCHEFFE_MODELS = ("linear", "quadratic", "special_cubic", "cubic")
_TOL = 1e-9


@dataclass
class MixtureDesign:
    """A mixture design: rows of ``matrix`` are blends summing to one."""

    matrix: np.ndarray
    components: Tuple[str, ...]
    kind: str
    point_types: Tuple[str, ...]

    @property
    def n_runs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.matrix.shape[1])


def _names(q: int, components: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if components is None:
        return tuple(f"x{i + 1}" for i in range(q))
    names = tuple(str(c) for c in components)
    if len(names) != q:
        raise ValueError(f"components must name {q} components")
    if len(set(names)) != q:
        raise ValueError("component names must be unique")
    return names


def _check_q(q: int) -> None:
    if not isinstance(q, (int, np.integer)) or isinstance(q, bool) or q < 2:
        raise ValueError("q (number of components) must be an integer >= 2")


def _point_type(row: np.ndarray) -> str:
    nz = int(np.sum(row > _TOL))
    if nz == 1:
        return "vertex"
    if nz == 2:
        return "binary"
    if nz == len(row):
        return "interior"
    return f"{nz}-blend"


def _augment(rows: List[np.ndarray], types: List[str], q: int) -> None:
    centroid = np.full(q, 1.0 / q)
    if not any(np.allclose(r, centroid) for r in rows):
        rows.append(centroid)
        types.append("centroid")
    for i in range(q):
        r = np.full(q, 1.0 / (2 * q))
        r[i] = (q + 1) / (2 * q)
        rows.append(r)
        types.append("axial")


def simplex_lattice(q: int, m: int, *, augment: bool = False,
                    components: Optional[Sequence[str]] = None) -> MixtureDesign:
    """The ``{q, m}`` simplex-lattice design (proportions ``0, 1/m, ..., 1``)."""

    _check_q(q)
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError("m (lattice degree) must be an integer >= 1")
    rows: List[np.ndarray] = []
    for combo in itertools.combinations_with_replacement(range(q), m):
        counts = np.bincount(np.asarray(combo), minlength=q)
        rows.append(counts / m)
    # order: vertices first, then by number of non-zero components
    rows.sort(key=lambda r: (int(np.sum(r > _TOL)), tuple(-r)))
    types = [_point_type(r) for r in rows]
    if augment:
        _augment(rows, types, q)
    return MixtureDesign(np.vstack(rows), _names(q, components), f"simplex-lattice {{{q},{m}}}",
                         tuple(types))


def simplex_centroid(q: int, *, augment: bool = False,
                     components: Optional[Sequence[str]] = None) -> MixtureDesign:
    """Simplex-centroid design: the centroid of every non-empty subset (``2^q - 1`` runs)."""

    _check_q(q)
    rows: List[np.ndarray] = []
    types: List[str] = []
    for size in range(1, q + 1):
        for subset in itertools.combinations(range(q), size):
            r = np.zeros(q)
            r[list(subset)] = 1.0 / size
            rows.append(r)
            types.append("centroid" if size == q else _point_type(r))
    if augment:
        for i in range(q):
            r = np.full(q, 1.0 / (2 * q))
            r[i] = (q + 1) / (2 * q)
            rows.append(r)
            types.append("axial")
    return MixtureDesign(np.vstack(rows), _names(q, components), f"simplex-centroid q={q}", tuple(types))


def _bounds(q: Optional[int], lower, upper) -> Tuple[np.ndarray, np.ndarray]:
    if lower is None and upper is None:
        if q is None:
            raise ValueError("give q or at least one of lower/upper")
    n = q if q is not None else len(lower if lower is not None else upper)
    lo = np.zeros(n) if lower is None else np.asarray(lower, dtype=float).reshape(-1)
    hi = np.ones(n) if upper is None else np.asarray(upper, dtype=float).reshape(-1)
    if lo.size != n or hi.size != n:
        raise ValueError("lower and upper must have one bound per component")
    if n < 2:
        raise ValueError("a mixture needs at least 2 components")
    if np.any(lo < -_TOL) or np.any(hi > 1 + _TOL) or np.any(lo > hi + _TOL):
        raise ValueError("bounds must satisfy 0 <= lower <= upper <= 1")
    if lo.sum() > 1 + _TOL or hi.sum() < 1 - _TOL:
        raise ValueError("bounds are inconsistent: need sum(lower) <= 1 <= sum(upper)")
    return lo, hi


def extreme_vertices(lower: Optional[Sequence[float]] = None, upper: Optional[Sequence[float]] = None,
                     *, q: Optional[int] = None, centroids: bool = True,
                     components: Optional[Sequence[str]] = None) -> MixtureDesign:
    """McLean–Anderson extreme vertices of ``{L <= x <= U, sum x = 1}``.

    For every component ``i`` and every assignment of the other components
    to a bound, ``x_i = 1 - sum(others)`` is kept if it lies within
    ``[L_i, U_i]``. With ``centroids=True`` the centroids of the region's
    edges (pairs of vertices sharing ``q - 2`` active bounds) and the overall
    centroid (mean of vertices) are appended.
    """

    lo, hi = _bounds(q, lower, upper)
    n = lo.size
    verts: List[np.ndarray] = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for choice in itertools.product((0, 1), repeat=n - 1):
            r = np.empty(n)
            for j, c in zip(others, choice):
                r[j] = hi[j] if c else lo[j]
            r[i] = 1.0 - r[others].sum()
            if lo[i] - _TOL <= r[i] <= hi[i] + _TOL:
                r = np.clip(r, lo, hi)
                if not any(np.allclose(r, v, atol=1e-9) for v in verts):
                    verts.append(r)
    verts.sort(key=lambda r: tuple(-r))
    rows = list(verts)
    types = ["vertex"] * len(verts)
    if centroids and len(verts) > 1:
        def active(v: np.ndarray) -> set:
            return {(j, "L") for j in range(n) if abs(v[j] - lo[j]) < 1e-9} | {
                (j, "U") for j in range(n) if abs(v[j] - hi[j]) < 1e-9}

        acts = [active(v) for v in verts]
        if n >= 3:
            for a, b in itertools.combinations(range(len(verts)), 2):
                shared = acts[a] & acts[b]
                if len(shared) >= n - 2:
                    # edge: the two vertices lie on a common 1-D face
                    mid = (verts[a] + verts[b]) / 2
                    if not any(np.allclose(mid, r) for r in rows):
                        rows.append(mid)
                        types.append("edge-centroid")
        overall = np.mean(np.vstack(verts), axis=0)
        if not any(np.allclose(overall, r) for r in rows):
            rows.append(overall)
            types.append("centroid")
    return MixtureDesign(np.vstack(rows), _names(n, components), "extreme-vertices", tuple(types))


def to_pseudocomponents(x, lower: Sequence[float]) -> np.ndarray:
    """L-pseudocomponents ``x'_i = (x_i - L_i) / (1 - sum L)``."""

    lo = np.asarray(lower, dtype=float).reshape(-1)
    total = 1.0 - lo.sum()
    if total <= _TOL:
        raise ValueError("sum(lower) must be < 1")
    xx = np.atleast_2d(np.asarray(x, dtype=float))
    if xx.shape[1] != lo.size:
        raise ValueError(f"x must have {lo.size} columns")
    return (xx - lo) / total


def from_pseudocomponents(z, lower: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`to_pseudocomponents`: ``x_i = L_i + (1 - sum L) z_i``."""

    lo = np.asarray(lower, dtype=float).reshape(-1)
    total = 1.0 - lo.sum()
    if total <= _TOL:
        raise ValueError("sum(lower) must be < 1")
    zz = np.atleast_2d(np.asarray(z, dtype=float))
    if zz.shape[1] != lo.size:
        raise ValueError(f"z must have {lo.size} columns")
    return lo + total * zz


def _check_model(model: str) -> None:
    if model not in SCHEFFE_MODELS:
        raise ValueError(f"model must be one of {SCHEFFE_MODELS}")


def scheffe_terms(components: Sequence[str], model: str = "quadratic") -> List[str]:
    """Term labels of a Scheffé canonical polynomial."""

    _check_model(model)
    names = list(components)
    terms = list(names)
    if model in ("quadratic", "special_cubic", "cubic"):
        terms += [f"{a}*{b}" for a, b in itertools.combinations(names, 2)]
    if model == "cubic":
        terms += [f"{a}*{b}*({a}-{b})" for a, b in itertools.combinations(names, 2)]
    if model in ("special_cubic", "cubic"):
        terms += [f"{a}*{b}*{c}" for a, b, c in itertools.combinations(names, 3)]
    return terms


def scheffe_model_matrix(x, model: str = "quadratic") -> np.ndarray:
    """Columns of the Scheffé polynomial (no intercept) evaluated at blends ``x``."""

    _check_model(model)
    xx = np.atleast_2d(np.asarray(x, dtype=float))
    q = xx.shape[1]
    cols = [xx[:, i] for i in range(q)]
    pairs = list(itertools.combinations(range(q), 2))
    if model in ("quadratic", "special_cubic", "cubic"):
        cols += [xx[:, i] * xx[:, j] for i, j in pairs]
    if model == "cubic":
        cols += [xx[:, i] * xx[:, j] * (xx[:, i] - xx[:, j]) for i, j in pairs]
    if model in ("special_cubic", "cubic"):
        cols += [xx[:, i] * xx[:, j] * xx[:, k] for i, j, k in itertools.combinations(range(q), 3)]
    return np.column_stack(cols)


@dataclass
class MixtureFit:
    """Least-squares Scheffé polynomial fit."""

    model: str
    components: Tuple[str, ...]
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    rmse: float
    resid_df: int
    beta: np.ndarray = field(repr=False)

    def predict(self, x) -> np.ndarray:
        """Predicted response at blends ``x`` (``(n, q)`` or a single blend)."""

        xx = np.atleast_2d(np.asarray(x, dtype=float))
        if xx.shape[1] != len(self.components):
            raise ValueError(f"blends must have {len(self.components)} components")
        return scheffe_model_matrix(xx, self.model) @ self.beta


def _check_blends(x: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("design must be a 2-D array with >= 2 component columns")
    if np.any(x < -1e-8) or not np.allclose(x.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("every blend must be non-negative and sum to 1")


def fit_scheffe(design, y: Sequence[float], *, model: str = "quadratic",
                components: Optional[Sequence[str]] = None) -> MixtureFit:
    """Fit a Scheffé canonical polynomial by least squares.

    The model has no intercept (it is absorbed by ``sum x = 1``); R² and the
    adjusted R² are computed about the mean with ``p - 1`` model degrees of
    freedom, as is standard for mixture models.
    """

    _check_model(model)
    if isinstance(design, MixtureDesign):
        x = design.matrix
        names = design.components
    else:
        x = np.asarray(design, dtype=float)
        names = None
    _check_blends(x)
    names = _names(x.shape[1], components if components is not None else names)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size != x.shape[0]:
        raise ValueError(f"y must have {x.shape[0]} values (one per run)")
    if not np.all(np.isfinite(yy)):
        raise ValueError("y must contain only finite values")
    X = scheffe_model_matrix(x, model)
    n, p = X.shape
    if np.linalg.matrix_rank(X) < p:
        raise ValueError(f"design cannot estimate the {model} Scheffé model (rank deficient); "
                         "add runs or use a smaller model")
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    sse = float(resid @ resid)
    sst = float(((yy - yy.mean()) ** 2).sum())
    resid_df = n - p
    r2 = 1 - sse / sst if sst > 0 else 1.0
    adj = 1 - (1 - r2) * (n - 1) / resid_df if resid_df > 0 and sst > 0 else float("nan")
    if resid_df > 0:
        sigma2 = sse / resid_df
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(X.T @ X)) * sigma2, 0, None))
        rmse = math.sqrt(sigma2)
    else:
        se = np.full(p, float("nan"))
        rmse = float("nan")
    terms = scheffe_terms(names, model)
    return MixtureFit(model=model, components=names,
                      coefficients={t: float(b) for t, b in zip(terms, beta)},
                      std_errors={t: float(s) for t, s in zip(terms, se)},
                      r_squared=float(r2), adj_r_squared=float(adj), rmse=rmse,
                      resid_df=int(resid_df), beta=beta)


def cox_effects(fit: MixtureFit, reference: Optional[Sequence[float]] = None,
                n_points: int = 21) -> Dict[str, Dict[str, object]]:
    """Response traces along Cox directions and total component effects.

    Along component ``i``'s Cox direction from reference blend ``s`` the other
    components keep their relative proportions:
    ``x_j = s_j (1 - x_i) / (1 - s_i)``. Returns, per component, the grid of
    ``x_i`` in ``[0, 1]``, the predicted trace and the total effect
    ``y(x_i = 1) - y(x_i = 0)``. The reference defaults to the centroid.
    """

    q = len(fit.components)
    s = np.full(q, 1.0 / q) if reference is None else np.asarray(reference, dtype=float).reshape(-1)
    if s.size != q:
        raise ValueError(f"reference must have {q} components")
    _check_blends(s[None, :])
    if np.any(s >= 1 - _TOL):
        raise ValueError("reference must not be a pure component")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    grid = np.linspace(0.0, 1.0, n_points)
    out: Dict[str, Dict[str, object]] = {}
    for i, name in enumerate(fit.components):
        pts = np.outer(1 - grid, s / (1 - s[i]))
        pts[:, i] = grid
        trace = fit.predict(pts)
        out[name] = {"x": grid, "response": trace, "total_effect": float(trace[-1] - trace[0])}
    return out


def best_blend(fit: MixtureFit, *, lower: Optional[Sequence[float]] = None,
               upper: Optional[Sequence[float]] = None, resolution: int = 20,
               maximize: bool = True) -> Tuple[np.ndarray, float]:
    """Grid-search the fitted model over the (bounded) simplex.

    Evaluates every ``{q, resolution}`` lattice blend satisfying the bounds
    and returns ``(blend, predicted_response)`` at the best one.
    """

    q = len(fit.components)
    lo, hi = _bounds(q, lower, upper)
    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    if math.comb(q + resolution - 1, resolution) > 2_000_000:
        raise ValueError("lattice too large; lower resolution")
    grid = simplex_lattice(q, resolution).matrix
    ok = np.all((grid >= lo - 1e-9) & (grid <= hi + 1e-9), axis=1)
    if not ok.any():
        raise ValueError("no lattice blend satisfies the bounds; increase resolution")
    cand = grid[ok]
    pred = fit.predict(cand)
    k = int(np.argmax(pred) if maximize else np.argmin(pred))
    return cand[k], float(pred[k])
