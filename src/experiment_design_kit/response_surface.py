"""Screening (Plackett–Burman) and response-surface designs with quadratic analysis.

Two-level factorials scale as ``2^k``. Two classical families fill the gaps:

* :func:`plackett_burman` builds an orthogonal two-level *screening* design
  in ``N`` runs (``N`` a multiple of 4) for up to ``N - 1`` factors
  (Plackett & Burman 1946). Main effects are estimated independently of
  each other; two-factor interactions are partially aliased with them, so
  the design is meant for finding the vital few factors. The returned
  :class:`~experiment_design_kit.factorial.FactorialDesign` plugs straight
  into :func:`~experiment_design_kit.factorial.factorial_effects` (use
  :func:`screening_effects`, which restricts the model to main effects and
  falls back to Lenth's method when saturated).
* :func:`central_composite` (Box & Wilson 1951) and :func:`box_behnken`
  (Box & Behnken 1960) are *response-surface* designs that support a full
  second-order model. :func:`fit_quadratic` fits that model by least
  squares and performs canonical analysis: stationary point
  ``x_s = -B^{-1} b / 2``, eigenvalues of ``B`` and the nature of the point
  (maximum, minimum or saddle).
* :func:`scaled_prediction_variance` computes ``N x' (X'X)^{-1} x`` so the
  design's prediction-variance profile (rotatability) can be inspected.

All designs are in coded units (``-1/+1`` factorial levels, ``0`` centre).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .factorial import (
    FACTOR_LETTERS,
    FactorialDesign,
    FactorialEffects,
    factorial_effects,
    fractional_factorial,
    full_factorial,
)

__all__ = [
    "PLACKETT_BURMAN_GENERATORS",
    "plackett_burman",
    "screening_effects",
    "ResponseSurfaceDesign",
    "central_composite",
    "box_behnken",
    "QuadraticFit",
    "fit_quadratic",
    "scaled_prediction_variance",
    "quadratic_terms",
]

#: First rows of the cyclic Plackett–Burman designs (Plackett & Burman 1946;
#: Montgomery, *Design and Analysis of Experiments*, Table 8.23).
PLACKETT_BURMAN_GENERATORS: Dict[int, str] = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def _hadamard(n: int) -> Optional[np.ndarray]:
    """A normalised Hadamard matrix of order ``n`` (first column all +1), or None."""

    if n == 1:
        return np.ones((1, 1))
    if n == 2:
        return np.array([[1.0, 1.0], [1.0, -1.0]])
    if n in PLACKETT_BURMAN_GENERATORS:
        row = np.array([1.0 if c == "+" else -1.0 for c in PLACKETT_BURMAN_GENERATORS[n]])
        body = np.array([np.roll(row, i) for i in range(n - 1)] + [-np.ones(n - 1)])
        return np.column_stack([np.ones(n), body])
    q = n - 1
    if _is_prime(q) and q % 4 == 3:  # Paley construction I
        residues = {(i * i) % q for i in range(1, q)}
        chi = np.array([0.0] + [1.0 if j in residues else -1.0 for j in range(1, q)])
        Q = np.array([[chi[(j - i) % q] for j in range(q)] for i in range(q)])
        S = np.zeros((n, n))
        S[0, 1:] = 1.0
        S[1:, 0] = -1.0
        S[1:, 1:] = Q
        H = S + np.eye(n)
        # normalise so that the first column is all +1
        H = H * H[:, [0]]
        return H
    if n % 2 == 0:
        half = _hadamard(n // 2)
        if half is not None:  # Sylvester doubling
            return np.block([[half, half], [half, -half]])
    return None


def plackett_burman(k: int, n_runs: Optional[int] = None) -> FactorialDesign:
    """A ``n_runs``-run Plackett–Burman screening design for ``k`` factors.

    ``n_runs`` defaults to the smallest multiple of 4 that exceeds ``k``.
    Sizes 12, 20 and 24 use the published cyclic generators; other sizes use
    Paley (``N - 1`` prime, ``= 3 mod 4``) or Sylvester doubling. The first
    ``k`` columns of the design are returned; columns are mutually
    orthogonal and balanced (``X'X = N I``).
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(FACTOR_LETTERS):
        raise ValueError(f"at most {len(FACTOR_LETTERS)} factors are supported")
    if n_runs is None:
        n_runs = 4 * (k // 4 + 1)
    if n_runs % 4 != 0 or n_runs <= k:
        raise ValueError("n_runs must be a multiple of 4 greater than k")
    H = _hadamard(n_runs)
    if H is None:
        raise ValueError(f"no Plackett-Burman construction available for N={n_runs}")
    matrix = H[:, 1 : k + 1].copy()
    factors = tuple(FACTOR_LETTERS[:k])
    return FactorialDesign(matrix=matrix, factors=factors, resolution=3 if k > 2 else None)


def screening_effects(design: FactorialDesign, y: Sequence[float], *, alpha: float = 0.05) -> FactorialEffects:
    """Main-effect analysis of a screening design (t tests, or Lenth when saturated)."""

    return factorial_effects(design, y, alpha=alpha, max_order=1)


@dataclass(frozen=True)
class ResponseSurfaceDesign:
    """A coded response-surface design."""

    matrix: np.ndarray
    factors: Tuple[str, ...]
    kind: str
    alpha: Optional[float] = None
    n_factorial: int = 0
    n_axial: int = 0
    n_center: int = 0
    point_types: Tuple[str, ...] = ()

    @property
    def n_runs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def k(self) -> int:
        return len(self.factors)

    def to_levels(self, levels: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Convert coded units to natural units given ``{factor: (low, high)}``."""

        missing = [f for f in self.factors if f not in levels]
        if missing:
            raise ValueError(f"levels are missing for factors {missing}")
        lo = np.array([levels[f][0] for f in self.factors], dtype=float)
        hi = np.array([levels[f][1] for f in self.factors], dtype=float)
        return (lo + hi) / 2 + self.matrix * (hi - lo) / 2


def central_composite(
    k: int,
    *,
    alpha: Union[str, float] = "rotatable",
    center: Tuple[int, int] = (4, 4),
    face: str = "circumscribed",
    fraction: int = 0,
) -> ResponseSurfaceDesign:
    """Central composite design: ``2^(k-p)`` cube + ``2k`` axial + centre points.

    ``alpha`` is ``"rotatable"`` (``F^{1/4}``), ``"orthogonal"`` (pure
    quadratic columns orthogonal: ``alpha^2 = (sqrt(F N) - F) / 2``),
    ``"face"`` (``alpha = 1``) or a positive number. ``center`` gives the
    number of centre runs in the cube and axial portions. ``face`` is
    ``"circumscribed"`` (CCC, axial points at ``±alpha``), ``"inscribed"``
    (CCI, the CCC scaled so axial points sit at ``±1``) or ``"faced"`` (CCF,
    same as ``alpha="face"``). ``fraction`` uses a minimum-aberration
    ``2^(k-p)`` cube (resolution V or higher is recommended).
    """

    if k < 2:
        raise ValueError("a central composite design needs k >= 2")
    if face not in ("circumscribed", "inscribed", "faced"):
        raise ValueError("face must be 'circumscribed', 'inscribed' or 'faced'")
    if len(center) != 2 or min(center) < 0:
        raise ValueError("center must be a pair of non-negative counts")
    cube = (fractional_factorial(k, p=fraction) if fraction else full_factorial(k)).matrix
    F = cube.shape[0]
    n_center = int(sum(center))
    N = F + 2 * k + n_center
    if face == "faced":
        alpha = "face"
    if isinstance(alpha, str):
        if alpha == "rotatable":
            a = F ** 0.25
        elif alpha == "orthogonal":
            a = float(np.sqrt((np.sqrt(F * N) - F) / 2))
        elif alpha == "face":
            a = 1.0
        else:
            raise ValueError("alpha must be 'rotatable', 'orthogonal', 'face' or a positive number")
    else:
        a = float(alpha)
        if not a > 0:
            raise ValueError("alpha must be positive")
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = -a
        axial[2 * i + 1, i] = a
    matrix = np.vstack([cube, np.zeros((center[0], k)), axial, np.zeros((center[1], k))])
    types = ("factorial",) * F + ("center",) * center[0] + ("axial",) * (2 * k) + ("center",) * center[1]
    if face == "inscribed":
        matrix = matrix / a
        a = 1.0
    return ResponseSurfaceDesign(matrix=matrix, factors=tuple(FACTOR_LETTERS[:k]),
                                 kind=f"ccd-{face}", alpha=float(a), n_factorial=F,
                                 n_axial=2 * k, n_center=n_center, point_types=types)


def box_behnken(k: int, *, center: int = 3) -> ResponseSurfaceDesign:
    """Box–Behnken design: a ``2^2`` factorial on every factor pair, others at 0.

    For ``k = 3, 4, 5`` this is exactly the Box & Behnken (1960) design
    (12, 24 and 40 edge points). For larger ``k`` the all-pairs
    construction is still a valid three-level second-order design. No run
    lies at a cube corner, which helps when extreme combinations are unsafe.
    """

    if k < 3:
        raise ValueError("a Box-Behnken design needs k >= 3")
    if center < 0:
        raise ValueError("center must be non-negative")
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.product((-1.0, 1.0), repeat=2):
            row = np.zeros(k)
            row[i], row[j] = a, b
            rows.append(row)
    edges = len(rows)
    matrix = np.vstack(rows + [np.zeros(k)] * center)
    types = ("edge",) * edges + ("center",) * center
    return ResponseSurfaceDesign(matrix=matrix, factors=tuple(FACTOR_LETTERS[:k]),
                                 kind="box-behnken", alpha=None, n_factorial=edges,
                                 n_axial=0, n_center=center, point_types=types)


def quadratic_terms(factors: Sequence[str]) -> List[str]:
    """Names of the full second-order model terms: ``1, A, B, ..., A^2, ..., AB, ...``."""

    factors = list(factors)
    return (["1"] + factors + [f"{f}^2" for f in factors]
            + [a + b for a, b in itertools.combinations(factors, 2)])


def _model_matrix(x: np.ndarray) -> np.ndarray:
    n, k = x.shape
    cols = [np.ones(n)] + [x[:, i] for i in range(k)] + [x[:, i] ** 2 for i in range(k)]
    cols += [x[:, i] * x[:, j] for i, j in itertools.combinations(range(k), 2)]
    return np.column_stack(cols)


@dataclass(frozen=True)
class QuadraticFit:
    """A fitted second-order response surface with canonical analysis."""

    coefficients: Dict[str, float]
    factors: Tuple[str, ...]
    r_squared: float
    adj_r_squared: float
    rmse: float
    resid_df: int
    b: np.ndarray = field(repr=False)
    B: np.ndarray = field(repr=False)
    stationary_point: Optional[np.ndarray] = None
    stationary_response: Optional[float] = None
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = field(default=None, repr=False)
    nature: str = "undetermined"

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        """Predicted response at coded points ``x`` (``(n, k)`` or a single point)."""

        pts = np.atleast_2d(np.asarray(x, dtype=float))
        if pts.shape[1] != len(self.factors):
            raise ValueError(f"points must have {len(self.factors)} coordinates")
        b0 = self.coefficients["1"]
        return b0 + pts @ self.b + np.einsum("ni,ij,nj->n", pts, self.B, pts)


def fit_quadratic(
    design: Union[ResponseSurfaceDesign, np.ndarray],
    y: Sequence[float],
    *,
    factors: Optional[Sequence[str]] = None,
    tol: float = 1e-8,
) -> QuadraticFit:
    """Fit ``y = b0 + x'b + x'Bx`` by least squares and run canonical analysis.

    ``B`` is symmetric with the pure quadratic coefficients on its diagonal
    and half the interaction coefficients off it. When ``B`` is
    non-singular the stationary point is ``-B^{-1} b / 2``; all-negative
    eigenvalues indicate a maximum, all-positive a minimum, mixed signs a
    saddle. Eigenvalues with ``|lambda| < tol * max|lambda|`` signal a ridge.
    """

    if isinstance(design, ResponseSurfaceDesign):
        x = design.matrix
        names = design.factors
    else:
        x = np.asarray(design, dtype=float)
        if x.ndim != 2:
            raise ValueError("design must be a 2-D array of coded settings")
        names = tuple(factors) if factors is not None else tuple(FACTOR_LETTERS[: x.shape[1]])
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size != x.shape[0]:
        raise ValueError(f"y must have {x.shape[0]} values (one per run)")
    if not np.all(np.isfinite(yy)):
        raise ValueError("y must contain only finite values")
    k = x.shape[1]
    X = _model_matrix(x)
    p = X.shape[1]
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("design cannot estimate the full quadratic model (rank deficient)")
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    sse = float(resid @ resid)
    sst = float(((yy - yy.mean()) ** 2).sum())
    n = yy.size
    resid_df = n - p
    r2 = 1 - sse / sst if sst > 0 else 1.0
    adj = 1 - (1 - r2) * (n - 1) / resid_df if resid_df > 0 and sst > 0 else float("nan")
    rmse = float(np.sqrt(sse / resid_df)) if resid_df > 0 else float("nan")
    terms = quadratic_terms(names)
    coefs = {t: float(c) for t, c in zip(terms, beta)}
    b = beta[1 : k + 1].copy()
    B = np.diag(beta[k + 1 : 2 * k + 1]).astype(float)
    for (i, j), c in zip(itertools.combinations(range(k), 2), beta[2 * k + 1 :]):
        B[i, j] = B[j, i] = c / 2
    evals, evecs = np.linalg.eigh(B)
    scale = float(np.max(np.abs(evals))) if evals.size else 0.0
    xs = ys = None
    if scale == 0 or np.any(np.abs(evals) < tol * scale):
        nature = "ridge"
    else:
        xs = -0.5 * np.linalg.solve(B, b)
        ys = float(beta[0] + 0.5 * xs @ b)
        if np.all(evals < 0):
            nature = "maximum"
        elif np.all(evals > 0):
            nature = "minimum"
        else:
            nature = "saddle"
    return QuadraticFit(coefficients=coefs, factors=tuple(names), r_squared=float(r2),
                        adj_r_squared=float(adj), rmse=rmse, resid_df=int(resid_df), b=b, B=B,
                        stationary_point=xs, stationary_response=ys, eigenvalues=evals,
                        eigenvectors=evecs, nature=nature)


def scaled_prediction_variance(
    design: Union[ResponseSurfaceDesign, np.ndarray],
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Scaled prediction variance ``N x'(X'X)^{-1} x`` of the quadratic model.

    For a rotatable design this depends only on the distance of ``x`` from
    the design centre.
    """

    x = design.matrix if isinstance(design, ResponseSurfaceDesign) else np.asarray(design, dtype=float)
    X = _model_matrix(x)
    XtX_inv = np.linalg.inv(X.T @ X)
    P = _model_matrix(np.atleast_2d(np.asarray(points, dtype=float)))
    return x.shape[0] * np.einsum("ni,ij,nj->n", P, XtX_inv, P)
