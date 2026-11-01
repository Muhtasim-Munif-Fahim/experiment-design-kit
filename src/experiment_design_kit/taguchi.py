"""Taguchi orthogonal arrays and signal-to-noise (robust design) analysis.

Taguchi's robust parameter design (Taguchi 1986; Phadke, *Quality
Engineering Using Robust Design*, 1989) places control factors in the
columns of an **orthogonal array** (OA). Each run is replicated across noise
conditions, every run is summarized by a **signal-to-noise (S/N) ratio**, and
the factor levels that maximize S/N are chosen.

* :func:`orthogonal_array` builds the Rao-Hamming array
  ``L_{q^k}(q^{(q^k-1)/(q-1)})`` of strength 2 over ``GF(q)`` for any prime
  power ``q <= 32``. Columns follow Taguchi's standard order: each new basic
  column is followed by its interactions with the previous columns. The
  results are ``L4``, ``L8``, ``L16``, ``L32``, ``L9``, ``L27``, ``L81``,
  ``L16(4^5)``, ``L25``, ``L64(4^21)`` and so on. Levels are coded
  ``0..q-1``.
* :func:`taguchi_array` looks up the standard arrays by name, including the
  mixed-level ``L18(2^1 3^7)``. :func:`taguchi_design` picks the smallest
  standard array that fits a list of factor level counts.
* :func:`interaction_columns` gives the columns that carry the interaction
  of two columns (the "triangular table"). Leave them empty, or accept the
  confounding.
* :func:`is_orthogonal_array` checks strength ``t`` (every ``t``-tuple of
  columns shows every level combination equally often).
* :func:`signal_to_noise` computes the larger-the-better,
  smaller-the-better and nominal-the-best S/N ratios in decibels.
  :func:`taguchi_analysis` builds the response tables (level averages of
  S/N and of the mean), delta and rank, the optimal levels, an additive-model
  prediction at the optimum, and an ANOVA on the S/N ratios with percent
  contribution. Error is estimated from unassigned columns, or by pooling
  the smallest factors.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .block_designs import _gf_tables

__all__ = [
    "TAGUCHI_ARRAYS",
    "TaguchiAnalysis",
    "orthogonal_array",
    "taguchi_array",
    "taguchi_design",
    "interaction_columns",
    "is_orthogonal_array",
    "signal_to_noise",
    "taguchi_analysis",
]

# Taguchi's L18 (2^1 x 3^7), levels coded from 1 as in the published table.
_L18 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 3, 3, 3, 3, 3, 3],
        [1, 2, 1, 1, 2, 2, 3, 3],
        [1, 2, 2, 2, 3, 3, 1, 1],
        [1, 2, 3, 3, 1, 1, 2, 2],
        [1, 3, 1, 2, 1, 3, 2, 3],
        [1, 3, 2, 3, 2, 1, 3, 1],
        [1, 3, 3, 1, 3, 2, 1, 2],
        [2, 1, 1, 3, 3, 2, 2, 1],
        [2, 1, 2, 1, 1, 3, 3, 2],
        [2, 1, 3, 2, 2, 1, 1, 3],
        [2, 2, 1, 2, 3, 1, 3, 2],
        [2, 2, 2, 3, 1, 2, 1, 3],
        [2, 2, 3, 1, 2, 3, 2, 1],
        [2, 3, 1, 3, 2, 3, 1, 2],
        [2, 3, 2, 1, 3, 1, 2, 3],
        [2, 3, 3, 2, 1, 2, 3, 1],
    ]
) - 1

# name -> (q, k) for Rao-Hamming arrays, or a fixed table
TAGUCHI_ARRAYS: Dict[str, Tuple] = {
    "L4": (2, 2),
    "L8": (2, 3),
    "L9": (3, 2),
    "L16": (2, 4),
    "L16b": (4, 2),  # L16(4^5)
    "L18": "mixed",  # L18(2^1 3^7)
    "L25": (5, 2),
    "L27": (3, 3),
    "L32": (2, 5),
    "L49": (7, 2),
    "L64b": (4, 3),  # L64(4^21)
    "L81": (3, 4),
}


def _column_vectors(q: int, k: int) -> List[Tuple[int, ...]]:
    """Generator vectors in Taguchi column order (basic column, then its interactions)."""

    _, mul = _gf_tables(q)
    add, _ = _gf_tables(q)
    cols: List[Tuple[int, ...]] = []
    for i in range(k):
        e = tuple(1 if j == i else 0 for j in range(k))
        previous = list(cols)
        cols.append(e)
        for u in previous:
            for c in range(1, q):
                cols.append(tuple(int(add[u[j], mul[c, e[j]]]) for j in range(k)))
    return cols


def orthogonal_array(q: int, k: int) -> np.ndarray:
    """Rao-Hamming orthogonal array ``OA(q^k, (q^k - 1)/(q - 1), q, 2)``.

    Parameters
    ----------
    q : int
        Number of levels: a prime power ``<= 32`` (for example 2, 3, 4, 5, 7, 8 or 9).
    k : int
        Number of basic columns (``k >= 2``). The array has ``q**k`` runs.

    Returns
    -------
    ndarray of shape (q**k, (q**k - 1) // (q - 1))
        Levels coded ``0..q-1``.
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if q ** k > 4096:
        raise ValueError("array too large (q**k > 4096)")
    add, mul = _gf_tables(q)
    rows = np.array(list(itertools.product(range(q), repeat=k)), dtype=int)  # first basic = most significant
    out = np.empty((rows.shape[0], (q ** k - 1) // (q - 1)), dtype=int)
    for c, u in enumerate(_column_vectors(q, k)):
        val = np.zeros(rows.shape[0], dtype=int)
        for j, uj in enumerate(u):
            if uj:
                val = add[val, mul[uj, rows[:, j]]]
        out[:, c] = val
    return out


def taguchi_array(name: str) -> np.ndarray:
    """Standard Taguchi array by name (``"L4"``, ``"L8"``, ``"L9"``, ``"L18"``, ...).

    ``"L16b"`` is ``L16(4^5)`` and ``"L64b"`` is ``L64(4^21)``. Levels are coded ``0..q-1``.
    """

    key = name.strip()
    if key not in TAGUCHI_ARRAYS:
        raise ValueError(f"unknown array {name!r}; choose from {sorted(TAGUCHI_ARRAYS)}")
    spec = TAGUCHI_ARRAYS[key]
    if spec == "mixed":
        return _L18.copy()
    return orthogonal_array(*spec)


def _array_levels(A: np.ndarray) -> List[int]:
    return [int(A[:, j].max()) + 1 for j in range(A.shape[1])]


def taguchi_design(levels: Sequence[int]) -> Tuple[str, np.ndarray, List[int]]:
    """Smallest standard array that accommodates factors with the given level counts.

    Returns ``(name, design, columns)``. ``design`` holds the assigned columns
    in factor order, and ``columns`` gives their indices in the full array.
    Factors are assigned to the earliest columns with the right number of
    levels, so later columns (often interaction columns) stay free for error.
    """

    levels = [int(v) for v in levels]
    if not levels or min(levels) < 2:
        raise ValueError("need at least one factor, each with >= 2 levels")
    candidates = sorted(TAGUCHI_ARRAYS, key=lambda n: (taguchi_array(n).shape[0], n))
    for name in candidates:
        A = taguchi_array(name)
        if A.shape[0] > 4096:
            continue
        lv = _array_levels(A)
        free = list(range(A.shape[1]))
        cols: List[int] = []
        ok = True
        for need in levels:
            match = next((c for c in free if lv[c] == need), None)
            if match is None:
                ok = False
                break
            cols.append(match)
            free.remove(match)
        if ok:
            return name, A[:, cols].copy(), cols
    raise ValueError(f"no standard Taguchi array fits factor levels {levels}")


def interaction_columns(q: int, k: int, i: int, j: int) -> List[int]:
    """Columns of ``orthogonal_array(q, k)`` that carry the ``i x j`` interaction.

    For two-level arrays this is a single column (Taguchi's triangular
    table). For ``q`` levels the interaction spans ``q - 1`` columns.
    """

    vecs = _column_vectors(q, k)
    n = len(vecs)
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise ValueError("i and j must be distinct column indices")
    add, mul = _gf_tables(q)
    index = {v: c for c, v in enumerate(vecs)}

    def normalize(v):
        lead = next(x for x in v if x)
        inv = next(s for s in range(1, q) if mul[lead, s] == 1)
        return tuple(int(mul[inv, x]) for x in v)

    out = []
    for c in range(1, q):
        w = tuple(int(add[a, mul[c, b]]) for a, b in zip(vecs[i], vecs[j]))
        out.append(index[normalize(w)])
    return sorted(set(out))


def is_orthogonal_array(A, strength: int = 2) -> bool:
    """True if every ``strength``-subset of columns is balanced over all level combinations."""

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    n, m = A.shape
    lv = _array_levels(A)
    if np.any(A < 0):
        return False
    for cols in itertools.combinations(range(m), min(strength, m)):
        total = int(np.prod([lv[c] for c in cols]))
        if n % total:
            return False
        _, counts = np.unique(A[:, cols], axis=0, return_counts=True)
        if counts.size != total or np.any(counts != n // total):
            return False
    return True


def signal_to_noise(y, kind: str = "larger") -> np.ndarray:
    """Taguchi S/N ratio in dB for each run (row) of replicated responses.

    ``kind``:

    * ``"larger"`` (larger-the-better): ``-10 log10(mean(1 / y^2))``
    * ``"smaller"`` (smaller-the-better): ``-10 log10(mean(y^2))``
    * ``"nominal"`` (nominal-the-best I): ``10 log10(ybar^2 / s^2)``
    * ``"nominal_variance"`` (nominal-the-best II): ``-10 log10(s^2)``
    """

    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.size == 0 or not np.all(np.isfinite(Y)):
        raise ValueError("y must be a finite (runs x replicates) array")
    if kind == "larger":
        if np.any(Y == 0):
            raise ValueError("larger-the-better S/N needs non-zero responses")
        return -10.0 * np.log10(np.mean(1.0 / Y ** 2, axis=1))
    if kind == "smaller":
        msq = np.mean(Y ** 2, axis=1)
        if np.any(msq == 0):
            raise ValueError("smaller-the-better S/N undefined for all-zero runs")
        return -10.0 * np.log10(msq)
    if kind in ("nominal", "nominal_variance"):
        if Y.shape[1] < 2:
            raise ValueError("nominal-the-best S/N needs >= 2 replicates per run")
        s2 = np.var(Y, axis=1, ddof=1)
        if np.any(s2 == 0):
            raise ValueError("nominal-the-best S/N undefined for zero within-run variance")
        if kind == "nominal_variance":
            return -10.0 * np.log10(s2)
        return 10.0 * np.log10(np.mean(Y, axis=1) ** 2 / s2)
    raise ValueError("kind must be 'larger', 'smaller', 'nominal' or 'nominal_variance'")


@dataclass
class TaguchiAnalysis:
    """Response tables, optimum and S/N ANOVA of a Taguchi experiment."""

    kind: str
    factors: List[str]
    sn: np.ndarray
    means: np.ndarray
    sn_table: Dict[str, np.ndarray]
    mean_table: Dict[str, np.ndarray]
    delta: Dict[str, float]
    rank: Dict[str, int]
    optimal_levels: Dict[str, int]
    predicted_sn: float
    predicted_mean: float
    anova: List[Dict[str, float]] = field(default_factory=list)
    pooled: Tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [f"Taguchi analysis ({self.kind}-the-better S/N, dB)" if not self.kind.startswith("nominal")
                 else f"Taguchi analysis ({self.kind} S/N, dB)"]
        width = max(len(f) for f in self.factors)
        nlev = max(len(v) for v in self.sn_table.values())
        lines.append(f"{'factor':<{width}}  " + "  ".join(f"{'L' + str(i + 1):>8}" for i in range(nlev))
                     + f"  {'delta':>8}  rank  best")
        for f in self.factors:
            cells = [f"{v:>8.3f}" for v in self.sn_table[f]] + [" " * 8] * (nlev - len(self.sn_table[f]))
            row = f"{f:<{width}}  " + "  ".join(cells)
            lines.append(row + f"  {self.delta[f]:>8.3f}  {self.rank[f]:>4}  L{self.optimal_levels[f] + 1}")
        lines.append(f"predicted S/N at optimum: {self.predicted_sn:.3f} dB; predicted mean: {self.predicted_mean:.4f}")
        for r in self.anova:
            f_txt = "" if math.isnan(r["F"]) else f"  F={r['F']:.2f} p={r['p']:.3g}"
            lines.append(f"  {r['source']:<{max(width, 6)}} df={int(r['df'])} SS={r['SS']:.4f} "
                         f"contrib={r['percent']:.1f}%{f_txt}")
        return "\n".join(lines)


def taguchi_analysis(
    design,
    y,
    kind: str = "larger",
    factors: Optional[Sequence[str]] = None,
    pool: int = 0,
) -> TaguchiAnalysis:
    """Analyze a Taguchi experiment.

    Parameters
    ----------
    design : array-like of shape (runs, f)
        Levels (coded ``0..L-1``) of the assigned factors, for example from
        :func:`taguchi_design`. Columns must be balanced and mutually orthogonal.
    y : array-like of shape (runs,) or (runs, replicates)
        Responses. Replicates usually correspond to the noise conditions of an
        outer array.
    kind : str
        S/N type, see :func:`signal_to_noise`.
    factors : sequence of str, optional
        Factor names (default ``A, B, C, ...``).
    pool : int
        Number of the smallest factors (by S/N sum of squares) to pool into
        error when the array leaves no error degrees of freedom.

    Returns
    -------
    TaguchiAnalysis
        The prediction at the optimum uses the additive model
        ``mean + sum(best level avg - mean)`` over the unpooled factors.
    """

    D = np.asarray(design)
    if D.ndim == 1:
        D = D[:, None]
    D = D.astype(int)
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, f = D.shape
    if Y.shape[0] != n:
        raise ValueError("design and y must have the same number of runs")
    if np.any(D < 0):
        raise ValueError("levels must be coded 0..L-1")
    if not is_orthogonal_array(D, strength=2 if f > 1 else 1):
        raise ValueError("design columns must be balanced and pairwise orthogonal")
    names = list(factors) if factors is not None else [chr(ord("A") + i) for i in range(f)]
    if len(names) != f or len(set(names)) != f:
        raise ValueError("need one distinct name per design column")

    sn = signal_to_noise(Y, kind)
    means = Y.mean(axis=1)
    grand_sn = float(sn.mean())
    grand_mean = float(means.mean())

    sn_table, mean_table, delta, ss, df = {}, {}, {}, {}, {}
    for j, name in enumerate(names):
        L = int(D[:, j].max()) + 1
        sn_table[name] = np.array([sn[D[:, j] == lv].mean() for lv in range(L)])
        mean_table[name] = np.array([means[D[:, j] == lv].mean() for lv in range(L)])
        delta[name] = float(sn_table[name].max() - sn_table[name].min())
        ss[name] = float((n / L) * np.sum((sn_table[name] - grand_sn) ** 2))
        df[name] = L - 1
    order = sorted(names, key=lambda nm: -delta[nm])
    rank = {nm: i + 1 for i, nm in enumerate(order)}
    best = {nm: int(np.argmax(sn_table[nm])) for nm in names}

    ss_total = float(np.sum((sn - grand_sn) ** 2))
    df_error = (n - 1) - sum(df.values())
    ss_error = max(ss_total - sum(ss.values()), 0.0)
    if pool < 0 or pool >= f:
        raise ValueError("pool must be in [0, number of factors)")
    pooled = tuple(sorted(names, key=lambda nm: ss[nm])[:pool])
    for nm in pooled:
        ss_error += ss[nm]
        df_error += df[nm]

    active = [nm for nm in names if nm not in pooled]
    pred_sn = grand_sn + sum(sn_table[nm][best[nm]] - grand_sn for nm in active)
    pred_mean = grand_mean + sum(mean_table[nm][best[nm]] - grand_mean for nm in active)

    anova = []
    ms_error = ss_error / df_error if df_error > 0 else float("nan")
    for nm in active:
        if df_error > 0 and ms_error > 0:
            F = (ss[nm] / df[nm]) / ms_error
            p = float(stats.f.sf(F, df[nm], df_error))
        else:
            F = p = float("nan")
        anova.append({"source": nm, "df": float(df[nm]), "SS": ss[nm],
                      "percent": 100.0 * ss[nm] / ss_total if ss_total > 0 else 0.0, "F": float(F), "p": p})
    if df_error > 0:
        anova.append({"source": "error", "df": float(df_error), "SS": ss_error,
                      "percent": 100.0 * ss_error / ss_total if ss_total > 0 else 0.0,
                      "F": float("nan"), "p": float("nan")})
    anova.append({"source": "total", "df": float(n - 1), "SS": ss_total, "percent": 100.0,
                  "F": float("nan"), "p": float("nan")})

    return TaguchiAnalysis(
        kind=kind, factors=names, sn=sn, means=means, sn_table=sn_table, mean_table=mean_table,
        delta=delta, rank=rank, optimal_levels=best, predicted_sn=float(pred_sn),
        predicted_mean=float(pred_mean), anova=anova, pooled=pooled,
    )
