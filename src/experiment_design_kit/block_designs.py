"""Latin squares, Graeco-Latin squares and balanced incomplete block designs.

These are classical row-column and incomplete-block designs (Cochran & Cox;
Montgomery, *Design and Analysis of Experiments*, ch. 4), with their
analyses:

* :func:`latin_square` -- an ``n x n`` Latin square, optionally randomised.
  Rows, columns and symbols are permuted independently, which is Fisher's
  randomisation within the family of squares isotopic to the cyclic one.
  It blocks on two nuisance factors at once.
* :func:`mutually_orthogonal_latin_squares` -- the complete set of ``n - 1``
  MOLS for any prime power ``n <= 32``, built from the finite field
  ``GF(n)``. :func:`graeco_latin_square` superimposes two of them, for three
  blocking factors plus treatment. (``n = 6`` has no Graeco-Latin square,
  Euler's 36 officers problem.)
* :func:`latin_square_anova` -- the ANOVA of a Latin square experiment: row,
  column and treatment sums of squares, with ``(n-1)(n-2)`` error degrees of
  freedom and F tests.
* :func:`balanced_incomplete_block_design` -- a BIBD ``(v, b, r, k, lambda)``.
  It uses the affine plane ``AG(2, q)`` (``v = q^2``, ``k = q``), the
  projective plane ``PG(2, q)`` (``v = q^2+q+1``, ``k = q+1``), the
  complementary design, or all ``k``-subsets when that is small.
  :func:`bibd_parameters` checks a block list.
* :func:`bibd_intrablock_analysis` -- the intrablock (Yates) analysis.
  Adjusted treatment totals are ``Q_i = T_i - B_(i)/k``, effects are
  ``tau_i = k Q_i / (lambda v)``, and the adjusted treatment SS has an F
  test. Efficiency factor ``E = lambda v / (r k)``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "LatinSquareANOVA",
    "BIBD",
    "BIBDAnalysis",
    "latin_square",
    "is_latin_square",
    "are_orthogonal",
    "mutually_orthogonal_latin_squares",
    "graeco_latin_square",
    "latin_square_anova",
    "balanced_incomplete_block_design",
    "bibd_parameters",
    "bibd_intrablock_analysis",
]

# ---------------------------------------------------------------------------
# finite fields GF(p^m) for small orders

# monic irreducible polynomials, coefficients low -> high (excluding leading 1)
_IRREDUCIBLE = {
    4: (2, [1, 1]),        # x^2 + x + 1
    8: (2, [1, 1, 0]),     # x^3 + x + 1
    9: (3, [1, 0]),        # x^2 + 1
    16: (2, [1, 1, 0, 0]),  # x^4 + x + 1
    25: (5, [2, 1]),       # x^2 + x + 2
    27: (3, [1, 2, 0]),    # x^3 + 2x + 1
    32: (2, [1, 0, 1, 0, 0]),  # x^5 + x^2 + 1
}


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(math.isqrt(n)) + 1))


def _gf_tables(q: int) -> Tuple[np.ndarray, np.ndarray]:
    """Addition and multiplication tables of GF(q) (elements 0..q-1)."""

    if _is_prime(q):
        a = np.arange(q)
        return (a[:, None] + a[None, :]) % q, (a[:, None] * a[None, :]) % q
    if q not in _IRREDUCIBLE:
        raise ValueError(f"order {q} is not a supported prime power (primes and {sorted(_IRREDUCIBLE)})")
    p, low = _IRREDUCIBLE[q]
    m = len(low)

    def digits(x):
        return [(x // p**i) % p for i in range(m)]

    def number(d):
        return sum(int(c) * p**i for i, c in enumerate(d))

    def mul(x, y):
        a, b = digits(x), digits(y)
        prod = [0] * (2 * m - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                prod[i + j] = (prod[i + j] + ai * bj) % p
        # reduce x^m = -low
        for deg in range(2 * m - 2, m - 1, -1):
            c = prod[deg]
            if c:
                prod[deg] = 0
                for i, li in enumerate(low):
                    prod[deg - m + i] = (prod[deg - m + i] - c * li) % p
        return number(prod[:m])

    add = np.array([[number([(u + v) % p for u, v in zip(digits(x), digits(y))]) for y in range(q)] for x in range(q)])
    mult = np.array([[mul(x, y) for y in range(q)] for x in range(q)])
    return add, mult


# ---------------------------------------------------------------------------
# Latin squares


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def latin_square(n: int, *, randomize: bool = True, seed=None) -> np.ndarray:
    """An ``n x n`` Latin square with symbols ``0..n-1``.

    The square is cyclic ``L[i, j] = (i + j) mod n``. With ``randomize``
    (the default), rows, columns and symbols are permuted at random.
    """

    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    i = np.arange(n)
    L = (i[:, None] + i[None, :]) % n
    if randomize:
        rng = _rng(seed)
        L = L[rng.permutation(n)][:, rng.permutation(n)]
        L = rng.permutation(n)[L]
    return L


def is_latin_square(square) -> bool:
    L = np.asarray(square)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        return False
    symbols = np.unique(L)
    if symbols.size != L.shape[0]:
        return False
    return all(np.unique(r).size == L.shape[0] for r in L) and all(np.unique(c).size == L.shape[0] for c in L.T)


def are_orthogonal(a, b) -> bool:
    """True when superimposing ``a`` and ``b`` gives every ordered symbol
    pair exactly once."""

    A, B = np.asarray(a), np.asarray(b)
    if A.shape != B.shape or not (is_latin_square(A) and is_latin_square(B)):
        return False
    pairs = set(zip(A.ravel().tolist(), B.ravel().tolist()))
    return len(pairs) == A.size


def mutually_orthogonal_latin_squares(n: int) -> List[np.ndarray]:
    """The complete set of ``n - 1`` MOLS of prime-power order ``n``.

    Square ``k`` (``k = 1..n-1``) is ``L_k[i, j] = a_k * a_i + a_j`` over
    ``GF(n)``.
    """

    if int(n) != n or n < 3:
        raise ValueError("n must be an integer >= 3")
    n = int(n)
    add, mul = _gf_tables(n)
    return [add[mul[k][:, None], np.arange(n)[None, :]] for k in range(1, n)]


def graeco_latin_square(n: int, *, randomize: bool = True, seed=None) -> Tuple[np.ndarray, np.ndarray]:
    """A pair of orthogonal Latin squares ``(latin, greek)`` of order ``n``.

    Prime powers use ``GF(n)``. Randomisation applies the same row and column
    permutations to both squares, plus separate symbol relabellings, so the
    two stay orthogonal. ``n = 2`` and ``n = 6`` have no such pair. Other
    composite orders (10, 12, ...) are not constructed here.
    """

    if n in (2, 6):
        raise ValueError(f"no Graeco-Latin square of order {n} exists")
    A, B = mutually_orthogonal_latin_squares(n)[:2]
    if randomize:
        rng = _rng(seed)
        r, c = rng.permutation(n), rng.permutation(n)
        A, B = A[r][:, c], B[r][:, c]
        A, B = rng.permutation(n)[A], rng.permutation(n)[B]
    return A, B


@dataclass
class LatinSquareANOVA:
    """ANOVA table of a Latin square experiment."""

    table: Dict[str, Dict[str, float]]
    treatment_means: np.ndarray
    grand_mean: float
    residual_df: int
    mse: float

    @property
    def p_value(self) -> float:
        return self.table["treatment"]["p"]

    def summary(self) -> str:
        lines = [f"{'source':<10} {'df':>4} {'SS':>12} {'MS':>12} {'F':>9} {'p':>9}"]
        for src, row in self.table.items():
            f = f"{row['F']:9.3f}" if not math.isnan(row["F"]) else " " * 9
            p = f"{row['p']:9.4g}" if not math.isnan(row["p"]) else " " * 9
            ms = f"{row['MS']:12.4f}" if not math.isnan(row["MS"]) else " " * 12
            lines.append(f"{src:<10} {int(row['df']):>4} {row['SS']:12.4f} {ms} {f} {p}")
        return "\n".join(lines)


def latin_square_anova(y, square) -> LatinSquareANOVA:
    """Row/column/treatment ANOVA for responses ``y[i, j]`` observed in a
    Latin square whose cell ``(i, j)`` received treatment ``square[i, j]``."""

    Y = np.asarray(y, dtype=float)
    L = np.asarray(square)
    if Y.shape != L.shape or Y.ndim != 2:
        raise ValueError("y and square must be matching n x n arrays")
    if not is_latin_square(L):
        raise ValueError("square is not a Latin square")
    if not np.all(np.isfinite(Y)):
        raise ValueError("y must be finite")
    n = Y.shape[0]
    if n < 3:
        raise ValueError("need n >= 3 for a positive error degree of freedom")
    symbols = np.unique(L)
    codes = np.searchsorted(symbols, L)
    gm = Y.mean()
    tmeans = np.array([Y[codes == t].mean() for t in range(n)])
    ss_rows = n * np.sum((Y.mean(axis=1) - gm) ** 2)
    ss_cols = n * np.sum((Y.mean(axis=0) - gm) ** 2)
    ss_trt = n * np.sum((tmeans - gm) ** 2)
    ss_tot = np.sum((Y - gm) ** 2)
    ss_err = max(ss_tot - ss_rows - ss_cols - ss_trt, 0.0)
    df_e = (n - 1) * (n - 2)
    mse = ss_err / df_e
    table = {}
    for name, ss in (("rows", ss_rows), ("columns", ss_cols), ("treatment", ss_trt)):
        ms = ss / (n - 1)
        F = ms / mse if mse > 0 else math.inf
        table[name] = {"df": n - 1, "SS": float(ss), "MS": float(ms), "F": float(F),
                       "p": float(stats.f.sf(F, n - 1, df_e)) if mse > 0 else 0.0}
    nan = float("nan")
    table["error"] = {"df": df_e, "SS": float(ss_err), "MS": float(mse), "F": nan, "p": nan}
    table["total"] = {"df": n * n - 1, "SS": float(ss_tot), "MS": nan, "F": nan, "p": nan}
    return LatinSquareANOVA(table=table, treatment_means=tmeans, grand_mean=float(gm),
                            residual_df=df_e, mse=float(mse))


# ---------------------------------------------------------------------------
# BIBDs


@dataclass
class BIBD:
    """A balanced incomplete block design on treatments ``0..v-1``."""

    blocks: List[Tuple[int, ...]]
    v: int
    b: int
    r: int
    k: int
    lam: int
    construction: str

    @property
    def efficiency(self) -> float:
        """Efficiency factor ``lambda v / (r k)`` relative to a RCBD."""

        return self.lam * self.v / (self.r * self.k)

    def incidence(self) -> np.ndarray:
        N = np.zeros((self.v, self.b), dtype=int)
        for j, blk in enumerate(self.blocks):
            N[list(blk), j] = 1
        return N

    def randomized(self, seed=None) -> "BIBD":
        """Randomly relabel treatments, shuffle block order and the order of
        treatments within each block. This is standard BIBD randomisation."""

        rng = _rng(seed)
        perm = rng.permutation(self.v)
        blocks = [tuple(int(perm[t]) for t in rng.permutation(list(blk))) for blk in self.blocks]
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        return BIBD(blocks, self.v, self.b, self.r, self.k, self.lam, self.construction + "+randomized")


def bibd_parameters(blocks: Sequence[Sequence[int]], v: Optional[int] = None) -> Dict[str, int]:
    """Verify that ``blocks`` form a BIBD and return ``v, b, r, k, lam``.

    Raises ``ValueError`` when blocks have unequal sizes, repeat a treatment,
    or when the replication or pair concurrences are not constant.
    """

    blocks = [tuple(int(t) for t in blk) for blk in blocks]
    if not blocks:
        raise ValueError("no blocks")
    ks = {len(b) for b in blocks}
    if len(ks) != 1:
        raise ValueError("blocks must all have the same size")
    k = ks.pop()
    if any(len(set(b)) != k for b in blocks):
        raise ValueError("a block repeats a treatment")
    v = (max(max(b) for b in blocks) + 1) if v is None else int(v)
    reps = np.zeros(v, dtype=int)
    conc = np.zeros((v, v), dtype=int)
    for blk in blocks:
        for t in blk:
            if not 0 <= t < v:
                raise ValueError("treatment label out of range")
            reps[t] += 1
        for a, c in itertools.combinations(blk, 2):
            conc[a, c] += 1
            conc[c, a] += 1
    if np.unique(reps).size != 1:
        raise ValueError("treatments are not equally replicated")
    off = conc[~np.eye(v, dtype=bool)]
    if off.size and np.unique(off).size != 1:
        raise ValueError("pairs do not occur together equally often")
    lam = int(off[0]) if off.size else 0
    return {"v": v, "b": len(blocks), "r": int(reps[0]), "k": k, "lam": lam}


def _affine_plane(q):
    add, mul = _gf_tables(q)
    pts = {(x, y): x * q + y for x in range(q) for y in range(q)}
    blocks = []
    for m in range(q):
        for c in range(q):
            blocks.append(tuple(sorted(pts[(x, int(add[mul[m, x], c]))] for x in range(q))))
    for c in range(q):
        blocks.append(tuple(sorted(pts[(c, y)] for y in range(q))))
    return blocks


def _projective_plane(q):
    add, mul = _gf_tables(q)
    pts = []
    for x in range(q):
        for y in range(q):
            pts.append((1, x, y))
    for y in range(q):
        pts.append((0, 1, y))
    pts.append((0, 0, 1))

    def dot(a, b):
        s = 0
        for u, w in zip(a, b):
            s = int(add[s, mul[u, w]])
        return s

    return [tuple(i for i, p in enumerate(pts) if dot(line, p) == 0) for line in pts]


def _prime_power(n: int) -> bool:
    return n >= 2 and (_is_prime(n) or n in _IRREDUCIBLE)


def balanced_incomplete_block_design(v: int, k: int, *, max_blocks: int = 500) -> BIBD:
    """Construct a BIBD with ``v`` treatments in blocks of size ``k``.

    Constructions are tried in this order:

    1. the affine plane ``AG(2, k)`` when ``v = k^2``;
    2. the projective plane ``PG(2, k-1)`` when ``v = k^2 - k + 1``;
    3. the complement of a projective plane when ``k = v - q - 1`` and
       ``v = q^2 + q + 1``;
    4. the unreduced design of all ``C(v, k)`` subsets, if there are at most
       ``max_blocks`` of them.

    The prime powers in 1-3 must be supported by the built-in fields. The
    result is checked with :func:`bibd_parameters`.
    """

    v, k = int(v), int(k)
    if not 2 <= k < v:
        raise ValueError("need 2 <= k < v")
    blocks, how = None, None
    q = int(math.isqrt(v))
    if q * q == v and q == k and _prime_power(k):
        blocks, how = _affine_plane(k), f"affine plane AG(2,{k})"
    elif k * k - k + 1 == v and _prime_power(k - 1):
        blocks, how = _projective_plane(k - 1), f"projective plane PG(2,{k - 1})"
    else:
        for qq in range(2, v):
            if qq * qq + qq + 1 == v and k == v - qq - 1 and _prime_power(qq):
                pp = _projective_plane(qq)
                blocks = [tuple(sorted(set(range(v)) - set(b))) for b in pp]
                how = f"complement of PG(2,{qq})"
                break
    if blocks is None:
        if math.comb(v, k) > max_blocks:
            raise ValueError(
                f"no construction available for v={v}, k={k} within max_blocks={max_blocks} "
                f"(unreduced design needs {math.comb(v, k)} blocks)"
            )
        blocks, how = list(itertools.combinations(range(v), k)), "unreduced (all k-subsets)"
    p = bibd_parameters(blocks, v)
    return BIBD(blocks=[tuple(b) for b in blocks], v=p["v"], b=p["b"], r=p["r"], k=p["k"],
                lam=p["lam"], construction=how)


@dataclass
class BIBDAnalysis:
    """Intrablock analysis of a BIBD."""

    effects: np.ndarray
    adjusted_means: np.ndarray
    se_difference: float
    table: Dict[str, Dict[str, float]]
    efficiency: float

    @property
    def p_value(self) -> float:
        return self.table["treatment (adj)"]["p"]


def bibd_intrablock_analysis(blocks: Sequence[Sequence[int]], y: Sequence[Sequence[float]]) -> BIBDAnalysis:
    """Intrablock analysis of responses ``y[j][m]`` for treatment
    ``blocks[j][m]``.

    Returns treatment effects ``tau_i = k Q_i / (lambda v)`` (they sum to
    zero), adjusted means ``grand + tau_i``, the SE of a pairwise difference
    ``sqrt(2 k MSE / (lambda v))``, and the ANOVA table (blocks unadjusted,
    treatments adjusted, error).
    """

    p = bibd_parameters(blocks)
    v, b, r, k, lam = p["v"], p["b"], p["r"], p["k"], p["lam"]
    Y = [np.asarray(row, dtype=float) for row in y]
    if len(Y) != b or any(row.shape != (k,) for row in Y):
        raise ValueError("y must give one response per treatment in each block")
    allv = np.concatenate(Y)
    if not np.all(np.isfinite(allv)):
        raise ValueError("y must be finite")
    N = allv.size
    gm = allv.mean()
    T = np.zeros(v)
    Bsum = np.zeros(v)
    for blk, row in zip(blocks, Y):
        for t, val in zip(blk, row):
            T[t] += val
        for t in blk:
            Bsum[t] += row.sum()
    Q = T - Bsum / k
    tau = k * Q / (lam * v)
    ss_tot = float(np.sum((allv - gm) ** 2))
    ss_blk = float(sum(row.sum() ** 2 for row in Y) / k - allv.sum() ** 2 / N)
    ss_trt = float(k * np.sum(Q**2) / (lam * v))
    ss_err = max(ss_tot - ss_blk - ss_trt, 0.0)
    df_t, df_b = v - 1, b - 1
    df_e = N - v - b + 1
    if df_e <= 0:
        raise ValueError("no error degrees of freedom")
    mse = ss_err / df_e
    F = (ss_trt / df_t) / mse if mse > 0 else math.inf
    nan = float("nan")
    table = {
        "blocks": {"df": df_b, "SS": ss_blk, "MS": ss_blk / df_b if df_b else nan, "F": nan, "p": nan},
        "treatment (adj)": {"df": df_t, "SS": ss_trt, "MS": ss_trt / df_t, "F": float(F),
                            "p": float(stats.f.sf(F, df_t, df_e)) if mse > 0 else 0.0},
        "error": {"df": df_e, "SS": ss_err, "MS": mse, "F": nan, "p": nan},
        "total": {"df": N - 1, "SS": ss_tot, "MS": nan, "F": nan, "p": nan},
    }
    return BIBDAnalysis(effects=tau, adjusted_means=gm + tau,
                        se_difference=float(math.sqrt(2 * k * mse / (lam * v))),
                        table=table, efficiency=lam * v / (r * k))
