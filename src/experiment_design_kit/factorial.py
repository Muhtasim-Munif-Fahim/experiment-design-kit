"""Two-level full and fractional factorial designs and effect analysis.

A/B tests vary one factor. When several factors (price, copy, layout,
...) may matter *and interact*, a ``2^k`` factorial design tests all of them
in one experiment, and a ``2^(k-p)`` fractional factorial runs only
``2^(k-p)`` of the combinations at the cost of aliasing some effects with
each other.

* :func:`full_factorial` builds the coded ``+/-1`` design in standard
  (Yates) order.
* :func:`fractional_factorial` adds generated columns (``"D=ABC"``), or
  uses the minimum-aberration generators from the standard tables, and
  reports the defining relation, word-length pattern, resolution and the
  alias structure of main effects and two-factor interactions.
* :func:`factorial_effects` estimates every main effect and interaction
  (``effect = mean(y | +) - mean(y | -) = 2 * coefficient``). For
  unreplicated designs it applies Lenth's (1989) pseudo standard error
  with the margin of error (ME) and simultaneous margin (SME); with
  replicates or centre-free residual degrees of freedom it uses ordinary
  t tests on the pooled error.
* :func:`fold_over` returns the mirror-image fraction that de-aliases main
  effects from two-factor interactions in a resolution III design.

Factors are named ``A, B, C, ...`` (``I`` is skipped, as in the textbook
notation, because it denotes the identity word).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp

__all__ = [
    "FactorialDesign",
    "FactorialEffects",
    "FACTOR_LETTERS",
    "MINIMUM_ABERRATION_GENERATORS",
    "full_factorial",
    "fractional_factorial",
    "defining_relation",
    "alias_structure",
    "factorial_effects",
    "fold_over",
    "lenth_pse",
]

FACTOR_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

#: Minimum-aberration generators for common ``2^(k-p)`` fractions
#: (Box, Hunter & Hunter 2005; Montgomery, Design and Analysis of Experiments, Table 8.14).
MINIMUM_ABERRATION_GENERATORS: Dict[Tuple[int, int], Tuple[str, ...]] = {
    (3, 1): ("C=AB",),
    (4, 1): ("D=ABC",),
    (5, 1): ("E=ABCD",),
    (5, 2): ("D=AB", "E=AC"),
    (6, 1): ("F=ABCDE",),
    (6, 2): ("E=ABC", "F=BCD"),
    (6, 3): ("D=AB", "E=AC", "F=BC"),
    (7, 1): ("G=ABCDEF",),
    (7, 2): ("F=ABCD", "G=ABDE"),
    (7, 3): ("E=ABC", "F=BCD", "G=ACD"),
    (7, 4): ("D=AB", "E=AC", "F=BC", "G=ABC"),
    (8, 2): ("G=ABCD", "H=ABEF"),
    (8, 3): ("F=ABC", "G=ABD", "H=BCDE"),
    (8, 4): ("E=BCD", "F=ACD", "G=ABC", "H=ABD"),
}


def _letters(k: int) -> str:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("number of factors must be a positive integer")
    if k > len(FACTOR_LETTERS):
        raise ValueError(f"at most {len(FACTOR_LETTERS)} factors are supported")
    return FACTOR_LETTERS[: int(k)]


def _canonical(word: str, letters: str) -> str:
    """Product of factor letters with squares cancelling (A*A = I)."""

    counts: Dict[str, int] = {}
    for ch in word:
        if ch not in letters:
            raise ValueError(f"unknown factor {ch!r} in word {word!r}")
        counts[ch] = counts.get(ch, 0) ^ 1
    return "".join(ch for ch in letters if counts.get(ch))


def _multiply(left: str, right: str, letters: str) -> str:
    return _canonical(left + right, letters)


@dataclass(frozen=True)
class FactorialDesign:
    """A coded two-level design (rows are runs, columns are factors)."""

    matrix: np.ndarray
    factors: Tuple[str, ...]
    generators: Tuple[str, ...] = ()
    defining_words: Tuple[str, ...] = ()
    resolution: Optional[int] = None
    word_length_pattern: Tuple[int, ...] = ()
    aliases: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def k(self) -> int:
        return len(self.factors)

    @property
    def p(self) -> int:
        return len(self.generators)

    @property
    def resolution_roman(self) -> str:
        if self.resolution is None:
            return "full"
        numerals = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
                    9: "IX", 10: "X"}
        return numerals.get(self.resolution, str(self.resolution))

    def to_levels(self, levels: Mapping[str, Tuple[object, object]]) -> List[Dict[str, object]]:
        """Translate coded runs to real settings ``{factor: (low, high)}``."""

        missing = [f for f in self.factors if f not in levels]
        if missing:
            raise ValueError(f"levels are missing for factors {missing}")
        rows = []
        for run in self.matrix:
            rows.append({f: levels[f][1] if v > 0 else levels[f][0] for f, v in zip(self.factors, run)})
        return rows

    def randomized_order(self, seed: Optional[int] = None) -> np.ndarray:
        """A random run order (indices into ``matrix``), as the protocol should use."""

        return np.random.default_rng(seed).permutation(self.n_runs)


def full_factorial(k: int) -> FactorialDesign:
    """``2^k`` design in standard order: factor ``A`` alternates fastest."""

    letters = _letters(k)
    n = 2 ** len(letters)
    runs = np.arange(n)[:, None]
    matrix = np.where((runs >> np.arange(len(letters))) & 1, 1, -1).astype(int)
    return FactorialDesign(matrix=matrix, factors=tuple(letters))


def _parse_generator(text: str, letters: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"generator {text!r} must look like 'D=ABC'")
    left, right = (part.strip().upper() for part in text.split("=", 1))
    sign = ""
    if right.startswith("-"):
        sign, right = "-", right[1:]
    elif right.startswith("+"):
        right = right[1:]
    if len(left) != 1 or left not in letters:
        raise ValueError(f"generator {text!r} must define a single factor letter")
    word = _canonical(right, letters)
    if len(word) < 2 or len(word) != len(right):
        raise ValueError(f"generator {text!r} needs at least two distinct base factors")
    return left, sign + word


def defining_relation(generators: Sequence[str], k: int) -> List[str]:
    """All words of the defining relation ``I = ...`` (signed, e.g. ``'-ABD'``)."""

    letters = _letters(k)
    words = []
    for gen in generators:
        left, right = _parse_generator(gen, letters)
        sign = "-" if right.startswith("-") else ""
        words.append((sign, _multiply(left, right.lstrip("-"), letters)))
    relation = []
    for size in range(1, len(words) + 1):
        for combo in itertools.combinations(words, size):
            negative = sum(s == "-" for s, _ in combo) % 2 == 1
            word = ""
            for _, w in combo:
                word = _multiply(word, w, letters)
            relation.append(("-" if negative else "") + word)
    return sorted(relation, key=lambda w: (len(w.lstrip("-")), w.lstrip("-")))


def alias_structure(defining_words: Sequence[str], k: int, max_order: int = 2) -> Dict[str, Tuple[str, ...]]:
    """Aliases of every effect up to ``max_order`` (main effects and 2FIs by default).

    Each effect ``E`` is aliased with ``E * W`` for every defining word
    ``W``; aliases are listed shortest first, with the sign of the word.
    """

    letters = _letters(k)
    out: Dict[str, Tuple[str, ...]] = {}
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(letters, order):
            effect = "".join(combo)
            aliases = []
            for word in defining_words:
                sign = "-" if word.startswith("-") else ""
                aliases.append(sign + _multiply(effect, word.lstrip("-"), letters))
            out[effect] = tuple(sorted(aliases, key=lambda w: (len(w.lstrip("-")), w.lstrip("-"))))
    return out


def fractional_factorial(k: int, generators: Optional[Sequence[str]] = None, p: Optional[int] = None) -> FactorialDesign:
    """``2^(k-p)`` fractional factorial from generators like ``("D=AB", "E=-AC")``.

    The first ``k - p`` factors form a full factorial; each generated factor
    is the (optionally negated) product of base columns. Pass ``p`` instead
    of ``generators`` to use the minimum-aberration table.
    """

    letters = _letters(k)
    if generators is None:
        if p is None:
            raise ValueError("pass generators or p")
        key = (int(k), int(p))
        if key not in MINIMUM_ABERRATION_GENERATORS:
            known = sorted(MINIMUM_ABERRATION_GENERATORS)
            raise ValueError(f"no tabulated generators for 2^({k}-{p}); known (k, p): {known}")
        generators = MINIMUM_ABERRATION_GENERATORS[key]
    generators = tuple(generators)
    if p is not None and p != len(generators):
        raise ValueError("p does not match the number of generators")
    p = len(generators)
    if p == 0:
        return full_factorial(k)
    if p >= len(letters):
        raise ValueError("need at least one base factor (p < k)")
    base = letters[: len(letters) - p]
    parsed = [_parse_generator(g, letters) for g in generators]
    defined = [left for left, _ in parsed]
    if sorted(defined) != sorted(letters[len(base):]):
        raise ValueError(
            f"generators must define exactly the last p factors {list(letters[len(base):])}"
        )
    for left, right in parsed:
        bad = set(right.lstrip("-")) - set(base)
        if bad:
            raise ValueError(f"generator for {left} uses non-base factors {sorted(bad)}")
    core = full_factorial(len(base)).matrix
    columns = {f: core[:, i] for i, f in enumerate(base)}
    for left, right in parsed:
        col = np.ones(core.shape[0], dtype=int)
        for ch in right.lstrip("-"):
            col = col * columns[ch]
        columns[left] = -col if right.startswith("-") else col
    matrix = np.column_stack([columns[f] for f in letters])
    words = defining_relation(generators, k)
    lengths = [len(w.lstrip("-")) for w in words]
    wlp = tuple(lengths.count(L) for L in range(3, len(letters) + 1))
    resolution = min(lengths)
    if len({tuple(r) for r in matrix}) != matrix.shape[0]:
        raise ValueError("generators produce duplicate runs")
    return FactorialDesign(
        matrix=matrix,
        factors=tuple(letters),
        generators=tuple(f"{l}={r}" for l, r in parsed),
        defining_words=tuple(words),
        resolution=resolution,
        word_length_pattern=wlp,
        aliases=alias_structure(words, k),
    )


def fold_over(design: FactorialDesign, factor: Optional[str] = None) -> FactorialDesign:
    """Mirror-image fold-over: append the runs with all signs (or one factor) reversed.

    The full fold-over of a resolution III fraction breaks every alias
    between a main effect and a two-factor interaction (result at least
    resolution IV); folding on a single factor de-aliases that factor and
    all its two-factor interactions.
    """

    flipped = -design.matrix.copy()
    if factor is not None:
        if factor not in design.factors:
            raise ValueError(f"unknown factor {factor!r}")
        flipped = design.matrix.copy()
        j = design.factors.index(factor)
        flipped[:, j] *= -1
    matrix = np.vstack([design.matrix, flipped])
    letters = "".join(design.factors)
    # surviving defining words: those whose sign is unchanged by the flip
    words = []
    for word in design.defining_words:
        body = word.lstrip("-")
        flips = len(body) if factor is None else int(factor in body)
        if flips % 2 == 0:
            words.append(word)
    lengths = [len(w.lstrip("-")) for w in words]
    resolution = min(lengths) if lengths else None
    wlp = tuple(lengths.count(L) for L in range(3, len(letters) + 1)) if lengths else ()
    return FactorialDesign(
        matrix=matrix,
        factors=design.factors,
        generators=design.generators,
        defining_words=tuple(words),
        resolution=resolution,
        word_length_pattern=wlp,
        aliases=alias_structure(words, len(letters)) if words else {},
    )


def lenth_pse(effects: Sequence[float]) -> float:
    """Lenth's pseudo standard error ``1.5 * median(|c| : |c| < 2.5 s0)``, ``s0 = 1.5 median|c|``."""

    c = np.abs(np.asarray(effects, dtype=float))
    if c.size < 2:
        raise ValueError("Lenth's PSE needs at least two effects")
    s0 = 1.5 * float(np.median(c))
    trimmed = c[c < 2.5 * s0]
    if trimmed.size == 0:
        return 0.0
    return 1.5 * float(np.median(trimmed))


@dataclass(frozen=True)
class FactorialEffects:
    """Estimated effects of a two-level design."""

    effects: Dict[str, float]
    intercept: float
    method: str
    std_error: Optional[float]
    df: Optional[float]
    p_values: Dict[str, float]
    margin_of_error: Optional[float] = None
    simultaneous_margin: Optional[float] = None
    significant: Tuple[str, ...] = ()
    aliases: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def ranked(self) -> List[Tuple[str, float]]:
        """Effects sorted by absolute size (largest first), e.g. for a Pareto/half-normal plot."""

        return sorted(self.effects.items(), key=lambda kv: -abs(kv[1]))

    def half_normal_scores(self) -> List[Tuple[str, float, float]]:
        """``(effect, |effect|, half-normal quantile)`` for a Daniel half-normal plot."""

        ranked = sorted(self.effects.items(), key=lambda kv: abs(kv[1]))
        m = len(ranked)
        return [
            (name, abs(value), float(sp.norm.ppf(0.5 + 0.5 * (i + 0.5) / m)))
            for i, (name, value) in enumerate(ranked)
        ]


def _estimable_terms(design: FactorialDesign, max_order: Optional[int]) -> List[str]:
    """One representative per alias class (shortest word), up to ``max_order``."""

    letters = "".join(design.factors)
    words = [w.lstrip("-") for w in design.defining_words]
    seen = set()
    terms = []
    limit = len(letters) if max_order is None else max_order
    for order in range(1, limit + 1):
        for combo in itertools.combinations(letters, order):
            term = "".join(combo)
            if term in seen:
                continue
            klass = {term} | {_multiply(term, w, letters) for w in words}
            if "" in klass:
                continue  # aliased with the intercept
            seen |= klass
            terms.append(term)
    return terms


def factorial_effects(
    design: FactorialDesign,
    y: Sequence[float],
    *,
    alpha: float = 0.05,
    max_order: Optional[int] = None,
) -> FactorialEffects:
    """Estimate main effects and interactions from responses ``y``.

    ``y`` has one response per design run, or a ``(runs, replicates)``
    array. Each estimable alias class is represented by its shortest word.
    With no residual degrees of freedom (an unreplicated saturated design)
    Lenth's PSE is used: effects whose ``|effect|`` exceeds the margin of
    error ``ME = t_{1-alpha/2, m/3} * PSE`` are flagged, and SME is the
    corresponding simultaneous margin. Otherwise ``se = 2 * s / sqrt(N)``
    from the residual mean square and t tests are reported.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    responses = np.asarray(y, dtype=float)
    if responses.ndim == 1:
        responses = responses[:, None]
    if responses.ndim != 2 or responses.shape[0] != design.n_runs:
        raise ValueError(f"y must have {design.n_runs} rows (one per design run)")
    if not np.all(np.isfinite(responses)):
        raise ValueError("y must contain only finite values")
    reps = responses.shape[1]
    terms = _estimable_terms(design, max_order)
    index = {f: i for i, f in enumerate(design.factors)}
    columns = []
    for term in terms:
        col = np.ones(design.n_runs)
        for ch in term:
            col = col * design.matrix[:, index[ch]]
        columns.append(col)
    X_runs = np.column_stack([np.ones(design.n_runs)] + columns)
    X = np.repeat(X_runs, reps, axis=0)
    yy = responses.reshape(-1)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        raise ValueError("design columns are not linearly independent")
    effects = {term: float(2.0 * b) for term, b in zip(terms, beta[1:])}
    resid_df = yy.size - X.shape[1]
    aliases = {t: design.aliases.get(t, ()) for t in terms if design.aliases.get(t)}
    if resid_df > 0:
        resid = yy - X @ beta
        s2 = float(resid @ resid) / resid_df
        se = 2.0 * np.sqrt(s2 / yy.size)
        p_values = {t: float(2 * sp.t.sf(abs(e) / se, resid_df)) if se > 0 else 0.0
                    for t, e in effects.items()}
        significant = tuple(t for t, pv in p_values.items() if pv < alpha)
        return FactorialEffects(effects, float(beta[0]), "t-test", float(se), float(resid_df),
                                p_values, significant=significant, aliases=aliases)
    m = len(effects)
    if m < 3:
        raise ValueError("Lenth's method needs at least three effects")
    pse = lenth_pse(list(effects.values()))
    d = m / 3.0
    me = float(sp.t.ppf(1 - alpha / 2, d) * pse)
    gamma = (1 + (1 - alpha) ** (1.0 / m)) / 2
    sme = float(sp.t.ppf(gamma, d) * pse)
    if pse > 0:
        p_values = {t: float(2 * sp.t.sf(abs(e) / pse, d)) for t, e in effects.items()}
    else:
        p_values = {t: (0.0 if e != 0 else 1.0) for t, e in effects.items()}
    significant = tuple(t for t, e in effects.items() if abs(e) > me)
    return FactorialEffects(effects, float(beta[0]), "lenth", float(pse), d, p_values,
                            margin_of_error=me, simultaneous_margin=sme,
                            significant=significant, aliases=aliases)
