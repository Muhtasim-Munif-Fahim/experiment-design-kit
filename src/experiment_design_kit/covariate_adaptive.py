"""Sequential covariate-adaptive allocation: Efron's biased coin and Pocock-Simon minimization.

Stratified and blocked randomization (:mod:`experiment_design_kit.randomization`)
need the whole sample up front or break down with many strata. In trials and
experiments where units arrive one at a time, *dynamic* allocation keeps arms
balanced as they come in:

* **Efron's biased coin** (Efron 1971): with ``D`` = (#treated - #control) so
  far, the next unit goes to the under-represented arm with probability ``p``
  (2/3 by default) and to either arm with probability 1/2 when ``D = 0``. The
  imbalance ``|D|`` stays bounded in probability instead of growing like
  ``sqrt(n)``.
* **Pocock-Simon minimization** (Pocock & Simon 1975): for a new unit with
  categorical covariates, compute for every candidate arm the imbalance the
  arms *would* have on each of the unit's covariate levels (range, variance
  or standard deviation of the per-arm counts), take a weighted sum over
  factors, and give the arm with the smallest total imbalance probability
  ``p_best``; remaining arms share the rest. Ties are broken at random.
  This balances many factors' *margins* at once, which full stratification
  cannot do with small samples.

:func:`marginal_imbalance` reports the per-level count spread, so
allocations can be compared with complete randomization.

References
----------
Efron, B. (1971). Forcing a sequential experiment to be balanced.
*Biometrika*, 58(3), 403-417.
Pocock, S. J. & Simon, R. (1975). Sequential treatment assignment with
balancing for prognostic factors in the controlled clinical trial.
*Biometrics*, 31(1), 103-115.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "BiasedCoinResult",
    "MinimizationResult",
    "efron_biased_coin",
    "pocock_simon_minimization",
    "marginal_imbalance",
]

_MEASURES = ("range", "variance", "sd")


@dataclass(frozen=True)
class BiasedCoinResult:
    """Efron biased-coin allocation of ``n`` sequential units to two arms."""

    assignment: np.ndarray  # 1 = treatment, 0 = control
    imbalance_path: np.ndarray  # D after each unit
    p: float

    @property
    def final_imbalance(self) -> int:
        return int(self.imbalance_path[-1]) if self.imbalance_path.size else 0

    @property
    def max_abs_imbalance(self) -> int:
        return int(np.max(np.abs(self.imbalance_path))) if self.imbalance_path.size else 0


@dataclass(frozen=True)
class MinimizationResult:
    """Pocock-Simon allocation and its balance diagnostics."""

    assignment: np.ndarray  # arm index 0..n_arms-1 per unit
    n_arms: int
    arm_sizes: np.ndarray
    p_best: float
    measure: str
    weights: dict
    deterministic_fraction: float  # share of units whose minimizing arm was unique
    imbalance: dict = field(default_factory=dict)  # output of marginal_imbalance


def _check_prob(value: float, name: str, low: float = 0.5) -> float:
    value = float(value)
    if not low <= value <= 1.0:
        raise ValueError(f"{name} must be in [{low}, 1]")
    return value


def efron_biased_coin(n: int, p: float = 2.0 / 3.0, seed: int | None = None) -> BiasedCoinResult:
    """Allocate ``n`` arriving units with Efron's biased coin (``p`` in [0.5, 1])."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    p = _check_prob(p, "p")
    rng = np.random.default_rng(seed)
    assignment = np.empty(int(n), dtype=int)
    path = np.empty(int(n), dtype=int)
    d = 0
    for i in range(int(n)):
        if d == 0:
            prob_t = 0.5
        elif d < 0:
            prob_t = p
        else:
            prob_t = 1.0 - p
        a = int(rng.random() < prob_t)
        assignment[i] = a
        d += 1 if a else -1
        path[i] = d
    return BiasedCoinResult(assignment=assignment, imbalance_path=path, p=p)


def _as_factor_frame(covariates) -> pd.DataFrame:
    if isinstance(covariates, pd.DataFrame):
        frame = covariates.copy()
    else:
        arr = np.asarray(covariates, dtype=object)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("covariates must be 2-D (units x factors)")
        frame = pd.DataFrame(arr, columns=[f"factor{j}" for j in range(arr.shape[1])])
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("covariates must have at least one unit and one factor")
    if frame.isna().any().any():
        raise ValueError("covariates must not contain missing values")
    return frame


def _spread(counts: np.ndarray, measure: str) -> np.ndarray:
    """Imbalance of per-arm counts along the last axis."""
    if measure == "range":
        return counts.max(axis=-1) - counts.min(axis=-1)
    if measure == "variance":
        return counts.var(axis=-1, ddof=1) if counts.shape[-1] > 1 else np.zeros(counts.shape[:-1])
    return counts.std(axis=-1, ddof=1) if counts.shape[-1] > 1 else np.zeros(counts.shape[:-1])


def pocock_simon_minimization(
    covariates,
    n_arms: int = 2,
    *,
    weights=None,
    p_best: float = 0.8,
    measure: str = "range",
    allocation_ratio=None,
    seed: int | None = None,
) -> MinimizationResult:
    """Pocock-Simon minimization over categorical factors, in arrival order.

    Parameters
    ----------
    covariates :
        DataFrame (or 2-D array) with one row per unit in arrival order and
        one categorical column per prognostic factor.
    n_arms :
        Number of arms (>= 2).
    weights :
        Per-factor weights (mapping column -> weight or sequence); default 1.
    p_best :
        Probability of assigning the arm with the least imbalance
        (``1`` = deterministic minimization; ``1/n_arms`` = complete
        randomization). Remaining probability is split evenly.
    measure :
        ``"range"`` (Pocock & Simon's default), ``"variance"`` or ``"sd"``.
    allocation_ratio :
        Optional target ratio per arm (e.g. ``(2, 1)``); counts are divided
        by it before measuring imbalance, so minimization targets that ratio.
    """
    frame = _as_factor_frame(covariates)
    if isinstance(n_arms, bool) or not isinstance(n_arms, (int, np.integer)) or n_arms < 2:
        raise ValueError("n_arms must be an integer >= 2")
    k = int(n_arms)
    p_best = _check_prob(p_best, "p_best", low=1.0 / k)
    if measure not in _MEASURES:
        raise ValueError(f"measure must be one of {_MEASURES}")
    cols = list(frame.columns)
    if weights is None:
        w = np.ones(len(cols))
    elif isinstance(weights, dict):
        unknown = set(weights) - set(cols)
        if unknown:
            raise ValueError(f"weights for unknown factor(s): {sorted(map(str, unknown))}")
        w = np.array([float(weights.get(c, 1.0)) for c in cols])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(cols),):
            raise ValueError("weights must have one entry per factor")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    if allocation_ratio is None:
        ratio = np.ones(k)
    else:
        ratio = np.asarray(allocation_ratio, dtype=float)
        if ratio.shape != (k,) or np.any(ratio <= 0):
            raise ValueError("allocation_ratio must have n_arms positive entries")

    rng = np.random.default_rng(seed)
    codes = []
    n_levels = []
    for c in cols:
        cat = pd.Categorical(frame[c])
        codes.append(np.asarray(cat.codes))
        n_levels.append(len(cat.categories))
    counts = [np.zeros((L, k)) for L in n_levels]
    n = frame.shape[0]
    assignment = np.empty(n, dtype=int)
    unique_best = 0
    for i in range(n):
        totals = np.zeros(k)
        for f in range(len(cols)):
            row = counts[f][codes[f][i]]  # current counts at this unit's level
            hypo = np.tile(row, (k, 1)) + np.eye(k)  # if assigned to arm a
            totals += w[f] * _spread(hypo / ratio, measure)
        best = np.flatnonzero(np.isclose(totals, totals.min()))
        if best.size == k:
            probs = np.full(k, 1.0 / k)
        else:
            chosen = int(rng.choice(best))
            unique_best += int(best.size == 1)
            probs = np.full(k, (1.0 - p_best) / (k - 1))
            probs[chosen] = p_best
        arm = int(rng.choice(k, p=probs))
        assignment[i] = arm
        for f in range(len(cols)):
            counts[f][codes[f][i], arm] += 1
    return MinimizationResult(
        assignment=assignment,
        n_arms=k,
        arm_sizes=np.bincount(assignment, minlength=k),
        p_best=p_best,
        measure=measure,
        weights={c: float(v) for c, v in zip(cols, w)},
        deterministic_fraction=unique_best / n,
        imbalance=marginal_imbalance(frame, assignment, k),
    )


def marginal_imbalance(covariates, assignment, n_arms: int | None = None) -> dict:
    """Per-factor, per-level arm counts and their range.

    Returns ``{"factors": {factor: {level: {"counts": [...], "range": r}}},
    "total_range": sum of all level ranges, "max_range": largest level range,
    "overall_range": range of the arm sizes}``.
    """
    frame = _as_factor_frame(covariates)
    a = np.asarray(assignment, dtype=int)
    if a.shape != (frame.shape[0],):
        raise ValueError("assignment must have one entry per unit")
    if a.min() < 0:
        raise ValueError("assignment must contain arm indices >= 0")
    k = int(a.max()) + 1 if n_arms is None else int(n_arms)
    if a.max() >= k:
        raise ValueError("assignment contains an arm index >= n_arms")
    factors: dict = {}
    total = 0
    worst = 0
    for c in frame.columns:
        levels: dict = {}
        for level, idx in frame.groupby(c, sort=True).indices.items():
            cnt = np.bincount(a[idx], minlength=k)
            r = int(cnt.max() - cnt.min())
            levels[level.item() if hasattr(level, "item") else level] = {"counts": cnt.tolist(), "range": r}
            total += r
            worst = max(worst, r)
        factors[c] = levels
    sizes = np.bincount(a, minlength=k)
    return {
        "factors": factors,
        "total_range": int(total),
        "max_range": int(worst),
        "overall_range": int(sizes.max() - sizes.min()),
    }
