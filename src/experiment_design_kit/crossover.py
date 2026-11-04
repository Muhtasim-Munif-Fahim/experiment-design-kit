"""Crossover designs: Williams squares and the AB/BA (2x2) crossover trial.

In a crossover each subject gets *several* treatments in sequence, so it
acts as its own control. Between-subject variance drops out of treatment
contrasts, which often cuts the sample size several-fold. The price is
**carryover**: a treatment's effect can leak into the next period.

* :func:`williams_design` -- Williams (1949) squares. Each treatment
  appears once per period and once per sequence, and each treatment is
  *immediately preceded by every other treatment equally often*, so
  first-order carryover is balanced. For even ``t`` one ``t x t`` square
  is enough. For odd ``t`` the square and its mirror image are used
  (``2t`` sequences).
* :func:`carryover_balance` / :func:`is_carryover_balanced` -- count the
  ordered (predecessor, successor) pairs in any design.
* :func:`ab_ba_crossover_analysis` -- the classical analysis of the 2x2
  crossover (Hills & Armitage, 1979; Senn, *Cross-over Trials in Clinical
  Research*). With period differences ``d = y1 - y2``, the treatment effect
  (A - B) is ``(mean d_AB - mean d_BA) / 2`` and the period effect
  (period 2 - period 1) is ``-(mean d_AB + mean d_BA) / 2``, each with a
  pooled two-sample t test on ``n_AB + n_BA - 2`` degrees of freedom. Grizzle's carryover test compares
  the subject totals ``y1 + y2`` between sequences. It is under-powered,
  so it is traditionally run at alpha = 0.10, and a significant result
  should prompt caution rather than a two-stage analysis.
* :func:`crossover_sample_size` -- total subjects for a 2x2 crossover,
  given the within-subject SD ``sigma_w`` (``Var(y1 - y2) = 2 sigma_w^2``).
  It solves ``N`` by t-distribution iteration (df = N - 2), with equal
  sequence sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "williams_design",
    "carryover_balance",
    "is_carryover_balanced",
    "CrossoverAnalysis",
    "ab_ba_crossover_analysis",
    "CrossoverSampleSize",
    "crossover_sample_size",
]


def williams_design(t: int) -> np.ndarray:
    """Williams design for ``t`` treatments: rows = sequences, cols = periods."""
    if int(t) != t or t < 2:
        raise ValueError("t must be an integer >= 2")
    t = int(t)
    first = [0]
    lo, hi = 1, t - 1
    take_lo = True
    while len(first) < t:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    base = np.array([[(x + i) % t for x in first] for i in range(t)], dtype=int)
    if t % 2 == 0:
        return base
    return np.vstack([base, base[:, ::-1]])


def carryover_balance(design) -> np.ndarray:
    """Matrix ``M[a, b]`` = times treatment ``b`` immediately follows ``a``."""
    D = np.asarray(design, dtype=int)
    if D.ndim != 2 or D.shape[1] < 2:
        raise ValueError("design must be 2-D with at least two periods")
    t = int(D.max()) + 1
    if D.min() < 0:
        raise ValueError("treatments must be labelled 0..t-1")
    M = np.zeros((t, t), dtype=int)
    for row in D:
        for a, b in zip(row[:-1], row[1:]):
            M[a, b] += 1
    return M


def is_carryover_balanced(design) -> bool:
    """True if no treatment follows itself and all ordered pairs are equally frequent."""
    D = np.asarray(design, dtype=int)
    t = int(D.max()) + 1
    for col in D.T:
        if not np.array_equal(np.sort(np.bincount(col, minlength=t)), np.full(t, D.shape[0] // t)) or D.shape[0] % t:
            return False
    M = carryover_balance(D)
    off = M[~np.eye(t, dtype=bool)]
    return bool(np.all(np.diag(M) == 0) and off.size and np.all(off == off[0]) and off[0] > 0)


@dataclass(frozen=True)
class CrossoverAnalysis:
    n_ab: int
    n_ba: int
    treatment_effect: float
    treatment_se: float
    treatment_t: float
    treatment_p: float
    treatment_ci: Tuple[float, float]
    period_effect: float
    period_p: float
    carryover_difference: float
    carryover_p: float
    carryover_alpha: float
    carryover_flag: bool
    within_subject_sd: float
    df: int


def _pooled_two_sample(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    nx, ny = x.size, y.size
    df = nx + ny - 2
    sp2 = (((x - x.mean()) ** 2).sum() + ((y - y.mean()) ** 2).sum()) / df
    se = math.sqrt(sp2 * (1.0 / nx + 1.0 / ny))
    return float(x.mean() - y.mean()), se, df


def ab_ba_crossover_analysis(
    ab_period1: Sequence[float],
    ab_period2: Sequence[float],
    ba_period1: Sequence[float],
    ba_period2: Sequence[float],
    *,
    alpha: float = 0.05,
    carryover_alpha: float = 0.10,
) -> CrossoverAnalysis:
    """Analyse an AB/BA crossover. Sequence AB gets A in period 1, B in period 2."""
    a1, a2 = np.asarray(ab_period1, float), np.asarray(ab_period2, float)
    b1, b2 = np.asarray(ba_period1, float), np.asarray(ba_period2, float)
    if a1.shape != a2.shape or b1.shape != b2.shape or a1.ndim != 1 or b1.ndim != 1:
        raise ValueError("period vectors must be 1-D and paired within each sequence")
    if a1.size < 2 or b1.size < 2:
        raise ValueError("need at least two subjects per sequence")
    if not (np.isfinite(a1).all() and np.isfinite(a2).all() and np.isfinite(b1).all() and np.isfinite(b2).all()):
        raise ValueError("responses must be finite")
    if not 0 < alpha < 1 or not 0 < carryover_alpha < 1:
        raise ValueError("alpha levels must be in (0, 1)")
    d_ab, d_ba = a1 - a2, b1 - b2
    diff, se_diff, df = _pooled_two_sample(d_ab, d_ba)
    tau, se = diff / 2.0, se_diff / 2.0
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    t_tau = tau / se if se > 0 else math.copysign(math.inf, tau) if tau else 0.0
    p_tau = float(2 * stats.t.sf(abs(t_tau), df))
    # Period effect: compare d_AB with -d_BA.
    pdiff, pse, _ = _pooled_two_sample(d_ab, -d_ba)
    pi = -pdiff / 2.0  # period 2 minus period 1
    t_pi = pi / (pse / 2.0) if pse > 0 else 0.0
    p_pi = float(2 * stats.t.sf(abs(t_pi), df))
    # Grizzle carryover test on subject totals.
    cdiff, cse, _ = _pooled_two_sample(a1 + a2, b1 + b2)
    t_c = cdiff / cse if cse > 0 else 0.0
    p_c = float(2 * stats.t.sf(abs(t_c), df))
    sw = math.sqrt((((d_ab - d_ab.mean()) ** 2).sum() + ((d_ba - d_ba.mean()) ** 2).sum()) / df / 2.0)
    return CrossoverAnalysis(
        n_ab=int(a1.size),
        n_ba=int(b1.size),
        treatment_effect=float(tau),
        treatment_se=float(se),
        treatment_t=float(t_tau),
        treatment_p=p_tau,
        treatment_ci=(float(tau - tcrit * se), float(tau + tcrit * se)),
        period_effect=float(pi),
        period_p=p_pi,
        carryover_difference=float(cdiff),
        carryover_p=p_c,
        carryover_alpha=float(carryover_alpha),
        carryover_flag=bool(p_c < carryover_alpha),
        within_subject_sd=float(sw),
        df=int(df),
    )


@dataclass(frozen=True)
class CrossoverSampleSize:
    n_total: int
    n_per_sequence: int
    achieved_power: float
    parallel_n_total: Optional[int] = None


def _crossover_power(m: int, delta: float, sigma_w: float, alpha: float) -> float:
    df = 2 * m - 2
    ncp = abs(delta) / (sigma_w / math.sqrt(m))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def crossover_sample_size(
    delta: float,
    sigma_w: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    sigma_b: Optional[float] = None,
) -> CrossoverSampleSize:
    """Subjects for a 2x2 crossover to detect a treatment difference ``delta``.

    ``Var(treatment estimate) = sigma_w^2 / m`` with ``m`` subjects per
    sequence. If the between-subject SD ``sigma_b`` is given, the total
    size of an equivalent two-arm parallel trial (outcome SD
    ``sqrt(sigma_b^2 + sigma_w^2)``) is reported for comparison.
    """
    if delta == 0 or not math.isfinite(delta):
        raise ValueError("delta must be finite and non-zero")
    if not sigma_w > 0:
        raise ValueError("sigma_w must be positive")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    m = max(2, int(math.ceil((z * sigma_w / delta) ** 2)))
    while _crossover_power(m, delta, sigma_w, alpha) < power:
        m += 1
    parallel = None
    if sigma_b is not None:
        if sigma_b < 0:
            raise ValueError("sigma_b must be non-negative")
        s2 = sigma_b ** 2 + sigma_w ** 2
        n_arm = max(2, int(math.ceil(2 * s2 * z * z / delta ** 2)))
        while True:
            df = 2 * n_arm - 2
            ncp = abs(delta) / math.sqrt(2 * s2 / n_arm)
            tcrit = stats.t.ppf(1 - alpha / 2, df)
            if stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp) >= power:
                break
            n_arm += 1
        parallel = 2 * n_arm
    return CrossoverSampleSize(
        n_total=2 * m,
        n_per_sequence=m,
        achieved_power=_crossover_power(m, delta, sigma_w, alpha),
        parallel_n_total=parallel,
    )
