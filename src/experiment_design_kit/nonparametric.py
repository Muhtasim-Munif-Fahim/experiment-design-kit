"""Mann-Whitney / Wilcoxon rank-sum power and sample-size planning.

Parametric t-test power assumes roughly Gaussian outcomes. When the metric is
ordinal, heavily skewed, or contaminated by outliers, the two-sample Wilcoxon
rank-sum (Mann-Whitney U) test is the usual alternative. Its power depends on
``p = P(X > Y) + 0.5 P(X = Y)`` — the probability that a random draw from the
treatment arm outranks a random draw from control — rather than on Cohen's d.

This module implements the Noether (1987) normal approximation:

    n_per_group ~= (z_{1-alpha/2} + z_power)^2 / (6 (p - 0.5)^2)

for equal groups under continuity (no ties), plus the corresponding power
formula and a helper that maps a location shift under a named distribution
family into ``p``. The approximation is the standard textbook planner; for
exact enumeration see Zhao, Rahardja & Qu (2008).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp

__all__ = [
    "MannWhitneyPlan",
    "mann_whitney_p_from_shift",
    "mann_whitney_power",
    "mann_whitney_sample_size",
]


@dataclass(frozen=True)
class MannWhitneyPlan:
    """Sample-size plan for a two-sided Mann-Whitney test."""

    n_per_group: int
    n_total: int
    p: float
    alpha: float
    power: float
    achieved_power: float


def _validate_p(p: float) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if abs(p - 0.5) < 1e-15:
        raise ValueError("p must differ from 0.5 (null has no power target)")
    return p


def _validate_alpha_power(alpha: float, power: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")


def mann_whitney_power(p: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Two-sided power of the Mann-Whitney U test (Noether approximation).

    ``p = P(X > Y) + 0.5 P(X = Y)`` under the alternative; ``n_per_group`` is
    the size of each arm. Under continuity and equal groups the U statistic is
    approximately normal with mean ``n^2 p`` and variance ``n^2 (2n+1)/12``
    under the null after the usual rank-sum centering.
    """
    p = _validate_p(p)
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = int(n_per_group)
    # Effect on the standardized U scale: sqrt(12 n) * (p - 0.5)
    # (Noether 1987; Lehmann, Nonparametrics §1.3).
    # Exact Noether SE uses (2n+1); asymptotic form is sqrt(6 n) (p-0.5).
    effect = n * (p - 0.5) * math.sqrt(12.0 / (2 * n + 1))
    z_alpha = sp.norm.ppf(1.0 - alpha / 2.0)
    return float(sp.norm.sf(z_alpha - effect) + sp.norm.cdf(-z_alpha - effect))


def mann_whitney_sample_size(
    p: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> MannWhitneyPlan:
    """Smallest equal-group n achieving ``power`` for the Mann-Whitney test.

    Starts from the closed-form Noether formula and walks upward until the
    exact (normal-approx) power function clears the target, so the returned
    integer is not an under-estimate of the approximation.
    """
    p = _validate_p(p)
    _validate_alpha_power(alpha, power)
    z_a = sp.norm.ppf(1.0 - alpha / 2.0)
    z_b = sp.norm.ppf(power)
    n_float = (z_a + z_b) ** 2 / (6.0 * (p - 0.5) ** 2)
    n = max(2, int(math.ceil(n_float)))
    while mann_whitney_power(p, n, alpha) < power:
        n += 1
    return MannWhitneyPlan(
        n_per_group=n,
        n_total=2 * n,
        p=p,
        alpha=float(alpha),
        power=float(power),
        achieved_power=mann_whitney_power(p, n, alpha),
    )


def mann_whitney_p_from_shift(
    delta: float,
    *,
    family: str = "normal",
    scale: float = 1.0,
    n_mc: int = 200_000,
    seed: Optional[int] = 0,
) -> float:
    """Estimate ``P(X > Y) + 0.5 P(X = Y)`` for a location shift ``delta``.

    ``family`` is one of ``"normal"``, ``"logistic"``, ``"laplace"``,
    ``"exponential"``. Control draws from the base family with scale
    ``scale``; treatment is the same plus ``delta`` (for exponential the
    shift is applied after drawing, so support stays positive only when
    ``delta >= 0``). Monte Carlo keeps the helper distribution-agnostic
    without needing a closed-form CDF difference for every family.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    family = family.lower()
    n = int(n_mc)
    if n < 1000:
        raise ValueError("n_mc must be at least 1000")
    rng = np.random.default_rng(seed)
    if family == "normal":
        y = rng.normal(0.0, scale, size=n)
        x = rng.normal(delta, scale, size=n)
    elif family == "logistic":
        y = rng.logistic(0.0, scale, size=n)
        x = rng.logistic(delta, scale, size=n)
    elif family == "laplace":
        y = rng.laplace(0.0, scale, size=n)
        x = rng.laplace(delta, scale, size=n)
    elif family == "exponential":
        y = rng.exponential(scale, size=n)
        x = rng.exponential(scale, size=n) + float(delta)
    else:
        raise ValueError("family must be normal, logistic, laplace, or exponential")
    gt = float(np.mean(x > y))
    eq = float(np.mean(x == y))
    return gt + 0.5 * eq
