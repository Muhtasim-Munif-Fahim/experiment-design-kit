"""Statistical power analysis for one-sample, two-sample, and proportion tests.

Power is the probability of rejecting the null hypothesis when the alternative
is true. Here ``alpha`` (type-I error) and ``power`` (1 - beta, type-II
error) are exposed as explicit parameters ("sliders") so callers can trade
them off during design. t-test power uses the exact noncentral t distribution;
proportion power uses the standard normal approximation.
"""

from __future__ import annotations

import math

from scipy import stats as sp

from .stats import SampleSizeResult


def power_two_sample(effect_size: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Power of a two-sided two-sample t-test with equal group sizes.

    ``effect_size`` is Cohen's d; ``n_per_group`` is the size of each group.
    Uses the noncentral t distribution with df = 2*n - 2.
    """
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2")
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    _validate_alpha(alpha)
    df = 2 * n_per_group - 2
    ncp = effect_size * math.sqrt(n_per_group / 2.0)
    return _nct_power(effect_size, df, ncp, alpha)


def power_one_sample(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided one-sample t-test (``effect_size`` is Cohen's d)."""
    _validate_n(n)
    if n < 2:
        raise ValueError("n must be at least 2")
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    _validate_alpha(alpha)
    df = n - 1
    ncp = effect_size * math.sqrt(n)
    return _nct_power(effect_size, df, ncp, alpha)


def power_proportion(
    p1: float, p2: float, n_per_group: int, alpha: float = 0.05, ratio: float = 1.0
) -> float:
    """Two-sided power of a two-proportion z-test (normal approximation).

    ``ratio`` is n2/n1. The pooled proportion ``p_bar = (p1 + ratio*p2)/(1+ratio)``
    is used for the null standard error, matching the planner in
    :func:`experiment_design_kit.stats.two_proportion_sample_size`.
    """
    _validate_n(n_per_group)
    for label, value in (("p1", p1), ("p2", p2)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be in [0, 1]")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    _validate_alpha(alpha)
    delta = abs(p1 - p2)
    if delta == 0:
        return float(alpha)
    n1 = n_per_group
    n2 = n_per_group * ratio
    p_bar = (p1 + ratio * p2) / (1.0 + ratio)
    se_null = math.sqrt(p_bar * (1.0 - p_bar) * (1.0 / n1 + 1.0 / n2))
    ncp = delta / se_null
    z_alpha = sp.norm.ppf(1.0 - alpha / 2.0)
    return float(sp.norm.sf(z_alpha - ncp) + sp.norm.cdf(-z_alpha - ncp))


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    kind: str = "two-sample",
) -> SampleSizeResult:
    """Smallest per-group (t-test) n achieving ``power`` for the given effect size.

    Uses the exact noncentral t distribution so the returned n is the true
    minimal integer, not a normal-approximation rule of thumb. ``kind`` selects
    ``"two-sample"`` or ``"one-sample"``.
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    _validate_alpha(alpha)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if kind not in ("two-sample", "one-sample"):
        raise ValueError("kind must be 'two-sample' or 'one-sample'")
    power_fn = power_two_sample if kind == "two-sample" else power_one_sample
    n = 2 if kind == "two-sample" else 2
    while power_fn(effect_size, n, alpha) < power:
        n += 1
    return SampleSizeResult(
        n_per_group=float(n),
        n_per_group_required=n,
        n_total_required=(2 * n if kind == "two-sample" else n),
        effect_size=effect_size,
        metadata={"kind": kind, "alpha": alpha, "target_power": power},
    )


def power_curve(
    effect_size: float,
    n_values: list[int],
    alpha: float = 0.05,
    kind: str = "two-sample",
) -> list[float]:
    """Evaluate power across a sequence of sample sizes."""
    power_fn = power_two_sample if kind == "two-sample" else power_one_sample
    return [power_fn(effect_size, n, alpha) for n in n_values]


def _nct_power(effect_size: float, df: float, ncp: float, alpha: float) -> float:
    t_crit = sp.t.ppf(1.0 - alpha / 2.0, df)
    return float(sp.nct.sf(t_crit, df, ncp) + sp.nct.cdf(-t_crit, df, ncp))


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _validate_n(n: int) -> None:
    if n < 1:
        raise ValueError("n must be at least 1")


__all__ = [
    "power_one_sample",
    "power_proportion",
    "power_two_sample",
    "power_curve",
    "required_sample_size",
]
