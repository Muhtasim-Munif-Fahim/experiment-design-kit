"""Effect-size and sample-size calculators for two-proportion and two-sample tests.

This module focuses on the descriptive statistics that drive experiment
design: standardised effect sizes (Cohen's h for proportions, Cohen's d for
means) and the closed-form sample-size formulas used at the planning stage.
Statistical *power* (one-/two-sample, alpha/beta trade-offs) lives in
``experiment_design_kit.power``; minimum detectable effects live in
``experiment_design_kit.mde``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as sp


@dataclass(frozen=True)
class ProportionTestResult:
    """Result of a two-proportion z-test."""

    z: float
    p_value: float
    pooled_p: float
    se: float


@dataclass(frozen=True)
class TTestResult:
    """Result of a two-sample t-test (pooled or Welch)."""

    statistic: float
    p_value: float
    df: float
    se: float


@dataclass(frozen=True)
class SampleSizeResult:
    """Result of a sample-size calculation."""

    n_per_group: float
    n_per_group_required: int
    n_total_required: int
    effect_size: float
    metadata: dict


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions.

    Uses the arcsine square-root transform: ``h = 2*arcsin(sqrt(p1)) -
    2*arcsin(sqrt(p2))``. The convention returns a non-negative value.
    """
    for label, value in (("p1", p1), ("p2", p2)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be in [0, 1]")
    phi1 = math.asin(math.sqrt(p1))
    phi2 = math.asin(math.sqrt(p2))
    return abs(2.0 * (phi1 - phi2))


def cohens_d(mean1: float, mean2: float, sd1: float, sd2: float) -> float:
    """Cohen's d using the pooled standard deviation of two samples."""
    for label, value in (("sd1", sd1), ("sd2", sd2)):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    pooled_sd = math.sqrt((sd1 ** 2 + sd2 ** 2) / 2.0)
    return abs(mean1 - mean2) / pooled_sd


def two_proportion_z(
    p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05
) -> ProportionTestResult:
    """Two-sided two-proportion z-test with a pooled standard error.

    Compares proportions ``p1`` and ``p2`` observed over ``n1`` and ``n2``
    Bernoulli trials. Returns the z-statistic, two-sided p-value, the pooled
    proportion and its standard error.
    """
    for label, value in (("p1", p1), ("p2", p2)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be in [0, 1]")
    for label, value in (("n1", n1), ("n2", n2)):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        z = 0.0
    else:
        z = (p1 - p2) / se
    p_value = 2.0 * sp.norm.sf(abs(z))
    return ProportionTestResult(z=z, p_value=p_value, pooled_p=pooled, se=se)


def welch_t(
    mean1: float, var1: float, n1: int,
    mean2: float, var2: float, n2: int,
) -> TTestResult:
    """Welch's two-sample t-test (unequal variances, Satterthwaite df)."""
    for label, value in (("n1", n1), ("n2", n2)):
        if value < 2:
            raise ValueError(f"{label} must be at least 2")
    for label, value in (("var1", var1), ("var2", var2)):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        t_stat = 0.0
    else:
        t_stat = (mean1 - mean2) / se
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else float(n1 + n2 - 2)
    p_value = 2.0 * sp.t.sf(abs(t_stat), df)
    return TTestResult(statistic=t_stat, p_value=p_value, df=df, se=se)


def pooled_t(
    mean1: float, var1: float, n1: int,
    mean2: float, var2: float, n2: int,
) -> TTestResult:
    """Student's two-sample t-test with pooled variance."""
    for label, value in (("n1", n1), ("n2", n2)):
        if value < 2:
            raise ValueError(f"{label} must be at least 2")
    for label, value in (("var1", var1), ("var2", var2)):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    df = n1 + n2 - 2
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    se = math.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        t_stat = 0.0
    else:
        t_stat = (mean1 - mean2) / se
    p_value = 2.0 * sp.t.sf(abs(t_stat), df)
    return TTestResult(statistic=t_stat, p_value=p_value, df=df, se=se)


def two_proportion_sample_size(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> SampleSizeResult:
    """Per-group sample size for a two-proportion z-test (equal allocation by default).

    Uses the standard normal-approximation formula. ``ratio`` is n2/n1, so
    ``ratio=1.0`` gives equal-sized groups. The returned ``n_per_group`` is the
    closed-form plan value for group 1; ``n_per_group_required`` is its ceiling.
    """
    for label, value in (("p1", p1), ("p2", p2)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{label} must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    delta = abs(p1 - p2)
    if delta == 0:
        raise ValueError("p1 and p2 must differ for a sample-size calculation")
    z_alpha = sp.norm.ppf(1.0 - alpha / 2.0)
    z_beta = sp.norm.ppf(power)
    p_bar = (p1 + ratio * p2) / (1.0 + ratio)
    h = cohen_h(p1, p2)
    numerator = (
        z_alpha * math.sqrt(p_bar * (1.0 - p_bar) * (1.0 + 1.0 / ratio))
        + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2) / ratio)
    ) / delta
    n_group1 = numerator ** 2
    n_group1_required = max(1, math.ceil(n_group1))
    n_group2 = n_group1 * ratio
    n_group2_required = max(1, math.ceil(n_group2))
    return SampleSizeResult(
        n_per_group=n_group1,
        n_per_group_required=n_group1_required,
        n_total_required=n_group1_required + n_group2_required,
        effect_size=h,
        metadata={
            "p1": p1,
            "p2": p2,
            "p_bar": p_bar,
            "ratio": ratio,
            "n_group1": n_group1,
            "n_group2": n_group2,
            "n_group2_required": n_group2_required,
        },
    )


def two_sample_t_sample_size(
    d: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> SampleSizeResult:
    """Per-group sample size for a two-sample t-test (equal n, equal variance).

    Uses the closed-form normal-approximation n = 2*(z_alpha/2 + z_beta)^2 / d^2,
    which is the standard planning rule of thumb. For the exact (noncentral t)
    value used in power calculations, see :func:`experiment_design_kit.power.required_sample_size`.
    """
    if d <= 0:
        raise ValueError("effect size d must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    z_alpha = sp.norm.ppf(1.0 - alpha / 2.0)
    z_beta = sp.norm.ppf(power)
    n = 2.0 * (z_alpha + z_beta) ** 2 / d ** 2
    n_required = max(2, math.ceil(n))
    return SampleSizeResult(
        n_per_group=n,
        n_per_group_required=n_required,
        n_total_required=2 * n_required,
        effect_size=d,
        metadata={"method": "normal approximation"},
    )


__all__ = [
    "ProportionTestResult",
    "SampleSizeResult",
    "TTestResult",
    "cohen_h",
    "cohens_d",
    "two_proportion_sample_size",
    "two_proportion_z",
    "two_sample_t_sample_size",
    "welch_t",
    "pooled_t",
]
