"""Minimum detectable effect (MDE) calculators.

Given a planned sample size and a target power, the MDE is the smallest
true effect that the design can reliably detect. For t-tests it is reported
as Cohen's d (standardised) and, by applying a per-group standard deviation,
in raw measurement units. For proportions it is reported as an absolute lift
and a relative lift, together with Cohen's h.
"""

from __future__ import annotations

from dataclasses import dataclass

from .power import power_one_sample, power_proportion, power_two_sample
from .stats import cohen_h


@dataclass(frozen=True)
class MDEResult:
    """Standardised minimum detectable effect for a t-test."""

    effect_size: float
    n_per_group: int
    alpha: float
    power: float
    kind: str


@dataclass(frozen=True)
class ProportionMDE:
    """Minimum detectable lift for a two-proportion test."""

    lift_absolute: float
    lift_relative: float
    effect_size_h: float
    p_control: float
    p_treatment: float
    n_per_group: int
    alpha: float
    power: float


def minimum_detectable_effect(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
    kind: str = "two-sample",
) -> MDEResult:
    """Smallest Cohen's d detectable at the given n, alpha, and power.

    Uses a bisection search over the noncentral-t power function, so the
    result is consistent with :func:`experiment_design_kit.power.required_sample_size`.
    """
    _require_valid_kind(kind)
    _require_valid_power_target(power)
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2")
    power_fn = power_two_sample if kind == "two-sample" else power_one_sample
    lo, hi = 1e-4, 5.0
    if power_fn(hi, n_per_group, alpha) < power:
        raise ValueError("n_per_group is too small to reach the requested power even for large effects")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if power_fn(mid, n_per_group, alpha) < power:
            lo = mid
        else:
            hi = mid
    d = hi
    return MDEResult(effect_size=d, n_per_group=n_per_group, alpha=alpha, power=power, kind=kind)


def minimum_detectable_effect_raw(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
    sigma: float = 1.0,
    kind: str = "two-sample",
) -> float:
    """MDE in raw measurement units = standardised MDE * per-group sigma."""
    _require_positive(sigma, "sigma")
    return minimum_detectable_effect(n_per_group, alpha, power, kind).effect_size * sigma


def minimum_detectable_effect_proportion(
    n_per_group: int,
    p_control: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> ProportionMDE:
    """Smallest absolute conversion-rate lift detectable at the given n and power."""
    _require_positive(n_per_group, "n_per_group")
    if not 0.0 < p_control < 1.0:
        raise ValueError("p_control must be in (0, 1)")
    _require_valid_power_target(power)
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    lo, hi = 0.0, 1.0 - p_control
    if power_proportion(p_control, p_control + hi, n_per_group, alpha, ratio) < power:
        raise ValueError("n_per_group is too small to reach the requested power")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if power_proportion(p_control, p_control + mid, n_per_group, alpha, ratio) < power:
            lo = mid
        else:
            hi = mid
    lift = hi
    p_treatment = p_control + lift
    return ProportionMDE(
        lift_absolute=lift,
        lift_relative=lift / p_control,
        effect_size_h=cohen_h(p_control, p_treatment),
        p_control=p_control,
        p_treatment=p_treatment,
        n_per_group=n_per_group,
        alpha=alpha,
        power=power,
    )


def _require_valid_kind(kind: str) -> None:
    if kind not in ("two-sample", "one-sample"):
        raise ValueError("kind must be 'two-sample' or 'one-sample'")


def _require_valid_power_target(power: float) -> None:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")


def _require_positive(value: float, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be positive")


__all__ = [
    "MDEResult",
    "ProportionMDE",
    "minimum_detectable_effect",
    "minimum_detectable_effect_proportion",
    "minimum_detectable_effect_raw",
]
