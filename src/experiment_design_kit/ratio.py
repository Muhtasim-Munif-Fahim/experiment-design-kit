"""Delta-method inference for ratio metrics in A/B tests.

Many product metrics are ratios of two per-user sums: click-through rate
(clicks / page views), revenue per session, average order value. The
randomization unit is the user, but the analysis unit (page view, session,
order) is not, so observations inside a user are correlated and the naive
"treat every page view as independent" standard error is too small, which
inflates the false-positive rate.

The delta method (Deng, Knoblich & Lu, 2018, *Applying the Delta Method in
Metric Analytics*) linearises ``R = mean(Y) / mean(X)`` around the user-level
means and gives

    Var(R) ~= (1 / n) * [ s_Y^2 / mu_X^2
                          - 2 * mu_Y * s_XY / mu_X^3
                          + mu_Y^2 * s_X^2 / mu_X^4 ]

where ``Y`` and ``X`` are the per-user numerator and denominator. The same
linearisation gives a standard error for the relative lift ``R_t / R_c - 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp

__all__ = [
    "RatioMetricResult",
    "delta_method_ratio_test",
    "ratio_metric_variance",
]


@dataclass(frozen=True)
class RatioMetricResult:
    """Result of a delta-method two-sample test on a ratio metric."""

    ratio_control: float
    ratio_treatment: float
    absolute_diff: float
    se_diff: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    relative_lift: float
    se_relative: float
    relative_ci_low: float
    relative_ci_high: float
    n_control: int
    n_treatment: int
    alpha: float

    @property
    def significant(self) -> bool:
        """Whether the two-sided test rejects at ``alpha``."""
        return self.p_value < self.alpha


def _paired_arrays(numerator, denominator, label: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(numerator, dtype=float).ravel()
    x = np.asarray(denominator, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError(f"{label}: numerator and denominator must have the same length")
    if y.size < 2:
        raise ValueError(f"{label}: at least two randomization units are required")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
        raise ValueError(f"{label}: values must be finite")
    if x.mean() <= 0.0:
        raise ValueError(f"{label}: the denominator mean must be positive")
    return y, x


def ratio_metric_variance(numerator, denominator) -> tuple[float, float]:
    """Return ``(ratio, variance)`` of ``sum(Y) / sum(X)`` by the delta method.

    ``numerator`` and ``denominator`` hold one value per randomization unit
    (e.g. clicks and page views per user). Sample (co)variances use
    ``ddof=1``.
    """
    y, x = _paired_arrays(numerator, denominator, "ratio")
    return _ratio_variance(y, x)


def _ratio_variance(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    n = y.size
    mu_y, mu_x = float(y.mean()), float(x.mean())
    cov = np.cov(y, x, ddof=1)
    var_y, var_x, cov_xy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    ratio = mu_y / mu_x
    variance = (
        var_y / mu_x ** 2
        - 2.0 * mu_y * cov_xy / mu_x ** 3
        + mu_y ** 2 * var_x / mu_x ** 4
    ) / n
    return ratio, max(variance, 0.0)


def delta_method_ratio_test(
    numerator_control,
    denominator_control,
    numerator_treatment,
    denominator_treatment,
    alpha: float = 0.05,
) -> RatioMetricResult:
    """Two-sided z-test for a difference in ratio metrics between two arms.

    Inputs are per-randomization-unit numerators and denominators for each
    arm. Returns the absolute difference ``R_t - R_c`` with its delta-method
    standard error, z-statistic, p-value and ``1 - alpha`` confidence
    interval, plus the relative lift ``R_t / R_c - 1`` with its own
    delta-method standard error and interval.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y_c, x_c = _paired_arrays(numerator_control, denominator_control, "control")
    y_t, x_t = _paired_arrays(numerator_treatment, denominator_treatment, "treatment")
    r_c, v_c = _ratio_variance(y_c, x_c)
    r_t, v_t = _ratio_variance(y_t, x_t)
    diff = r_t - r_c
    se = math.sqrt(v_c + v_t)
    if se > 0.0:
        z = diff / se
        p_value = float(2.0 * sp.norm.sf(abs(z)))
    else:
        z = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
        p_value = 1.0 if diff == 0.0 else 0.0
    crit = float(sp.norm.ppf(1.0 - alpha / 2.0))
    if r_c == 0.0:
        relative, se_rel = math.nan, math.nan
    else:
        relative = r_t / r_c - 1.0
        se_rel = math.sqrt(v_t / r_c ** 2 + (r_t ** 2) * v_c / r_c ** 4)
    return RatioMetricResult(
        ratio_control=r_c,
        ratio_treatment=r_t,
        absolute_diff=diff,
        se_diff=se,
        z=float(z),
        p_value=p_value,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        relative_lift=relative,
        se_relative=se_rel,
        relative_ci_low=relative - crit * se_rel,
        relative_ci_high=relative + crit * se_rel,
        n_control=int(y_c.size),
        n_treatment=int(y_t.size),
        alpha=float(alpha),
    )
