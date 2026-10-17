"""Equivalence (TOST) and non-inferiority tests and their planning.

Superiority tests ask "is B different from A?". Many product decisions ask
something else:

* **Non-inferiority** -- "is B *not worse* than A by more than a margin?"
  (e.g. a cheaper model, a performance refactor, a simplified checkout).
  One-sided test of ``H0: delta <= -margin`` vs ``H1: delta > -margin``,
  where ``delta = treatment - control`` and higher is better.
* **Equivalence** -- "is the difference inside ``(lower, upper)``?" via
  Schuirmann's two one-sided tests (TOST): reject both
  ``H0: delta <= lower`` and ``H0: delta >= upper`` at level ``alpha``.
  Equivalently the ``1 - 2*alpha`` confidence interval lies inside the
  margins.

A non-significant superiority test is *not* evidence of equivalence; these
tests put the burden of proof on showing the difference is small.

Means use Welch's t (Satterthwaite df); proportions use the unpooled Wald
standard error. Planning uses the usual normal approximation:

    n_control = (z_{1-alpha} + z_{power})^2 * (s_c^2 + s_t^2 / ratio)
                / (delta + margin)^2              (non-inferiority)

and, for TOST, the smallest ``n`` whose approximate power
``Phi((upper - delta)/se - z) + Phi((delta - lower)/se - z) - 1``
reaches the target (``ratio = n_treatment / n_control``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp

__all__ = [
    "EquivalenceTestResult",
    "NonInferiorityResult",
    "EquivalenceSampleSizeResult",
    "tost_means",
    "tost_proportions",
    "noninferiority_means",
    "noninferiority_proportions",
    "equivalence_power",
    "equivalence_sample_size",
    "noninferiority_power",
    "noninferiority_sample_size",
]


@dataclass(frozen=True)
class EquivalenceTestResult:
    """TOST result for ``delta = treatment - control``."""

    difference: float
    se: float
    df: float
    lower_margin: float
    upper_margin: float
    p_lower: float
    p_upper: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    equivalent: bool


@dataclass(frozen=True)
class NonInferiorityResult:
    """One-sided non-inferiority result for ``delta = treatment - control``."""

    difference: float
    se: float
    df: float
    margin: float
    statistic: float
    p_value: float
    confidence_bound: float
    alpha: float
    noninferior: bool
    higher_is_better: bool


@dataclass(frozen=True)
class EquivalenceSampleSizeResult:
    """Planned sample size for a non-inferiority or TOST equivalence test."""

    design: str
    n_control: int
    n_treatment: int
    n_total: int
    achieved_power: float
    metadata: dict


# -- validation helpers ---------------------------------------------------

def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    return alpha


def _check_power(power: float) -> float:
    power = float(power)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    return power


def _margins(margin: float, upper_margin: Optional[float]) -> Tuple[float, float]:
    """Return ``(lower, upper)``; a single positive ``margin`` is symmetric."""
    if upper_margin is None:
        m = float(margin)
        if not m > 0:
            raise ValueError("margin must be positive")
        return -m, m
    lower, upper = float(margin), float(upper_margin)
    if not lower < upper:
        raise ValueError("lower margin must be below upper margin")
    if not (lower < 0.0 < upper):
        raise ValueError("equivalence margins must bracket zero (lower < 0 < upper)")
    return lower, upper


def _check_ni_margin(margin: float) -> float:
    m = float(margin)
    if not m > 0:
        raise ValueError("margin must be positive")
    return m


def _sample(values: Sequence[float], label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"{label} needs at least 2 observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values")
    return arr


def _welch(control, treatment) -> Tuple[float, float, float]:
    c = _sample(control, "control")
    t = _sample(treatment, "treatment")
    vc, vt = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    se = math.sqrt(vc + vt)
    if se == 0.0:
        raise ValueError("both samples are constant; the standard error is zero")
    den = vc**2 / (c.size - 1) + vt**2 / (t.size - 1)
    df = (vc + vt) ** 2 / den if den > 0 else float(c.size + t.size - 2)
    return float(t.mean() - c.mean()), se, float(df)


def _wald(successes_c: int, n_c: int, successes_t: int, n_t: int) -> Tuple[float, float]:
    for label, s, n in (("control", successes_c, n_c), ("treatment", successes_t, n_t)):
        if n < 1:
            raise ValueError(f"n_{label} must be at least 1")
        if not 0 <= s <= n:
            raise ValueError(f"successes_{label} must be between 0 and n_{label}")
    pc, pt = successes_c / n_c, successes_t / n_t
    se = math.sqrt(pc * (1 - pc) / n_c + pt * (1 - pt) / n_t)
    if se == 0.0:
        raise ValueError("both proportions are 0 or 1; the Wald standard error is zero")
    return pt - pc, se


def _tost(diff, se, df, lower, upper, alpha) -> EquivalenceTestResult:
    dist = sp.norm if math.isinf(df) else sp.t(df)
    p_lower = float(dist.sf((diff - lower) / se))
    p_upper = float(dist.cdf((diff - upper) / se))
    crit = float(dist.ppf(1.0 - alpha))
    ci_low, ci_high = diff - crit * se, diff + crit * se
    p_value = max(p_lower, p_upper)
    return EquivalenceTestResult(
        difference=float(diff), se=float(se), df=float(df),
        lower_margin=lower, upper_margin=upper,
        p_lower=p_lower, p_upper=p_upper, p_value=p_value,
        ci_low=float(ci_low), ci_high=float(ci_high), alpha=alpha,
        equivalent=bool(p_value < alpha),
    )


def _noninferiority(diff, se, df, margin, alpha, higher_is_better) -> NonInferiorityResult:
    dist = sp.norm if math.isinf(df) else sp.t(df)
    crit = float(dist.ppf(1.0 - alpha))
    if higher_is_better:
        stat = (diff + margin) / se
        bound = diff - crit * se
    else:
        stat = (margin - diff) / se
        bound = diff + crit * se
    p_value = float(dist.sf(stat))
    return NonInferiorityResult(
        difference=float(diff), se=float(se), df=float(df), margin=margin,
        statistic=float(stat), p_value=p_value, confidence_bound=float(bound),
        alpha=alpha, noninferior=bool(p_value < alpha),
        higher_is_better=bool(higher_is_better),
    )


# -- analysis ---------------------------------------------------------------

def tost_means(
    control: Sequence[float],
    treatment: Sequence[float],
    margin: float,
    alpha: float = 0.05,
    upper_margin: Optional[float] = None,
) -> EquivalenceTestResult:
    """Welch TOST equivalence test for a difference in means.

    With only ``margin`` the equivalence region is ``(-margin, margin)``;
    pass ``upper_margin`` to use ``(margin, upper_margin)`` (``margin < 0``).
    ``equivalent`` is true when both one-sided p-values are below ``alpha``,
    i.e. the ``1 - 2*alpha`` CI ``[ci_low, ci_high]`` lies inside the region.
    """
    alpha = _check_alpha(alpha)
    lower, upper = _margins(margin, upper_margin)
    diff, se, df = _welch(control, treatment)
    return _tost(diff, se, df, lower, upper, alpha)


def tost_proportions(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    margin: float,
    alpha: float = 0.05,
    upper_margin: Optional[float] = None,
) -> EquivalenceTestResult:
    """TOST equivalence test for a difference in proportions (Wald z)."""
    alpha = _check_alpha(alpha)
    lower, upper = _margins(margin, upper_margin)
    diff, se = _wald(successes_control, n_control, successes_treatment, n_treatment)
    return _tost(diff, se, math.inf, lower, upper, alpha)


def noninferiority_means(
    control: Sequence[float],
    treatment: Sequence[float],
    margin: float,
    alpha: float = 0.025,
    higher_is_better: bool = True,
) -> NonInferiorityResult:
    """One-sided Welch non-inferiority test for a difference in means.

    With ``higher_is_better`` (default) tests ``H0: delta <= -margin``; set it
    to ``False`` for metrics where lower is better (latency, cost), which tests
    ``H0: delta >= margin``. ``confidence_bound`` is the one-sided
    ``1 - alpha`` bound compared against the margin. The default
    ``alpha = 0.025`` matches the regulatory convention (a two-sided 95% CI).
    """
    alpha = _check_alpha(alpha)
    margin = _check_ni_margin(margin)
    diff, se, df = _welch(control, treatment)
    return _noninferiority(diff, se, df, margin, alpha, higher_is_better)


def noninferiority_proportions(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    margin: float,
    alpha: float = 0.025,
    higher_is_better: bool = True,
) -> NonInferiorityResult:
    """One-sided non-inferiority test for a difference in proportions (Wald z)."""
    alpha = _check_alpha(alpha)
    margin = _check_ni_margin(margin)
    diff, se = _wald(successes_control, n_control, successes_treatment, n_treatment)
    return _noninferiority(diff, se, math.inf, margin, alpha, higher_is_better)


# -- planning ---------------------------------------------------------------

def _unit_variance(sd, p_control, true_diff, ratio) -> Tuple[float, dict]:
    """Variance factor ``v`` with ``se = sqrt(v / n_control)``."""
    ratio = float(ratio)
    if not ratio > 0:
        raise ValueError("ratio must be positive")
    if p_control is not None:
        if sd is not None:
            raise ValueError("pass either sd (means) or p_control (proportions), not both")
        pc = float(p_control)
        pt = pc + float(true_diff)
        if not (0.0 < pc < 1.0 and 0.0 < pt < 1.0):
            raise ValueError("p_control and p_control + true_diff must be in (0, 1)")
        return pc * (1 - pc) + pt * (1 - pt) / ratio, {"kind": "proportions", "p_control": pc, "p_treatment": pt}
    sd = 1.0 if sd is None else float(sd)
    if not sd > 0:
        raise ValueError("sd must be positive")
    return sd * sd * (1.0 + 1.0 / ratio), {"kind": "means", "sd": sd}


def _tost_power_normal(lower, upper, delta, se, alpha) -> float:
    z = sp.norm.ppf(1.0 - alpha)
    value = sp.norm.cdf((upper - delta) / se - z) + sp.norm.cdf((delta - lower) / se - z) - 1.0
    return float(min(1.0, max(0.0, value)))


def equivalence_power(
    margin: float,
    n_control: int,
    *,
    sd: Optional[float] = None,
    p_control: Optional[float] = None,
    true_diff: float = 0.0,
    alpha: float = 0.05,
    ratio: float = 1.0,
    upper_margin: Optional[float] = None,
) -> float:
    """Approximate power of TOST at ``n_control`` (and ``ratio * n_control``)."""
    alpha = _check_alpha(alpha)
    lower, upper = _margins(margin, upper_margin)
    if n_control < 2:
        raise ValueError("n_control must be at least 2")
    v, _ = _unit_variance(sd, p_control, true_diff, ratio)
    return _tost_power_normal(lower, upper, float(true_diff), math.sqrt(v / n_control), alpha)


def equivalence_sample_size(
    margin: float,
    *,
    sd: Optional[float] = None,
    p_control: Optional[float] = None,
    true_diff: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    upper_margin: Optional[float] = None,
    max_n: int = 100_000_000,
) -> EquivalenceSampleSizeResult:
    """Smallest per-arm size for TOST to reach ``power`` (normal approximation).

    ``true_diff`` must lie strictly inside the equivalence region; the closer
    it is to a margin, the larger the required sample.
    """
    alpha = _check_alpha(alpha)
    power = _check_power(power)
    lower, upper = _margins(margin, upper_margin)
    delta = float(true_diff)
    if not lower < delta < upper:
        raise ValueError("true_diff must lie strictly inside the equivalence margins")
    v, meta = _unit_variance(sd, p_control, delta, ratio)

    def pw(n: int) -> float:
        return _tost_power_normal(lower, upper, delta, math.sqrt(v / n), alpha)

    hi = 2
    while pw(hi) < power:
        hi *= 2
        if hi > max_n:
            raise ValueError("required sample size exceeds max_n")
    lo = max(2, hi // 2)
    if pw(lo) >= power:
        hi = lo
    while lo < hi:
        mid = (lo + hi) // 2
        if pw(mid) >= power:
            hi = mid
        else:
            lo = mid + 1
    n_c = max(2, hi)
    n_t = int(math.ceil(n_c * float(ratio)))
    meta.update({"lower_margin": lower, "upper_margin": upper, "true_diff": delta,
                 "alpha": alpha, "target_power": power, "ratio": float(ratio)})
    return EquivalenceSampleSizeResult(
        design="equivalence", n_control=n_c, n_treatment=n_t, n_total=n_c + n_t,
        achieved_power=pw(n_c), metadata=meta,
    )


def noninferiority_power(
    margin: float,
    n_control: int,
    *,
    sd: Optional[float] = None,
    p_control: Optional[float] = None,
    true_diff: float = 0.0,
    alpha: float = 0.025,
    ratio: float = 1.0,
) -> float:
    """Approximate power of the one-sided non-inferiority test (higher is better)."""
    alpha = _check_alpha(alpha)
    margin = _check_ni_margin(margin)
    if n_control < 2:
        raise ValueError("n_control must be at least 2")
    v, _ = _unit_variance(sd, p_control, true_diff, ratio)
    se = math.sqrt(v / n_control)
    z = sp.norm.ppf(1.0 - alpha)
    return float(sp.norm.cdf((float(true_diff) + margin) / se - z))


def noninferiority_sample_size(
    margin: float,
    *,
    sd: Optional[float] = None,
    p_control: Optional[float] = None,
    true_diff: float = 0.0,
    alpha: float = 0.025,
    power: float = 0.8,
    ratio: float = 1.0,
) -> EquivalenceSampleSizeResult:
    """Per-arm size for a non-inferiority test (closed-form normal approximation).

    Requires ``true_diff > -margin`` (the treatment is assumed truly
    non-inferior). ``true_diff > 0`` (treatment better) shrinks the sample.
    """
    alpha = _check_alpha(alpha)
    power = _check_power(power)
    margin = _check_ni_margin(margin)
    delta = float(true_diff)
    if not delta + margin > 0:
        raise ValueError("true_diff must exceed -margin")
    v, meta = _unit_variance(sd, p_control, delta, ratio)
    z = sp.norm.ppf(1.0 - alpha) + sp.norm.ppf(power)
    n_exact = z * z * v / (delta + margin) ** 2
    n_c = max(2, int(math.ceil(n_exact - 1e-9)))
    n_t = int(math.ceil(n_c * float(ratio)))
    meta.update({"margin": margin, "true_diff": delta, "alpha": alpha,
                 "target_power": power, "ratio": float(ratio), "n_control_exact": float(n_exact)})
    achieved = noninferiority_power(
        margin, n_c, sd=meta.get("sd") if meta["kind"] == "means" else None,
        p_control=meta.get("p_control"), true_diff=delta, alpha=alpha, ratio=ratio,
    )
    return EquivalenceSampleSizeResult(
        design="noninferiority", n_control=n_c, n_treatment=n_t, n_total=n_c + n_t,
        achieved_power=achieved, metadata=meta,
    )
