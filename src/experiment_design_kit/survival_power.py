"""Log-rank (time-to-event) power and sample-size planning.

For a two-arm trial analysed with the log-rank test, power depends on the
number of *events*, not on the number of subjects. Planning is therefore two
steps:

1. **Events** needed to detect a hazard ratio ``HR`` —
   Schoenfeld (1981): ``d = (z_{1-a/s} + z_{1-b})^2 / (p (1-p) log(HR)^2)``;
   Freedman (1982): ``d = (z_{1-a/s} + z_{1-b})^2 (1 + r HR)^2 / (r (1 - HR)^2)``
   with ``p`` the treatment allocation fraction, ``r = p / (1 - p)`` and
   ``s`` the number of sides.
2. **Subjects** needed so that, with uniform accrual over ``accrual_time``,
   additional ``follow_up`` and exponential event / dropout hazards, the
   expected number of events reaches ``d``:
   ``P(event) = lam / (lam + eta) * (1 - (e^{-(lam+eta) f} - e^{-(lam+eta)(a+f)}) / ((lam+eta) a))``
   (Lachin & Foulkes 1986 under exponential survival; ``a = 0`` gives
   ``lam / (lam + eta) * (1 - e^{-(lam+eta) f})``).

Medians convert to hazards by ``lam = log 2 / median``; annual dropout
probabilities convert by ``eta = -log(1 - q)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats as sp

__all__ = [
    "LogRankPlan",
    "exponential_event_probability",
    "freedman_events",
    "hazard_from_median",
    "logrank_power",
    "logrank_sample_size",
    "schoenfeld_events",
]

_METHODS = ("schoenfeld", "freedman")


@dataclass(frozen=True)
class LogRankPlan:
    """Sample-size plan for a two-arm log-rank test."""

    events: int
    n_total: int
    n_treatment: int
    n_control: int
    hazard_ratio: float
    hazard_control: float
    hazard_treatment: float
    event_prob_control: float
    event_prob_treatment: float
    event_prob_pooled: float
    method: str
    alpha: float
    power: float
    achieved_power: float


def hazard_from_median(median: float) -> float:
    """Exponential hazard with the given median survival time."""
    if not median > 0:
        raise ValueError("median must be positive")
    return math.log(2.0) / float(median)


def _check(hr, alpha, power, allocation, sides):
    if not hr > 0 or hr == 1.0:
        raise ValueError("hazard ratio must be positive and different from 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 < allocation < 1.0:
        raise ValueError("allocation must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")


def _z_sum(alpha, power, sides):
    return sp.norm.ppf(1.0 - alpha / sides) + sp.norm.ppf(power)


def schoenfeld_events(hazard_ratio, alpha=0.05, power=0.8, allocation=0.5, sides=2) -> int:
    """Events required by Schoenfeld's formula (rounded up)."""
    _check(hazard_ratio, alpha, power, allocation, sides)
    z = _z_sum(alpha, power, sides)
    d = z * z / (allocation * (1.0 - allocation) * math.log(hazard_ratio) ** 2)
    return math.ceil(d - 1e-9)


def freedman_events(hazard_ratio, alpha=0.05, power=0.8, allocation=0.5, sides=2) -> int:
    """Events required by Freedman's formula (rounded up)."""
    _check(hazard_ratio, alpha, power, allocation, sides)
    z = _z_sum(alpha, power, sides)
    r = allocation / (1.0 - allocation)
    d = z * z * (1.0 + r * hazard_ratio) ** 2 / (r * (1.0 - hazard_ratio) ** 2)
    return math.ceil(d - 1e-9)


def exponential_event_probability(hazard, accrual_time, follow_up, dropout_hazard=0.0) -> float:
    """P(an enrolled subject's event is observed) under uniform accrual.

    Subjects enter uniformly on ``[0, accrual_time]`` and are followed until
    ``accrual_time + follow_up``; events and dropouts are competing
    exponential risks with hazards ``hazard`` and ``dropout_hazard``.
    """
    if not hazard > 0:
        raise ValueError("hazard must be positive")
    if accrual_time < 0 or follow_up < 0 or accrual_time + follow_up <= 0:
        raise ValueError("accrual_time and follow_up must be non-negative with a positive total")
    if dropout_hazard < 0:
        raise ValueError("dropout_hazard must be non-negative")
    total = hazard + dropout_hazard
    share = hazard / total
    a, f = float(accrual_time), float(follow_up)
    if a == 0:
        return share * (1.0 - math.exp(-total * f))
    tail = (math.exp(-total * f) - math.exp(-total * (a + f))) / (total * a)
    return share * (1.0 - tail)


def _event_probs(hr, hazard_control, accrual_time, follow_up, dropout_hazard, allocation):
    lam_c = float(hazard_control)
    lam_t = hr * lam_c
    pc = exponential_event_probability(lam_c, accrual_time, follow_up, dropout_hazard)
    pt = exponential_event_probability(lam_t, accrual_time, follow_up, dropout_hazard)
    return lam_c, lam_t, pc, pt, allocation * pt + (1.0 - allocation) * pc


def _power_from_events(events, hr, alpha, allocation, sides, method):
    if method == "schoenfeld":
        effect = math.sqrt(events * allocation * (1.0 - allocation)) * abs(math.log(hr))
    else:
        r = allocation / (1.0 - allocation)
        effect = math.sqrt(events * r) * abs(1.0 - hr) / (1.0 + r * hr)
    return float(sp.norm.cdf(effect - sp.norm.ppf(1.0 - alpha / sides)))


def logrank_power(
    n_total,
    hazard_ratio,
    hazard_control,
    accrual_time,
    follow_up,
    *,
    dropout_hazard=0.0,
    alpha=0.05,
    allocation=0.5,
    sides=2,
    method="schoenfeld",
) -> float:
    """Approximate power of the log-rank test for ``n_total`` enrolled subjects."""
    _check(hazard_ratio, alpha, None, allocation, sides)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if n_total < 2:
        raise ValueError("n_total must be at least 2")
    *_, pooled = _event_probs(hazard_ratio, hazard_control, accrual_time, follow_up, dropout_hazard, allocation)
    return _power_from_events(n_total * pooled, hazard_ratio, alpha, allocation, sides, method)


def logrank_sample_size(
    hazard_ratio,
    hazard_control,
    accrual_time,
    follow_up,
    *,
    dropout_hazard=0.0,
    alpha=0.05,
    power=0.8,
    allocation=0.5,
    sides=2,
    method="schoenfeld",
) -> LogRankPlan:
    """Events and subjects needed to detect ``hazard_ratio`` with the log-rank test.

    ``hazard_control`` is the control-arm exponential hazard (use
    :func:`hazard_from_median`). Times share one unit (e.g. months).
    Arm sizes are rounded up separately so the allocation is respected.
    """
    _check(hazard_ratio, alpha, power, allocation, sides)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    events_fn = schoenfeld_events if method == "schoenfeld" else freedman_events
    d = events_fn(hazard_ratio, alpha, power, allocation, sides)
    lam_c, lam_t, pc, pt, pooled = _event_probs(
        hazard_ratio, hazard_control, accrual_time, follow_up, dropout_hazard, allocation
    )
    n = d / pooled
    n_t = math.ceil(allocation * n - 1e-9)
    n_c = math.ceil((1.0 - allocation) * n - 1e-9)
    achieved = _power_from_events(n_t * pt + n_c * pc, hazard_ratio, alpha, allocation, sides, method)
    return LogRankPlan(
        events=d,
        n_total=n_t + n_c,
        n_treatment=n_t,
        n_control=n_c,
        hazard_ratio=float(hazard_ratio),
        hazard_control=lam_c,
        hazard_treatment=lam_t,
        event_prob_control=pc,
        event_prob_treatment=pt,
        event_prob_pooled=pooled,
        method=method,
        alpha=float(alpha),
        power=float(power),
        achieved_power=achieved,
    )
