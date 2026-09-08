"""Always-valid sequential monitoring for A/B testing.

Sequential testing lets you peek at the data multiple times while keeping
the false-positive rate under control. These functions implement a
mixture Sequential Probability Ratio Test (mSPRT) approach that yields
*always-valid* p-values: a p-value below ``alpha`` at any look implies
evidence against the null, regardless of how many looks preceded it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from scipy import stats as sp


@dataclass
class SequentialResult:
    """Result of a single sequential look."""

    look: int
    z_statistic: float
    p_value: float
    always_valid_pvalue: float
    alpha: float
    significant: bool
    cumulative_n: int


@dataclass
class SequentialReport:
    """Full report from a sequential monitoring run."""

    results: List[SequentialResult] = field(default_factory=list)
    stopped: bool = False
    stopped_at: int = -1
    final_pvalue: float = 1.0

    @property
    def is_significant(self) -> bool:
        return any(r.significant for r in self.results)


def sequential_proportion_test(
    successes: List[int],
    totals: List[int],
    alpha: float = 0.05,
    m: float = 0.01,
    stop_on_significant: bool = False,
) -> SequentialReport:
    """Sequential z-test for proportions with always-valid p-values.

    Parameters
    ----------
    successes
        Number of successes *at each look* (incremental, not cumulative).
        These are accumulated internally to form cumulative counts.
    totals
        Sample size *at each look* (incremental, not cumulative).
    alpha
        Nominal false-positive rate.
    m
        Mixture parameter (variance of the mixing distribution). Smaller
        values make the test more conservative.
    stop_on_significant
        If ``True``, stop processing after the first significant look.
        If ``False`` (default), all looks are processed.
    """
    if len(successes) != len(totals):
        raise ValueError("successes and totals must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if m <= 0:
        raise ValueError("m must be positive")

    report = SequentialReport()
    cumulative_successes = 0
    cumulative_total = 0
    first_significant = -1

    for i, (s, n) in enumerate(zip(successes, totals)):
        if n <= 0:
            raise ValueError("totals must be positive")
        cumulative_successes += s
        cumulative_total += n
        p_hat = cumulative_successes / cumulative_total

        # Always-valid p-value via mixture SPRT
        # Test against a point null of p=0.5 with mixing variance m
        se = math.sqrt(0.5 * 0.5 / cumulative_total)
        z = abs(p_hat - 0.5) / se if se > 0 else 0.0

        always_valid_pvalue = -0.5 * (z ** 2) / (1.0 + m * cumulative_total)
        always_valid_pvalue = math.exp(always_valid_pvalue)
        always_valid_pvalue = min(1.0, max(0.0, always_valid_pvalue))

        p_value = 2.0 * sp.norm.sf(z)

        significant = always_valid_pvalue < alpha

        report.results.append(
            SequentialResult(
                look=i + 1,
                z_statistic=z,
                p_value=float(p_value),
                always_valid_pvalue=always_valid_pvalue,
                alpha=alpha,
                significant=significant,
                cumulative_n=cumulative_total,
            )
        )

        if significant and first_significant < 0:
            first_significant = i + 1
            report.final_pvalue = always_valid_pvalue

        if significant and stop_on_significant and not report.stopped:
            report.stopped = True
            report.stopped_at = i + 1
            report.final_pvalue = always_valid_pvalue
            break

    if report.results:
        report.final_pvalue = report.results[-1].always_valid_pvalue
    return report


def always_valid_pvalue(z_statistic: float, n: int, m: float = 0.01) -> float:
    """Compute the always-valid p-value for a z-statistic.

    Uses the mixture sequential probability ratio test (mSPRT) formula:

    .. math::
        p_{av} = \\exp\\left(-\\frac{1}{2} \\frac{z^2}{1 + m \\cdot n}\\right)

    where ``m`` is the mixture parameter and ``n`` is the current sample
    size. This p-value is valid for repeated testing: ``p_{av} < alpha``
    at any look implies evidence against the null.
    """
    if n <= 0:
        return 1.0
    log_p = -0.5 * (z_statistic ** 2) / (1.0 + m * n)
    return min(1.0, max(0.0, math.exp(log_p)))


__all__ = [
    "SequentialResult",
    "SequentialReport",
    "always_valid_pvalue",
    "sequential_proportion_test",
]
