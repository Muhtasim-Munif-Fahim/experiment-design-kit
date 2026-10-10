"""Multiple-comparison corrections and A/B/n sample-size planning.

Testing several metrics, segments or treatment variants at once inflates the
chance of at least one false positive. With ``m`` independent tests at
``alpha = 0.05`` the family-wise error rate (FWER) is ``1 - 0.95^m``: about
40% for ten metrics. Two families of corrections are provided:

* **FWER control** (no false positive anywhere in the family):
  ``bonferroni`` (``m * p``), ``sidak`` (``1 - (1 - p)^m``), the step-down
  ``holm`` and ``holm-sidak`` procedures (uniformly more powerful than their
  single-step versions), and the step-up ``hochberg`` procedure (valid under
  independence / positive dependence).
* **FDR control** (expected share of false discoveries among rejections):
  Benjamini-Hochberg ``bh`` (independence / PRDS) and Benjamini-Yekutieli
  ``by`` (any dependence, at the cost of a ``sum(1/i)`` factor).

All adjusted p-values are monotone in the raw ones, capped at 1, and
returned in the caller's original order, so ``adjusted <= alpha`` is the
rejection rule for every method.

:func:`multi_arm_sample_size` plans an A/B/n conversion test with ``k``
treatment arms each compared against one control, splitting ``alpha`` with
Bonferroni or Šidák so the many-to-one family keeps its FWER.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .stats import two_proportion_sample_size

__all__ = [
    "METHODS",
    "MultiArmSampleSizeResult",
    "MultipleTestingResult",
    "adjust_pvalues",
    "corrected_alpha",
    "family_wise_error_rate",
    "multi_arm_sample_size",
]

METHODS = ("bonferroni", "sidak", "holm", "holm-sidak", "hochberg", "bh", "by")
_ALIASES = {
    "fdr_bh": "bh",
    "benjamini-hochberg": "bh",
    "fdr_by": "by",
    "benjamini-yekutieli": "by",
    "holm-bonferroni": "holm",
    "holm_sidak": "holm-sidak",
}
_FDR = {"bh", "by"}


@dataclass(frozen=True)
class MultipleTestingResult:
    """Adjusted p-values and rejection decisions for a family of tests."""

    method: str
    alpha: float
    p_values: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]

    @property
    def n_tests(self) -> int:
        return len(self.p_values)

    @property
    def n_rejected(self) -> int:
        return int(sum(self.rejected))

    @property
    def controls(self) -> str:
        """``"FDR"`` or ``"FWER"`` depending on the method."""
        return "FDR" if self.method in _FDR else "FWER"


def _normalize_method(method: str) -> str:
    key = str(method).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return key


def _checked_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return alpha


def _checked_pvalues(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("p_values must not be empty")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p_values must lie in [0, 1]")
    return p


def adjust_pvalues(
    p_values: Sequence[float],
    method: str = "holm",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Adjust a family of p-values for multiple comparisons.

    ``method`` is one of :data:`METHODS` (aliases such as ``"fdr_bh"`` are
    accepted). A hypothesis is rejected when its adjusted p-value is at most
    ``alpha``.
    """
    method = _normalize_method(method)
    alpha = _checked_alpha(alpha)
    p = _checked_pvalues(p_values)
    m = p.size

    if method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "sidak":
        adj = 1.0 - (1.0 - p) ** m
    elif method in ("holm", "holm-sidak"):
        order = np.argsort(p, kind="mergesort")
        sorted_p = p[order]
        factors = m - np.arange(m)  # m, m-1, ..., 1
        if method == "holm":
            step = sorted_p * factors
        else:
            step = 1.0 - (1.0 - sorted_p) ** factors
        step = np.minimum(np.maximum.accumulate(step), 1.0)
        adj = np.empty(m)
        adj[order] = step
    else:  # step-up: hochberg, bh, by
        order = np.argsort(p, kind="mergesort")[::-1]  # descending
        sorted_p = p[order]
        ranks = m - np.arange(m)  # rank of each descending p: m, m-1, ..., 1
        if method == "hochberg":
            step = sorted_p * (m - ranks + 1)
        else:
            step = sorted_p * m / ranks
            if method == "by":
                step = step * float(np.sum(1.0 / np.arange(1, m + 1)))
        step = np.minimum(np.minimum.accumulate(step), 1.0)
        adj = np.empty(m)
        adj[order] = step

    adj = np.clip(adj, 0.0, 1.0)
    rejected = adj <= alpha
    return MultipleTestingResult(
        method=method,
        alpha=alpha,
        p_values=tuple(float(v) for v in p),
        adjusted=tuple(float(v) for v in adj),
        rejected=tuple(bool(v) for v in rejected),
    )


def corrected_alpha(alpha: float, n_tests: int, method: str = "bonferroni") -> float:
    """Per-comparison significance level for a single-step FWER correction."""
    alpha = _checked_alpha(alpha)
    if isinstance(n_tests, bool) or int(n_tests) != n_tests or n_tests < 1:
        raise ValueError("n_tests must be a positive integer")
    method = str(method).strip().lower()
    if method == "bonferroni":
        return alpha / n_tests
    if method == "sidak":
        return 1.0 - (1.0 - alpha) ** (1.0 / n_tests)
    if method == "none":
        return alpha
    raise ValueError("method must be 'bonferroni', 'sidak' or 'none'")


def family_wise_error_rate(alpha: float, n_tests: int) -> float:
    """FWER of ``n_tests`` independent uncorrected tests at level ``alpha``."""
    alpha = _checked_alpha(alpha)
    if isinstance(n_tests, bool) or int(n_tests) != n_tests or n_tests < 1:
        raise ValueError("n_tests must be a positive integer")
    return 1.0 - (1.0 - alpha) ** n_tests


@dataclass(frozen=True)
class MultiArmSampleSizeResult:
    """Sample-size plan for a many-to-one (A/B/n) conversion test."""

    p_control: float
    p_treatments: tuple[float, ...]
    alpha: float
    alpha_per_comparison: float
    power: float
    correction: str
    n_per_arm_by_comparison: tuple[int, ...]
    n_per_arm_required: int
    n_total_required: int

    @property
    def n_arms(self) -> int:
        return 1 + len(self.p_treatments)


def multi_arm_sample_size(
    p_control: float,
    p_treatments: Sequence[float],
    alpha: float = 0.05,
    power: float = 0.8,
    correction: str = "bonferroni",
) -> MultiArmSampleSizeResult:
    """Per-arm sample size so every treatment-vs-control test has ``power``.

    Each of the ``k`` comparisons is planned with a two-sided two-proportion
    z-test at the corrected per-comparison level. With equal allocation the
    plan uses the largest per-comparison requirement for every arm, so the
    hardest-to-detect variant still gets the requested power.
    """
    treatments = tuple(float(v) for v in p_treatments)
    if not treatments:
        raise ValueError("p_treatments must contain at least one treatment rate")
    alpha = _checked_alpha(alpha)
    per = corrected_alpha(alpha, len(treatments), correction)
    sizes = tuple(
        int(two_proportion_sample_size(p_control, pt, alpha=per, power=power).n_per_group_required)
        for pt in treatments
    )
    n_arm = max(sizes)
    return MultiArmSampleSizeResult(
        p_control=float(p_control),
        p_treatments=treatments,
        alpha=alpha,
        alpha_per_comparison=float(per),
        power=float(power),
        correction=str(correction).strip().lower(),
        n_per_arm_by_comparison=sizes,
        n_per_arm_required=n_arm,
        n_total_required=int(n_arm * (len(treatments) + 1)),
    )
