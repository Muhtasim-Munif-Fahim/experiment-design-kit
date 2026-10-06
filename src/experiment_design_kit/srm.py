"""Sample ratio mismatch (SRM) checks for randomized experiments.

An SRM is a gap between the traffic split an experiment was *designed*
with (say 50/50) and the split it actually *received*. Even a small gap
on a large sample is a red flag: it usually means assignment, logging,
bot filtering, or a redirect is dropping units unevenly between arms, and
then every downstream metric comparison is suspect. The standard guard is
to test the observed arm counts against the planned ratio *before*
reading any treatment effect.

Two tests are provided (numpy + scipy only):

* **Fixed-horizon** :func:`sample_ratio_mismatch` -- Pearson chi-square
  (or likelihood-ratio G-test) goodness-of-fit of the observed counts to
  the planned ratio, for two or more arms. The conventional SRM alarm
  level is ``alpha = 0.001`` because the check is run on every
  experiment and a false alarm halts analysis.
* **Sequential** :func:`sequential_srm_test` -- an anytime-valid
  Dirichlet-multinomial Bayes factor (Lindon & Malek, 2020). The Bayes
  factor against the planned ratio is a nonnegative martingale under the
  null, so by Ville's inequality the running ``p = min(1, 1 / BF)`` may
  be checked after every batch of traffic -- daily, hourly, or per unit
  -- without inflating the false-alarm rate.

:func:`srm_from_assignment` runs the fixed-horizon check directly on the
arm ids (or an assignment object) produced by this kit's randomizers.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special
from scipy import stats as sp

__all__ = [
    "SRMResult",
    "SequentialSRMLook",
    "SequentialSRMResult",
    "sample_ratio_mismatch",
    "sequential_srm_test",
    "srm_from_assignment",
]

_METHODS = {"chi-square", "g-test"}


@dataclass(frozen=True)
class SRMResult:
    """Fixed-horizon sample ratio mismatch test.

    ``expected`` holds the counts implied by the planned ratio at the
    observed total. ``residuals`` are Pearson residuals
    ``(observed - expected) / sqrt(expected)`` per arm: the arm with the
    largest absolute residual is the one drifting furthest from plan.
    ``mismatch`` is ``p_value < alpha``.
    """

    observed: tuple[int, ...]
    expected: tuple[float, ...]
    expected_share: tuple[float, ...]
    observed_share: tuple[float, ...]
    residuals: tuple[float, ...]
    statistic: float
    p_value: float
    dof: int
    alpha: float
    mismatch: bool
    method: str
    arm_labels: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(sum(self.observed))

    @property
    def worst_arm(self) -> str:
        """Label of the arm with the largest absolute Pearson residual."""
        index = int(np.argmax(np.abs(np.asarray(self.residuals))))
        return self.arm_labels[index]


@dataclass(frozen=True)
class SequentialSRMLook:
    """One look of a sequential SRM test (cumulative counts so far)."""

    look: int
    counts: tuple[int, ...]
    n: int
    log_bayes_factor: float
    p_value: float
    mismatch: bool


@dataclass
class SequentialSRMResult:
    """Anytime-valid SRM monitoring over a sequence of looks."""

    looks: list[SequentialSRMLook] = field(default_factory=list)
    expected_share: tuple[float, ...] = ()
    prior: tuple[float, ...] = ()
    alpha: float = 0.05
    arm_labels: tuple[str, ...] = ()

    @property
    def mismatch(self) -> bool:
        return any(look.mismatch for look in self.looks)

    @property
    def first_mismatch_look(self) -> int | None:
        for look in self.looks:
            if look.mismatch:
                return look.look
        return None

    @property
    def final_p_value(self) -> float:
        return self.looks[-1].p_value if self.looks else 1.0


def _labels(arm_labels: Sequence[str] | None, k: int) -> tuple[str, ...]:
    if arm_labels is None:
        if k == 2:
            return ("control", "treatment")
        return tuple(f"arm_{i}" for i in range(k))
    labels = tuple(str(label) for label in arm_labels)
    if len(labels) != k:
        raise ValueError("arm_labels must have one label per arm")
    return labels


def _expected_share(ratio: Sequence[float] | None, k: int) -> np.ndarray:
    if ratio is None:
        return np.full(k, 1.0 / k)
    if isinstance(ratio, (str, bytes)):
        raise ValueError("ratio must be a sequence of positive weights")
    weights = np.asarray(list(ratio), dtype=float)
    if weights.shape != (k,):
        raise ValueError("ratio must have one positive weight per arm")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("ratio weights must be positive and finite")
    return weights / weights.sum()


def _count_vector(counts: Sequence[int], name: str = "counts") -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1-d sequence of arm counts")
    if values.size < 2:
        raise ValueError(f"{name} must have at least two arms")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative and finite")
    if not np.all(values == np.round(values)):
        raise ValueError(f"{name} must be whole numbers")
    return values.astype(np.int64)


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return alpha


def sample_ratio_mismatch(
    counts: Sequence[int],
    ratio: Sequence[float] | None = None,
    *,
    alpha: float = 0.001,
    method: str = "chi-square",
    arm_labels: Sequence[str] | None = None,
) -> SRMResult:
    """Test observed arm counts against the planned allocation ratio.

    Parameters
    ----------
    counts:
        Units (users, sessions, ...) observed in each arm.
    ratio:
        Planned allocation weights, one per arm (``(1, 1)`` for 50/50,
        ``(1, 2)`` for 1:2, ``(90, 10)`` for a 10% holdout). Defaults to
        an equal split. Weights are normalized; they need not sum to 1.
    alpha:
        Alarm level. Defaults to the industry-standard ``0.001``.
    method:
        ``"chi-square"`` (Pearson, default) or ``"g-test"`` (likelihood
        ratio). Both are compared to chi-square with ``k - 1`` degrees of
        freedom; they agree closely at experiment scale.
    arm_labels:
        Optional names for reporting; defaults to control/treatment for
        two arms and ``arm_0 ..`` otherwise.

    Returns
    -------
    SRMResult
        Statistic, p-value, expected counts, Pearson residuals, and the
        ``mismatch`` flag.
    """
    observed = _count_vector(counts)
    k = observed.size
    share = _expected_share(ratio, k)
    alpha = _check_alpha(alpha)
    if method not in _METHODS:
        raise ValueError("method must be 'chi-square' or 'g-test'")
    total = int(observed.sum())
    if total == 0:
        raise ValueError("counts must contain at least one unit")

    expected = share * total
    if method == "chi-square":
        statistic = float(np.sum((observed - expected) ** 2 / expected))
    else:
        positive = observed > 0
        statistic = float(
            2.0 * np.sum(observed[positive] * np.log(observed[positive] / expected[positive]))
        )
        statistic = max(statistic, 0.0)
    dof = k - 1
    p_value = float(sp.chi2.sf(statistic, dof))
    residuals = (observed - expected) / np.sqrt(expected)
    return SRMResult(
        observed=tuple(int(c) for c in observed),
        expected=tuple(float(e) for e in expected),
        expected_share=tuple(float(s) for s in share),
        observed_share=tuple(float(c) / total for c in observed),
        residuals=tuple(float(r) for r in residuals),
        statistic=statistic,
        p_value=p_value,
        dof=dof,
        alpha=alpha,
        mismatch=bool(p_value < alpha),
        method=method,
        arm_labels=_labels(arm_labels, k),
    )


def srm_from_assignment(
    assignment: Any,
    ratio: Sequence[float] | None = None,
    *,
    n_arms: int | None = None,
    alpha: float = 0.001,
    method: str = "chi-square",
    arm_labels: Sequence[str] | None = None,
) -> SRMResult:
    """Run :func:`sample_ratio_mismatch` on an assignment.

    ``assignment`` is either an assignment object from this kit (anything
    with ``arm_counts`` and ``arm_labels``, e.g. the result of
    :func:`~experiment_design_kit.stratified_randomization`) or a 1-d
    array of integer arm ids ``0 .. n_arms - 1`` -- for example the arm
    column of the units that actually showed up in the analysis data.
    ``n_arms`` defaults to ``max(arm id) + 1`` (at least 2), so an arm
    that received no units at all is still counted when you pass it.
    """
    if hasattr(assignment, "arm_counts") and not isinstance(assignment, np.ndarray):
        counts = list(assignment.arm_counts)
        labels = arm_labels if arm_labels is not None else getattr(assignment, "arm_labels", None)
        if n_arms is not None and n_arms != len(counts):
            raise ValueError("n_arms does not match the assignment's number of arms")
        return sample_ratio_mismatch(
            counts, ratio, alpha=alpha, method=method, arm_labels=labels
        )

    ids = np.asarray(assignment)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("assignment must be a non-empty 1-d array of arm ids")
    if not np.issubdtype(ids.dtype, np.number) or not np.all(ids == np.round(ids)):
        raise ValueError("arm ids must be integers")
    ids = ids.astype(np.int64)
    if np.any(ids < 0):
        raise ValueError("arm ids must be nonnegative")
    inferred = max(int(ids.max()) + 1, 2)
    if n_arms is None:
        n_arms = inferred
    elif n_arms < inferred:
        raise ValueError("an arm id is out of range for n_arms")
    counts = np.bincount(ids, minlength=int(n_arms))
    return sample_ratio_mismatch(
        counts, ratio, alpha=alpha, method=method, arm_labels=arm_labels
    )


def _log_bayes_factor(counts: np.ndarray, share: np.ndarray, prior: np.ndarray) -> float:
    """Dirichlet-multinomial vs multinomial(share) log Bayes factor.

    ``log BF = log B(prior + x) - log B(prior) - sum x_i log share_i``,
    where ``B`` is the multivariate Beta function. The multinomial
    coefficient cancels between numerator and denominator.
    """
    x = counts.astype(float)
    n = float(x.sum())
    a0 = float(prior.sum())
    log_marginal = (
        special.gammaln(a0)
        - special.gammaln(a0 + n)
        + float(np.sum(special.gammaln(prior + x) - special.gammaln(prior)))
    )
    log_null = float(np.sum(x * np.log(share)))
    return float(log_marginal - log_null)


def sequential_srm_test(
    counts: Sequence[Sequence[int]],
    ratio: Sequence[float] | None = None,
    *,
    alpha: float = 0.05,
    cumulative: bool = False,
    prior: Sequence[float] | None = None,
    concentration: float | None = None,
    arm_labels: Sequence[str] | None = None,
) -> SequentialSRMResult:
    """Anytime-valid SRM monitoring with a Dirichlet-multinomial Bayes factor.

    At each look the Bayes factor of a Dirichlet-mixture alternative
    against the planned multinomial split is computed from the cumulative
    arm counts. Under the null it is a nonnegative martingale with mean 1,
    so ``P(sup_n BF_n >= 1/alpha) <= alpha`` (Ville's inequality). The
    reported ``p_value`` is the running minimum of ``min(1, 1 / BF_n)``
    and a look is flagged once it falls below ``alpha``. Peeking as often
    as you like -- after every unit, if you want -- keeps the false-alarm
    rate at or below ``alpha``.

    Parameters
    ----------
    counts:
        Sequence of looks, each a vector of arm counts. By default these
        are *incremental* (new units since the previous look), matching
        :func:`~experiment_design_kit.sequential_two_proportion_test`.
        Pass ``cumulative=True`` for running totals.
    ratio:
        Planned allocation weights (default: equal).
    alpha:
        Anytime-valid level. ``0.05`` is common for monitoring; use
        ``0.001`` for the conventional SRM alarm.
    prior, concentration:
        Dirichlet prior on the arm shares. By default it is centred on the
        planned split, ``prior = concentration * share``, with
        ``concentration`` defaulting to the number of arms (a flat
        Dirichlet for an equal split). A larger concentration targets
        smaller mismatches; any positive prior keeps the guarantee.
    arm_labels:
        Optional names for reporting.
    """
    rows = [np.asarray(_count_vector(row, "each look"), dtype=np.int64) for row in counts]
    if not rows:
        raise ValueError("at least one look is required")
    k = rows[0].size
    if any(row.size != k for row in rows):
        raise ValueError("every look must have the same number of arms")
    share = _expected_share(ratio, k)
    alpha = _check_alpha(alpha)

    if prior is not None:
        if concentration is not None:
            raise ValueError("pass either prior or concentration, not both")
        prior_vec = np.asarray(list(prior), dtype=float)
        if prior_vec.shape != (k,):
            raise ValueError("prior must have one positive value per arm")
        if not np.all(np.isfinite(prior_vec)) or np.any(prior_vec <= 0):
            raise ValueError("prior values must be positive and finite")
    else:
        conc = float(k if concentration is None else concentration)
        if not math.isfinite(conc) or conc <= 0:
            raise ValueError("concentration must be positive and finite")
        prior_vec = conc * share

    stacked = np.vstack(rows)
    if cumulative:
        if np.any(np.diff(stacked, axis=0) < 0):
            raise ValueError("cumulative counts must be non-decreasing")
        totals = stacked
    else:
        totals = np.cumsum(stacked, axis=0)

    looks: list[SequentialSRMLook] = []
    running_p = 1.0
    for index, row in enumerate(totals, start=1):
        log_bf = _log_bayes_factor(row, share, prior_vec)
        p_now = 1.0 if log_bf <= 0 else float(math.exp(-log_bf))
        running_p = min(running_p, p_now)
        looks.append(
            SequentialSRMLook(
                look=index,
                counts=tuple(int(c) for c in row),
                n=int(row.sum()),
                log_bayes_factor=log_bf,
                p_value=running_p,
                mismatch=bool(running_p < alpha),
            )
        )
    return SequentialSRMResult(
        looks=looks,
        expected_share=tuple(float(s) for s in share),
        prior=tuple(float(a) for a in prior_vec),
        alpha=alpha,
        arm_labels=_labels(arm_labels, k),
    )
