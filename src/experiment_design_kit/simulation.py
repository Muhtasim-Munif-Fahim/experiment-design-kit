"""Seedable A/B test outcome simulator and significance checking.

Two experiment flavours are supported:

* **proportion** metrics (e.g. conversion rate) simulated from Bernoulli
  draws and tested with a Pearson chi-square test of independence on the 2x2
  contingency table of successes vs failures.
* **continuous** metrics (e.g. revenue per user) simulated from normal draws
  and tested with Welch's two-sample t-test.

Every simulator is driven by an explicit ``seed`` so an outcome can be
reproduced verbatim, and each significance function returns a p-value, the
test statistic, degrees of freedom, and a significance flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp

from .stats import TTestResult, cohen_h, cohens_d


@dataclass(frozen=True)
class SimulatedProportionOutcome:
    n_per_group: int
    seed: int
    control_successes: int
    treatment_successes: int
    control_rate: float
    treatment_rate: float
    lift_absolute: float
    lift_relative: float


@dataclass(frozen=True)
class SimulatedContinuousOutcome:
    n_per_group: int
    seed: int
    control_mean: float
    treatment_mean: float
    control_sd: float
    treatment_sd: float
    lift_absolute: float
    lift_relative: float


@dataclass(frozen=True)
class ChiSquareResult:
    statistic: float
    p_value: float
    dof: int
    significant: bool
    alpha: float
    cramers_v: float


@dataclass(frozen=True)
class ABTestResult:
    metric: str
    control: float
    treatment: float
    n_per_group: int
    alpha: float
    seed: int
    control_estimate: float
    treatment_estimate: float
    lift_absolute: float
    lift_relative: float
    statistic: float
    p_value: float
    dof: float
    significant: bool
    effect_size: float


def simulate_proportion(
    p_control: float,
    p_treatment: float,
    n_per_group: int,
    seed: int = 42,
) -> SimulatedProportionOutcome:
    """Simulate two independent Bernoulli samples of size ``n_per_group``."""
    _require_rate(p_control, "p_control")
    _require_rate(p_treatment, "p_treatment")
    _require_positive(n_per_group, "n_per_group")
    rng = np.random.default_rng(seed)
    control = rng.binomial(1, p_control, size=n_per_group)
    treatment = rng.binomial(1, p_treatment, size=n_per_group)
    c_succ = int(control.sum())
    t_succ = int(treatment.sum())
    c_rate = c_succ / n_per_group
    t_rate = t_succ / n_per_group
    lift = t_rate - c_rate
    return SimulatedProportionOutcome(
        n_per_group=n_per_group,
        seed=seed,
        control_successes=c_succ,
        treatment_successes=t_succ,
        control_rate=c_rate,
        treatment_rate=t_rate,
        lift_absolute=lift,
        lift_relative=lift / c_rate if c_rate > 0 else math.inf,
    )


def _draw_continuous(mean_control, sd, treatment_effect, n_per_group, seed):
    _require_positive(sd, "sd")
    _require_positive(n_per_group, "n_per_group")
    rng = np.random.default_rng(seed)
    control = rng.normal(mean_control, sd, size=n_per_group)
    treatment = rng.normal(mean_control + treatment_effect, sd, size=n_per_group)
    return control, treatment


def simulate_continuous(
    mean_control: float,
    sd: float,
    treatment_effect: float,
    n_per_group: int,
    seed: int = 42,
) -> SimulatedContinuousOutcome:
    """Simulate two independent normal samples separated by ``treatment_effect``."""
    control, treatment = _draw_continuous(mean_control, sd, treatment_effect, n_per_group, seed)
    c_mean = float(control.mean())
    t_mean = float(treatment.mean())
    c_sd = float(control.std(ddof=1))
    t_sd = float(treatment.std(ddof=1))
    lift = t_mean - c_mean
    return SimulatedContinuousOutcome(
        n_per_group=n_per_group,
        seed=seed,
        control_mean=c_mean,
        treatment_mean=t_mean,
        control_sd=c_sd,
        treatment_sd=t_sd,
        lift_absolute=lift,
        lift_relative=lift / abs(c_mean) if c_mean != 0 else math.inf,
    )


def chi_square_significance(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Pearson chi-square test of independence on a 2x2 contingency table."""
    _require_non_negative(control_successes, "control_successes")
    _require_non_negative(treatment_successes, "treatment_successes")
    for label, value in (("control_n", control_n), ("treatment_n", treatment_n)):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if control_successes > control_n or treatment_successes > treatment_n:
        raise ValueError("successes cannot exceed the group size")
    table = [
        [control_successes, control_n - control_successes],
        [treatment_successes, treatment_n - treatment_successes],
    ]
    chi2, p_value, dof, _expected = sp.chi2_contingency(table)
    n_total = control_n + treatment_n
    cramers_v = math.sqrt(chi2 / n_total) if n_total > 0 else 0.0
    return ChiSquareResult(
        statistic=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        significant=bool(p_value < alpha),
        alpha=alpha,
        cramers_v=cramers_v,
    )


def t_test_significance(
    control: np.ndarray | list[float],
    treatment: np.ndarray | list[float],
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TTestResult:
    """Welch (default) or Student's two-sample t-test on raw samples."""
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each sample must contain at least 2 observations")
    res = sp.ttest_ind(a, b, equal_var=equal_var)
    var1 = float(np.var(a, ddof=1))
    var2 = float(np.var(b, ddof=1))
    se = math.sqrt(var1 / a.size + var2 / b.size)
    df = float(getattr(res, "df", a.size + b.size - 2))
    return TTestResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        df=df,
        se=se,
    )


def run_proportion_ab_test(
    p_control: float,
    p_treatment: float,
    n_per_group: int,
    alpha: float = 0.05,
    seed: int = 42,
) -> ABTestResult:
    """Simulate a conversion-rate A/B test and test it with chi-square."""
    outcome = simulate_proportion(p_control, p_treatment, n_per_group, seed)
    test = chi_square_significance(
        outcome.control_successes,
        n_per_group,
        outcome.treatment_successes,
        n_per_group,
        alpha=alpha,
    )
    return ABTestResult(
        metric="proportion",
        control=p_control,
        treatment=p_treatment,
        n_per_group=n_per_group,
        alpha=alpha,
        seed=seed,
        control_estimate=outcome.control_rate,
        treatment_estimate=outcome.treatment_rate,
        lift_absolute=outcome.lift_absolute,
        lift_relative=outcome.lift_relative,
        statistic=test.statistic,
        p_value=test.p_value,
        dof=float(test.dof),
        significant=test.significant,
        effect_size=cohen_h(outcome.control_rate, outcome.treatment_rate),
    )


def run_continuous_ab_test(
    mean_control: float,
    sd: float,
    treatment_effect: float,
    n_per_group: int,
    alpha: float = 0.05,
    seed: int = 42,
) -> ABTestResult:
    """Simulate a continuous-metric A/B test and test it with Welch's t-test."""
    control, treatment = _draw_continuous(mean_control, sd, treatment_effect, n_per_group, seed)
    outcome = simulate_continuous(mean_control, sd, treatment_effect, n_per_group, seed)
    test = t_test_significance(control, treatment, alpha=alpha, equal_var=False)
    d = cohens_d(outcome.treatment_mean, outcome.control_mean, outcome.treatment_sd, outcome.control_sd)
    return ABTestResult(
        metric="continuous",
        control=mean_control,
        treatment=mean_control + treatment_effect,
        n_per_group=n_per_group,
        alpha=alpha,
        seed=seed,
        control_estimate=outcome.control_mean,
        treatment_estimate=outcome.treatment_mean,
        lift_absolute=outcome.lift_absolute,
        lift_relative=outcome.lift_relative,
        statistic=test.statistic,
        p_value=test.p_value,
        dof=test.df,
        significant=test.p_value < alpha,
        effect_size=d,
    )


def _require_rate(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")


def _require_positive(value: float, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be positive")


def _require_non_negative(value: int, label: str) -> None:
    if value < 0:
        raise ValueError(f"{label} must be non-negative")


__all__ = [
    "ABTestResult",
    "ChiSquareResult",
    "SimulatedContinuousOutcome",
    "SimulatedProportionOutcome",
    "chi_square_significance",
    "run_continuous_ab_test",
    "run_proportion_ab_test",
    "simulate_continuous",
    "simulate_proportion",
    "t_test_significance",
]
