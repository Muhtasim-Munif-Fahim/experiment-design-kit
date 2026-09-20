"""Bayesian A/B testing with Beta-Binomial posteriors.

Analysis helpers summarise a single observed conversion experiment. Planning
helpers estimate *Bayesian power* — the Monte Carlo probability of reaching a
decision under an assumed lift — and search for a sample size that hits a
target decision rate.

Two decision rules are supported:

* **threshold**: declare a winner when ``P(treatment > control)`` (or the
  reverse) exceeds a posterior probability threshold (default 0.95).
* **ROPE**: declare superiority, inferiority, or practical equivalence when
  the posterior mass of the lift above, below, or inside a region of
  practical equivalence exceeds the same threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp

_THRESHOLD = "threshold"
_ROPE = "rope"
_DECISIONS = {_THRESHOLD, _ROPE}
_ABSOLUTE = "absolute"
_RELATIVE = "relative"
_SCALES = {_ABSOLUTE, _RELATIVE}

_WIN_TREATMENT = 1
_WIN_CONTROL = -1
_EQUIVALENT = 2
_INCONCLUSIVE = 0


@dataclass(frozen=True)
class BayesianProportionResult:
    """Posterior summary for a single variant's conversion rate."""

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    credible_mass: float


@dataclass(frozen=True)
class BayesianABTestResult:
    """Head-to-head Bayesian A/B test for two proportions."""

    control: BayesianProportionResult
    treatment: BayesianProportionResult
    prob_treatment_better: float
    expected_lift: float
    credible_lift_lower: float
    credible_lift_upper: float
    credible_mass: float


def bayesian_proportion_test(
    successes: int,
    trials: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_mass: float = 0.95,
) -> BayesianProportionResult:
    """Return the posterior summary for a single binomial proportion.

    Uses a Beta(prior_alpha, prior_beta) prior and updates it with the
    observed successes/trials to obtain a Beta(alpha', beta') posterior.
    The credible interval is a highest-density interval computed via
    ``scipy.stats.beta.ppf`` (equal-tailed).

    Parameters
    ----------
    successes:
        Number of observed successes (e.g. conversions).
    trials:
        Total number of trials (e.g. visitors). Must be non-negative.
    prior_alpha, prior_beta:
        Beta prior hyperparameters. Uniform prior = (1, 1).
    credible_mass:
        Desired coverage of the credible interval (0 < mass < 1).
    """
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must be in (0, 1)")

    alpha_post = prior_alpha + int(successes)
    beta_post = prior_beta + int(trials) - int(successes)
    dist = sp.beta(alpha_post, beta_post)
    mean = float(dist.mean())
    std = float(dist.std())
    tail = (1.0 - credible_mass) / 2.0
    ci_lower = float(dist.ppf(tail))
    ci_upper = float(dist.ppf(1.0 - tail))
    return BayesianProportionResult(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        credible_mass=credible_mass,
    )


def bayesian_ab_test(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_mass: float = 0.95,
    n_samples: int = 100_000,
    random_state: Optional[int] = None,
) -> BayesianABTestResult:
    """Compare two proportions with a Bayesian Beta-Binomial model.

    Draws posterior samples for both variants and estimates the probability
    that the treatment's conversion rate exceeds the control's. Also returns
    the expected lift and a credible interval for the lift.

    Parameters
    ----------
    control_successes, control_trials:
        Observed successes and trials for the control variant.
    treatment_successes, treatment_trials:
        Observed successes and trials for the treatment variant.
    prior_alpha, prior_beta:
        Shared Beta prior hyperparameters (uniform = 1, 1).
    credible_mass:
        Coverage for the lift credible interval.
    n_samples:
        Number of Monte Carlo posterior samples.
    random_state:
        Seed for reproducibility.
    """
    control = bayesian_proportion_test(
        control_successes, control_trials,
        prior_alpha=prior_alpha, prior_beta=prior_beta,
        credible_mass=credible_mass,
    )
    treatment = bayesian_proportion_test(
        treatment_successes, treatment_trials,
        prior_alpha=prior_alpha, prior_beta=prior_beta,
        credible_mass=credible_mass,
    )

    rng = np.random.default_rng(random_state)
    c_post_alpha = prior_alpha + control_successes
    c_post_beta = prior_beta + control_trials - control_successes
    t_post_alpha = prior_alpha + treatment_successes
    t_post_beta = prior_beta + treatment_trials - treatment_successes

    control_samples = rng.beta(c_post_alpha, c_post_beta, size=n_samples)
    treatment_samples = rng.beta(t_post_alpha, t_post_beta, size=n_samples)
    lift_samples = (treatment_samples - control_samples) / np.clip(control_samples, 1e-12, None)

    prob_better = float(np.mean(treatment_samples > control_samples))
    expected_lift = float(np.mean(lift_samples))
    tail = (1.0 - credible_mass) / 2.0
    lift_lower = float(np.percentile(lift_samples, tail * 100))
    lift_upper = float(np.percentile(lift_samples, (1.0 - tail) * 100))

    return BayesianABTestResult(
        control=control,
        treatment=treatment,
        prob_treatment_better=prob_better,
        expected_lift=expected_lift,
        credible_lift_lower=lift_lower,
        credible_lift_upper=lift_upper,
        credible_mass=credible_mass,
    )


@dataclass(frozen=True)
class BayesianPowerResult:
    """Monte Carlo Bayesian decision power for a two-arm conversion test."""

    p_control: float
    p_treatment: float
    n_control: int
    n_treatment: int
    n_trials: int
    n_posterior_samples: int
    threshold: float
    decision: str
    rope_lower: Optional[float]
    rope_upper: Optional[float]
    rope_scale: str
    prior_alpha: float
    prior_beta: float
    power: float
    prob_treatment_wins: float
    prob_control_wins: float
    prob_equivalent: float
    prob_inconclusive: float
    mean_prob_treatment_better: float
    seed: Optional[int]

    @property
    def n_per_group(self) -> int:
        return self.n_control


@dataclass(frozen=True)
class BayesianSampleSizeResult:
    """Smallest per-group n whose simulated Bayesian power meets a target."""

    n_per_group: int
    n_total: int
    n_control: int
    n_treatment: int
    achieved_power: float
    target_power: float
    p_control: float
    p_treatment: float
    threshold: float
    decision: str
    rope_lower: Optional[float]
    rope_upper: Optional[float]
    rope_scale: str
    n_trials: int
    seed: Optional[int]
    power_result: BayesianPowerResult


def bayesian_power_proportion(
    p_control: float,
    p_treatment: float,
    n_per_group: int,
    *,
    threshold: float = 0.95,
    decision: str = _THRESHOLD,
    rope: Optional[tuple[float, float]] = None,
    rope_scale: str = _ABSOLUTE,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    ratio: float = 1.0,
    n_trials: int = 1000,
    n_posterior_samples: int = 2000,
    random_state: Optional[int] = None,
) -> BayesianPowerResult:
    """Estimate Bayesian power for a two-arm conversion test by simulation.

    Each Monte Carlo trial draws binomial conversion counts under the assumed
    rates, updates independent Beta posteriors, and applies a decision rule:

    * ``decision="threshold"`` — decide if ``P(θ_t > θ_c) ≥ threshold`` or
      ``P(θ_c > θ_t) ≥ threshold``.
    * ``decision="rope"`` — decide if the posterior mass of the lift above
      the ROPE, below the ROPE, or inside the ROPE is at least ``threshold``.
      ``rope`` is ``(lower, upper)`` on the absolute difference
      ``θ_t - θ_c`` or the relative lift ``(θ_t - θ_c) / θ_c``.

    ``power`` is the share of trials that reach any of those decisions
    (winner or, for ROPE, practical equivalence). The breakdown
    ``prob_treatment_wins`` / ``prob_control_wins`` / ``prob_equivalent``
    is also returned.

    Parameters
    ----------
    p_control, p_treatment:
        Assumed true conversion rates (the design lift is ``p_treatment -
        p_control``).
    n_per_group:
        Control-arm sample size. The treatment arm is
        ``round(n_per_group * ratio)``.
    threshold:
        Posterior probability needed to decide. Must be in ``(0.5, 1)`` so
        the three ROPE/threshold events are mutually exclusive.
    decision:
        ``"threshold"`` or ``"rope"``.
    rope:
        Required when ``decision="rope"``. Inclusive bounds on the lift.
    rope_scale:
        ``"absolute"`` (default) or ``"relative"``.
    prior_alpha, prior_beta:
        Shared Beta prior hyperparameters (uniform = 1, 1).
    ratio:
        ``n_treatment / n_control``.
    n_trials:
        Number of simulated experiments.
    n_posterior_samples:
        Posterior draws per experiment used to estimate decision
        probabilities.
    random_state:
        Seed for reproducibility.
    """
    decision_key = _normalize_decision(decision)
    scale_key = _normalize_rope_scale(rope_scale)
    rope_bounds = _validate_power_args(
        p_control=p_control,
        p_treatment=p_treatment,
        n_per_group=n_per_group,
        threshold=threshold,
        decision=decision_key,
        rope=rope,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        ratio=ratio,
        n_trials=n_trials,
        n_posterior_samples=n_posterior_samples,
    )
    n_control = int(n_per_group)
    n_treatment = max(1, int(round(n_per_group * ratio)))
    rng = np.random.default_rng(random_state)
    labels, p_better = _simulate_decisions(
        rng,
        p_control=p_control,
        p_treatment=p_treatment,
        n_control=n_control,
        n_treatment=n_treatment,
        threshold=threshold,
        decision=decision_key,
        rope=rope_bounds,
        rope_scale=scale_key,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        n_trials=n_trials,
        n_posterior_samples=n_posterior_samples,
    )
    n = float(n_trials)
    p_treat = float(np.mean(labels == _WIN_TREATMENT))
    p_ctrl = float(np.mean(labels == _WIN_CONTROL))
    p_eq = float(np.mean(labels == _EQUIVALENT))
    p_none = float(np.mean(labels == _INCONCLUSIVE))
    return BayesianPowerResult(
        p_control=float(p_control),
        p_treatment=float(p_treatment),
        n_control=n_control,
        n_treatment=n_treatment,
        n_trials=n_trials,
        n_posterior_samples=n_posterior_samples,
        threshold=float(threshold),
        decision=decision_key,
        rope_lower=None if rope_bounds is None else float(rope_bounds[0]),
        rope_upper=None if rope_bounds is None else float(rope_bounds[1]),
        rope_scale=scale_key,
        prior_alpha=float(prior_alpha),
        prior_beta=float(prior_beta),
        power=float((n - np.sum(labels == _INCONCLUSIVE)) / n),
        prob_treatment_wins=p_treat,
        prob_control_wins=p_ctrl,
        prob_equivalent=p_eq,
        prob_inconclusive=p_none,
        mean_prob_treatment_better=float(np.mean(p_better)),
        seed=random_state,
    )


def required_bayesian_sample_size(
    p_control: float,
    p_treatment: float,
    *,
    target_power: float = 0.8,
    threshold: float = 0.95,
    decision: str = _THRESHOLD,
    rope: Optional[tuple[float, float]] = None,
    rope_scale: str = _ABSOLUTE,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    ratio: float = 1.0,
    n_trials: int = 800,
    n_posterior_samples: int = 1500,
    random_state: Optional[int] = None,
    max_n_per_group: int = 1_000_000,
    min_n_per_group: int = 10,
) -> BayesianSampleSizeResult:
    """Smallest per-group n whose simulated Bayesian power meets ``target_power``.

    Uses exponential growth followed by binary search. Each candidate ``n``
    is evaluated with :func:`bayesian_power_proportion`. Monte Carlo error
    can make the empirical power curve slightly non-monotone, so the search
    walks upward a few steps if the binary-search candidate undershoots.

    A threshold-rule search requires a non-zero assumed lift: under equal
    rates the probability of declaring a winner shrinks toward 0 as ``n``
    grows. A ROPE search is valid for a zero lift (it plans for equivalence).
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if max_n_per_group < min_n_per_group:
        raise ValueError("max_n_per_group must be >= min_n_per_group")
    if min_n_per_group < 1:
        raise ValueError("min_n_per_group must be at least 1")
    decision_key = _normalize_decision(decision)
    if decision_key == _THRESHOLD and abs(p_treatment - p_control) == 0.0:
        raise ValueError(
            "threshold-rule sample-size search requires a non-zero assumed lift"
        )

    def _eval(n: int) -> BayesianPowerResult:
        seed = None if random_state is None else int(random_state) + n
        return bayesian_power_proportion(
            p_control,
            p_treatment,
            n,
            threshold=threshold,
            decision=decision_key,
            rope=rope,
            rope_scale=rope_scale,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            ratio=ratio,
            n_trials=n_trials,
            n_posterior_samples=n_posterior_samples,
            random_state=seed,
        )

    n = int(min_n_per_group)
    result = _eval(n)
    while result.power < target_power:
        if n >= max_n_per_group:
            raise ValueError(
                "max_n_per_group is too small to reach the requested Bayesian power"
            )
        n = min(max(n * 2, n + 1), max_n_per_group)
        result = _eval(n)

    lo = max(int(min_n_per_group), n // 2)
    hi = n
    best = result
    while lo < hi:
        mid = (lo + hi) // 2
        cand = _eval(mid)
        if cand.power >= target_power:
            hi = mid
            best = cand
        else:
            lo = mid + 1
    best = _eval(lo)
    if best.power < target_power:
        step = max(lo // 20, 1)
        n_walk = lo
        while best.power < target_power:
            n_walk = min(n_walk + step, max_n_per_group)
            best = _eval(n_walk)
            if n_walk >= max_n_per_group and best.power < target_power:
                raise ValueError(
                    "max_n_per_group is too small to reach the requested Bayesian power"
                )
            if n_walk == max_n_per_group:
                break
            lo = n_walk
    return BayesianSampleSizeResult(
        n_per_group=best.n_control,
        n_total=best.n_control + best.n_treatment,
        n_control=best.n_control,
        n_treatment=best.n_treatment,
        achieved_power=best.power,
        target_power=float(target_power),
        p_control=float(p_control),
        p_treatment=float(p_treatment),
        threshold=float(threshold),
        decision=best.decision,
        rope_lower=best.rope_lower,
        rope_upper=best.rope_upper,
        rope_scale=best.rope_scale,
        n_trials=n_trials,
        seed=random_state,
        power_result=best,
    )


def _simulate_decisions(
    rng: np.random.Generator,
    *,
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    threshold: float,
    decision: str,
    rope: Optional[tuple[float, float]],
    rope_scale: str,
    prior_alpha: float,
    prior_beta: float,
    n_trials: int,
    n_posterior_samples: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    ctrl_s = rng.binomial(n_control, p_control, size=n_trials)
    trt_s = rng.binomial(n_treatment, p_treatment, size=n_trials)
    labels = np.empty(n_trials, dtype=np.int8)
    p_better = np.empty(n_trials, dtype=float)
    for start in range(0, n_trials, batch_size):
        stop = min(start + batch_size, n_trials)
        sl = slice(start, stop)
        labels[sl], p_better[sl] = _decide_batch(
            rng,
            ctrl_s[sl],
            trt_s[sl],
            n_control=n_control,
            n_treatment=n_treatment,
            threshold=threshold,
            decision=decision,
            rope=rope,
            rope_scale=rope_scale,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            n_posterior_samples=n_posterior_samples,
        )
    return labels, p_better


def _decide_batch(
    rng: np.random.Generator,
    ctrl_s: np.ndarray,
    trt_s: np.ndarray,
    *,
    n_control: int,
    n_treatment: int,
    threshold: float,
    decision: str,
    rope: Optional[tuple[float, float]],
    rope_scale: str,
    prior_alpha: float,
    prior_beta: float,
    n_posterior_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    ctrl_alpha = prior_alpha + ctrl_s.astype(float)
    ctrl_beta = prior_beta + float(n_control) - ctrl_s.astype(float)
    trt_alpha = prior_alpha + trt_s.astype(float)
    trt_beta = prior_beta + float(n_treatment) - trt_s.astype(float)
    n_batch = int(ctrl_s.shape[0])
    shape = (n_batch, n_posterior_samples)
    ctrl_post = rng.beta(
        np.broadcast_to(ctrl_alpha[:, None], shape),
        np.broadcast_to(ctrl_beta[:, None], shape),
    )
    trt_post = rng.beta(
        np.broadcast_to(trt_alpha[:, None], shape),
        np.broadcast_to(trt_beta[:, None], shape),
    )
    p_better = np.mean(trt_post > ctrl_post, axis=1)
    labels = np.zeros(n_batch, dtype=np.int8)
    if decision == _THRESHOLD:
        labels[p_better >= threshold] = _WIN_TREATMENT
        labels[p_better <= (1.0 - threshold)] = _WIN_CONTROL
        return labels, p_better

    assert rope is not None
    lift = trt_post - ctrl_post
    if rope_scale == _RELATIVE:
        lift = lift / np.clip(ctrl_post, 1e-12, None)
    p_above = np.mean(lift > rope[1], axis=1)
    p_below = np.mean(lift < rope[0], axis=1)
    p_inside = np.mean((lift >= rope[0]) & (lift <= rope[1]), axis=1)
    labels[p_above >= threshold] = _WIN_TREATMENT
    labels[p_below >= threshold] = _WIN_CONTROL
    labels[p_inside >= threshold] = _EQUIVALENT
    return labels, p_better


def _normalize_decision(decision: str) -> str:
    key = decision.strip().lower().replace("_", "-")
    if key not in _DECISIONS:
        raise ValueError("decision must be 'threshold' or 'rope'")
    return key


def _normalize_rope_scale(rope_scale: str) -> str:
    key = rope_scale.strip().lower()
    if key not in _SCALES:
        raise ValueError("rope_scale must be 'absolute' or 'relative'")
    return key


def _validate_power_args(
    *,
    p_control: float,
    p_treatment: float,
    n_per_group: int,
    threshold: float,
    decision: str,
    rope: Optional[tuple[float, float]],
    prior_alpha: float,
    prior_beta: float,
    ratio: float,
    n_trials: int,
    n_posterior_samples: int,
) -> Optional[tuple[float, float]]:
    for label, value in (("p_control", p_control), ("p_treatment", p_treatment)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be in [0, 1]")
    if n_per_group < 1:
        raise ValueError("n_per_group must be at least 1")
    if not 0.5 < threshold < 1.0:
        raise ValueError("threshold must be in (0.5, 1)")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_posterior_samples < 50:
        raise ValueError("n_posterior_samples must be at least 50")
    if decision == _ROPE:
        if rope is None:
            raise ValueError("rope is required when decision='rope'")
        if len(rope) != 2:
            raise ValueError("rope must be a (lower, upper) pair")
        lower, upper = float(rope[0]), float(rope[1])
        if lower >= upper:
            raise ValueError("rope lower bound must be < upper bound")
        return (lower, upper)
    if rope is not None:
        raise ValueError("rope is only used when decision='rope'")
    return None


__all__ = [
    "BayesianProportionResult",
    "BayesianABTestResult",
    "BayesianPowerResult",
    "BayesianSampleSizeResult",
    "bayesian_proportion_test",
    "bayesian_ab_test",
    "bayesian_power_proportion",
    "required_bayesian_sample_size",
]
