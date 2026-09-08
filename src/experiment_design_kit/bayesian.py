"""Bayesian A/B testing with Beta-Binomial posteriors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp


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


__all__ = [
    "BayesianProportionResult",
    "BayesianABTestResult",
    "bayesian_proportion_test",
    "bayesian_ab_test",
]
