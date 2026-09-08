"""Tests for Bayesian proportion A/B testing."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import (
    BayesianABTestResult,
    BayesianProportionResult,
    bayesian_ab_test,
    bayesian_proportion_test,
)


def test_single_proportion_posterior_mean() -> None:
    result = bayesian_proportion_test(80, 1000)
    assert 0.07 < result.mean < 0.09
    assert abs(result.ci_lower) < result.mean < result.ci_upper


def test_no_trials_returns_prior_mean() -> None:
    result = bayesian_proportion_test(0, 0, prior_alpha=1.0, prior_beta=1.0)
    assert abs(result.mean - 0.5) < 1e-6


def test_all_successes_high_posterior() -> None:
    result = bayesian_proportion_test(10, 10)
    assert result.mean > 0.9
    assert result.ci_upper > 0.99


def test_invalid_successes_raise() -> None:
    with pytest.raises(ValueError):
        bayesian_proportion_test(-1, 10)
    with pytest.raises(ValueError):
        bayesian_proportion_test(11, 10)


def test_credible_mass_validation() -> None:
    with pytest.raises(ValueError):
        bayesian_proportion_test(5, 10, credible_mass=1.0)
    with pytest.raises(ValueError):
        bayesian_proportion_test(5, 10, credible_mass=0.0)


def test_ab_prob_treatment_better_in_unit_interval() -> None:
    result = bayesian_ab_test(80, 1000, 95, 1000, random_state=0)
    assert 0.0 <= result.prob_treatment_better <= 1.0


def test_ab_prob_treatment_better_reflects_observed_rates() -> None:
    result = bayesian_ab_test(50, 200, 120, 200, random_state=0)
    assert result.prob_treatment_better > 0.9


def test_ab_expected_lift_sign_matches_effect() -> None:
    result = bayesian_ab_test(50, 200, 120, 200, random_state=0)
    assert result.expected_lift > 0.0
    assert result.credible_lift_lower < result.expected_lift < result.credible_lift_upper


def test_ab_control_beats_treatment_reflects_likelihood() -> None:
    result = bayesian_ab_test(120, 200, 50, 200, random_state=0)
    assert result.prob_treatment_better < 0.1
    assert result.expected_lift < 0.0


def test_ab_reproducible_with_seed() -> None:
    a = bayesian_ab_test(80, 1000, 95, 1000, random_state=42)
    b = bayesian_ab_test(80, 1000, 95, 1000, random_state=42)
    assert a.prob_treatment_better == b.prob_treatment_better
    assert a.expected_lift == b.expected_lift


def test_ab_lift_ci_covers_true_lift_approximately() -> None:
    np.random.seed(0)
    control = np.random.binomial(1, 0.10, size=2000)
    treatment = np.random.binomial(1, 0.12, size=2000)
    result = bayesian_ab_test(
        control.sum(), len(control), treatment.sum(), len(treatment),
        random_state=0,
    )
    assert result.credible_lift_lower < 0.02 < result.credible_lift_upper
