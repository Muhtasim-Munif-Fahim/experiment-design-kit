"""Tests for Bayesian conversion-test power and sample-size planning."""
from __future__ import annotations

import pytest

from experiment_design_kit import (
    BayesianPowerResult,
    BayesianSampleSizeResult,
    bayesian_power_proportion,
    required_bayesian_sample_size,
)


def _power(**kwargs):
    defaults = dict(
        p_control=0.20,
        p_treatment=0.35,
        n_per_group=200,
        threshold=0.90,
        n_trials=250,
        n_posterior_samples=800,
        random_state=0,
    )
    defaults.update(kwargs)
    return bayesian_power_proportion(**defaults)


def test_power_fields_are_probabilities() -> None:
    result = _power()
    assert isinstance(result, BayesianPowerResult)
    assert 0.0 <= result.power <= 1.0
    assert 0.0 <= result.prob_treatment_wins <= 1.0
    assert 0.0 <= result.prob_control_wins <= 1.0
    assert result.prob_equivalent == 0.0
    parts = (
        result.prob_treatment_wins
        + result.prob_control_wins
        + result.prob_equivalent
        + result.prob_inconclusive
    )
    assert parts == pytest.approx(1.0)
    assert result.power == pytest.approx(1.0 - result.prob_inconclusive)
    assert result.n_per_group == result.n_control == 200


def test_large_lift_high_n_almost_always_declares_treatment() -> None:
    result = _power(p_control=0.20, p_treatment=0.50, n_per_group=400, threshold=0.95)
    assert result.power > 0.9
    assert result.prob_treatment_wins > 0.9
    assert result.prob_control_wins < 0.05
    assert result.mean_prob_treatment_better > 0.95


def test_power_increases_with_sample_size() -> None:
    small = _power(n_per_group=40, n_trials=300)
    large = _power(n_per_group=300, n_trials=300)
    assert large.power > small.power
    assert large.prob_treatment_wins > small.prob_treatment_wins


def test_power_increases_with_lift() -> None:
    modest = _power(p_treatment=0.26, n_per_group=180, n_trials=300)
    large = _power(p_treatment=0.45, n_per_group=180, n_trials=300)
    assert large.prob_treatment_wins > modest.prob_treatment_wins


def test_null_lift_rarely_declares_a_winner_at_large_n() -> None:
    result = _power(p_treatment=0.20, n_per_group=800, threshold=0.95, n_trials=300)
    assert result.power < 0.15
    assert result.prob_inconclusive > 0.85


def test_reproducible_with_seed() -> None:
    a = _power(random_state=11)
    b = _power(random_state=11)
    assert a.power == b.power
    assert a.prob_treatment_wins == b.prob_treatment_wins
    assert a.mean_prob_treatment_better == b.mean_prob_treatment_better


def test_unequal_allocation_ratio() -> None:
    result = _power(n_per_group=120, ratio=2.0)
    assert result.n_control == 120
    assert result.n_treatment == 240


def test_rope_equivalence_under_null() -> None:
    result = bayesian_power_proportion(
        0.20,
        0.20,
        n_per_group=2500,
        decision="rope",
        rope=(-0.04, 0.04),
        threshold=0.90,
        n_trials=200,
        n_posterior_samples=800,
        random_state=1,
    )
    assert result.prob_equivalent > 0.7
    assert result.power == pytest.approx(
        result.prob_equivalent + result.prob_treatment_wins + result.prob_control_wins
    )


def test_rope_detects_lift_outside_interval() -> None:
    result = bayesian_power_proportion(
        0.20,
        0.40,
        n_per_group=400,
        decision="rope",
        rope=(-0.02, 0.02),
        threshold=0.90,
        n_trials=200,
        n_posterior_samples=800,
        random_state=2,
    )
    assert result.prob_treatment_wins > 0.85
    assert result.prob_equivalent < 0.1


def test_relative_rope_scale() -> None:
    result = bayesian_power_proportion(
        0.20,
        0.20,
        n_per_group=2000,
        decision="rope",
        rope=(-0.15, 0.15),
        rope_scale="relative",
        threshold=0.90,
        n_trials=150,
        n_posterior_samples=600,
        random_state=3,
    )
    assert result.rope_scale == "relative"
    assert result.prob_equivalent > 0.5


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="n_per_group"):
        _power(n_per_group=0)
    with pytest.raises(ValueError, match="threshold"):
        _power(threshold=0.5)
    with pytest.raises(ValueError, match="threshold"):
        _power(threshold=1.0)
    with pytest.raises(ValueError, match="p_control"):
        _power(p_control=-0.1)
    with pytest.raises(ValueError, match="decision"):
        _power(decision="p-value")
    with pytest.raises(ValueError, match="rope is required"):
        _power(decision="rope")
    with pytest.raises(ValueError, match="only used"):
        _power(rope=(-0.01, 0.01))
    with pytest.raises(ValueError, match="rope_scale"):
        _power(decision="rope", rope=(-0.01, 0.01), rope_scale="odds")
    with pytest.raises(ValueError, match="prior"):
        _power(prior_alpha=0.0)
    with pytest.raises(ValueError, match="n_trials"):
        _power(n_trials=0)
    with pytest.raises(ValueError, match="n_posterior_samples"):
        _power(n_posterior_samples=10)


def test_required_sample_size_reaches_target() -> None:
    plan = required_bayesian_sample_size(
        0.25,
        0.45,
        target_power=0.75,
        threshold=0.90,
        n_trials=200,
        n_posterior_samples=600,
        random_state=4,
        min_n_per_group=20,
    )
    assert isinstance(plan, BayesianSampleSizeResult)
    assert plan.n_per_group >= 20
    assert plan.n_total == plan.n_control + plan.n_treatment
    assert plan.achieved_power >= 0.75
    check = bayesian_power_proportion(
        0.25,
        0.45,
        plan.n_per_group,
        threshold=0.90,
        n_trials=250,
        n_posterior_samples=600,
        random_state=99,
    )
    assert check.power >= 0.65


def test_required_sample_size_rejects_null_threshold_search() -> None:
    with pytest.raises(ValueError, match="non-zero assumed lift"):
        required_bayesian_sample_size(0.1, 0.1, n_trials=50, n_posterior_samples=80)


def test_required_sample_size_rejects_unreachable_target() -> None:
    with pytest.raises(ValueError, match="max_n_per_group"):
        required_bayesian_sample_size(
            0.10,
            0.11,
            target_power=0.99,
            threshold=0.99,
            n_trials=40,
            n_posterior_samples=80,
            random_state=0,
            max_n_per_group=30,
            min_n_per_group=20,
        )


def test_stricter_threshold_needs_more_n() -> None:
    loose = required_bayesian_sample_size(
        0.22,
        0.40,
        target_power=0.70,
        threshold=0.85,
        n_trials=160,
        n_posterior_samples=500,
        random_state=5,
        min_n_per_group=20,
    )
    strict = required_bayesian_sample_size(
        0.22,
        0.40,
        target_power=0.70,
        threshold=0.97,
        n_trials=160,
        n_posterior_samples=500,
        random_state=5,
        min_n_per_group=20,
    )
    assert strict.n_per_group >= loose.n_per_group


def test_rope_sample_size_plans_for_equivalence() -> None:
    plan = required_bayesian_sample_size(
        0.30,
        0.30,
        target_power=0.70,
        threshold=0.90,
        decision="rope",
        rope=(-0.08, 0.08),
        n_trials=150,
        n_posterior_samples=500,
        random_state=6,
        min_n_per_group=40,
    )
    assert plan.decision == "rope"
    assert plan.achieved_power >= 0.70
    assert plan.power_result.prob_equivalent >= 0.55
