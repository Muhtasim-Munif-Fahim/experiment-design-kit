"""Tests for CUPED variance reduction."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import (
    CUPEDResult,
    adjust_metric,
    cuped_adjust,
    estimate_theta,
    simulate_cuped_data,
)


def _make_data(control_n: int = 40, treatment_n: int = 40, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=control_n + treatment_n)
    outcomes = covariate * 1.5 + rng.normal(scale=0.1, size=control_n + treatment_n)
    treatment = np.array([0] * control_n + [1] * treatment_n)
    return outcomes, treatment, covariate


def test_estimate_theta_recovers_known_slope() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = 2.5 * x + rng.normal(scale=0.05, size=2000)
    assert estimate_theta(y, x) == pytest.approx(2.5, abs=0.05)


def test_estimate_theta_zero_when_covariate_constant() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.ones(4)
    assert estimate_theta(y, x) == 0.0


def test_estimate_theta_rejects_short_or_mismatched_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_theta(np.zeros(4), np.zeros(3))
    with pytest.raises(ValueError, match="at least 2"):
        estimate_theta(np.array([1.0]), np.array([1.0]))


def test_nonfinite_values_raise() -> None:
    treatment = np.array([0, 0, 1, 1])
    finite = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="finite"):
        estimate_theta(np.array([1.0, np.nan, 3.0, 4.0]), finite)
    with pytest.raises(ValueError, match="finite"):
        adjust_metric(finite, np.array([1.0, np.inf, 3.0, 4.0]))
    with pytest.raises(ValueError, match="finite"):
        cuped_adjust(np.array([1.0, 2.0, np.nan, 4.0]), treatment, finite)
    with pytest.raises(ValueError, match="finite"):
        cuped_adjust(finite, np.array([0.0, 0.0, np.nan, 1.0]), finite)


def test_adjust_metric_is_mean_preserving() -> None:
    outcomes, _treatment, covariate = _make_data()
    adjusted = adjust_metric(outcomes, covariate)
    assert float(np.mean(adjusted)) == pytest.approx(float(np.mean(outcomes)), abs=1e-10)


def test_adjust_metric_uses_supplied_theta() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    adjusted = adjust_metric(y, x, theta=0.0)
    np.testing.assert_allclose(adjusted, y)


def test_simulate_cuped_data_shapes_and_correlation() -> None:
    outcomes, treatment, covariates = simulate_cuped_data(
        400, correlation=0.8, treatment_effect=0.0, seed=7
    )
    assert outcomes.shape == treatment.shape == covariates.shape == (800,)
    assert int(np.sum(treatment == 0)) == 400
    assert int(np.sum(treatment == 1)) == 400
    rho = float(np.corrcoef(outcomes, covariates)[0, 1])
    assert rho == pytest.approx(0.8, abs=0.05)


def test_simulate_cuped_data_validates_args() -> None:
    with pytest.raises(ValueError, match="n_per_group"):
        simulate_cuped_data(0)
    with pytest.raises(ValueError, match="correlation"):
        simulate_cuped_data(10, correlation=1.5)
    with pytest.raises(ValueError, match="outcome_sd"):
        simulate_cuped_data(10, outcome_sd=0.0)


def test_cuped_reduces_variance() -> None:
    outcomes, treatment, covariate = simulate_cuped_data(
        80, correlation=0.85, treatment_effect=0.4, seed=0
    )
    result = cuped_adjust(outcomes, treatment, covariate)
    assert isinstance(result, CUPEDResult)
    assert result.variance_reduction > 0.0
    assert result.variance_ratio < 1.0
    assert result.adjusted_variance < result.raw_variance


def test_uncorrelated_covariate_no_reduction() -> None:
    rng = np.random.default_rng(0)
    outcomes = rng.normal(size=80)
    treatment = np.array([0] * 40 + [1] * 40)
    covariate = rng.normal(size=80)
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.variance_reduction == pytest.approx(0.0, abs=0.1)
    assert result.variance_ratio == pytest.approx(1.0, abs=0.1)
    assert result.theta == pytest.approx(0.0, abs=0.2)


def test_raw_effect_matches_raw_difference() -> None:
    outcomes, treatment, covariate = _make_data()
    result = cuped_adjust(outcomes, treatment, covariate)
    raw_t = float(np.mean(outcomes[treatment == 1]))
    raw_c = float(np.mean(outcomes[treatment == 0]))
    assert result.raw_effect == pytest.approx(raw_t - raw_c, abs=1e-6)
    assert result.control_mean == pytest.approx(raw_c, abs=1e-12)
    assert result.treatment_mean == pytest.approx(raw_t, abs=1e-12)


def test_cuped_effect_is_adjusted_mean_difference() -> None:
    outcomes, treatment, covariate = simulate_cuped_data(
        60, correlation=0.75, treatment_effect=1.0, seed=3
    )
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.effect == pytest.approx(
        result.adjusted_treatment_mean - result.adjusted_control_mean, abs=1e-12
    )


def test_cuped_recovers_treatment_effect_on_synthetic_data() -> None:
    true_effect = 1.25
    outcomes, treatment, covariate = simulate_cuped_data(
        800, correlation=0.8, treatment_effect=true_effect, seed=11
    )
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.effect == pytest.approx(true_effect, abs=0.15)
    assert result.raw_effect == pytest.approx(true_effect, abs=0.2)


def test_theta_matches_ols_on_pooled_data() -> None:
    outcomes, treatment, covariate = simulate_cuped_data(
        100, correlation=0.6, treatment_effect=0.3, seed=4
    )
    result = cuped_adjust(outcomes, treatment, covariate, fit_on="pooled")
    assert result.theta == pytest.approx(estimate_theta(outcomes, covariate), abs=1e-12)
    assert result.fit_on == "pooled"


def test_control_only_theta_matches_control_ols() -> None:
    outcomes, treatment, covariate = simulate_cuped_data(
        100, correlation=0.65, treatment_effect=0.5, seed=5
    )
    result = cuped_adjust(outcomes, treatment, covariate, fit_on="control")
    control = treatment == 0
    expected = estimate_theta(outcomes[control], covariate[control])
    assert result.theta == pytest.approx(expected, abs=1e-12)
    assert result.fit_on == "control"


def test_invalid_fit_on_raises() -> None:
    outcomes, treatment, covariate = _make_data()
    with pytest.raises(ValueError, match="fit_on"):
        cuped_adjust(outcomes, treatment, covariate, fit_on="treatment")


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        cuped_adjust(np.zeros(10), np.zeros(10), np.zeros(5))
    with pytest.raises(ValueError, match="same length"):
        cuped_adjust(np.zeros(10), np.zeros(9), np.zeros(10))


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        cuped_adjust(np.array([]), np.array([]), np.array([]))


def test_missing_treatment_group_raises() -> None:
    outcomes = np.array([1.0, 2.0, 3.0])
    treatment = np.array([0, 0, 0])
    covariate = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="treatment"):
        cuped_adjust(outcomes, treatment, covariate)


def test_zero_covariance_returns_unadjusted() -> None:
    rng = np.random.default_rng(0)
    outcomes = rng.normal(size=80)
    treatment = np.array([0] * 40 + [1] * 40)
    covariate = np.ones(80)
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.theta == 0.0
    assert result.variance_reduction == pytest.approx(0.0, abs=1e-6)
    assert result.variance_ratio == pytest.approx(1.0, abs=1e-6)
    assert result.effect == pytest.approx(result.raw_effect, abs=1e-12)


def test_adjusted_mean_close_to_raw_mean() -> None:
    outcomes, treatment, covariate = _make_data()
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.adjusted_mean == pytest.approx(float(np.mean(outcomes)), abs=1e-4)


def test_strong_correlation_gives_strong_reduction() -> None:
    outcomes, treatment, covariate = simulate_cuped_data(
        80, correlation=0.99, treatment_effect=0.0, seed=0
    )
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.variance_reduction > 0.7
    assert result.variance_ratio < 0.3


def test_variance_reduction_tracks_squared_correlation() -> None:
    rho = 0.8
    outcomes, treatment, covariate = simulate_cuped_data(
        1500, correlation=rho, treatment_effect=0.0, seed=21
    )
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.correlation == pytest.approx(rho, abs=0.05)
    assert result.variance_reduction == pytest.approx(rho ** 2, abs=0.08)
    assert result.variance_ratio == pytest.approx(1.0 - result.variance_reduction, abs=1e-12)


def test_cuped_effect_has_lower_sampling_variance() -> None:
    """Across repeated randomised trials, CUPED should shrink ATE variance."""
    raw_effects = []
    cuped_effects = []
    for seed in range(40):
        outcomes, treatment, covariate = simulate_cuped_data(
            120, correlation=0.85, treatment_effect=0.5, seed=seed
        )
        result = cuped_adjust(outcomes, treatment, covariate)
        raw_effects.append(result.raw_effect)
        cuped_effects.append(result.effect)
    assert float(np.var(cuped_effects, ddof=1)) < float(np.var(raw_effects, ddof=1))
    assert float(np.mean(cuped_effects)) == pytest.approx(0.5, abs=0.1)
