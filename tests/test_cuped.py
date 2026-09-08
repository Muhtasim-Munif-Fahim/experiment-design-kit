"""Tests for CUPED variance reduction."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import cuped_adjust


def _make_data(control_n: int = 40, treatment_n: int = 40, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=control_n + treatment_n)
    outcomes = covariate * 1.5 + rng.normal(scale=0.1, size=control_n + treatment_n)
    treatment = np.array([0] * control_n + [1] * treatment_n)
    return outcomes, treatment, covariate


def test_cuped_reduces_variance() -> None:
    outcomes, treatment, covariate = _make_data()
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.variance_reduction < 1.0


def test_uncorrelated_covariate_no_reduction() -> None:
    rng = np.random.default_rng(0)
    outcomes = rng.normal(size=80)
    treatment = np.array([0] * 40 + [1] * 40)
    covariate = rng.normal(size=80)
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.variance_reduction == pytest.approx(1.0, abs=0.1)


def test_effect_matches_raw_difference() -> None:
    outcomes, treatment, covariate = _make_data()
    result = cuped_adjust(outcomes, treatment, covariate)
    raw_t = float(np.mean(outcomes[treatment == 1]))
    raw_c = float(np.mean(outcomes[treatment == 0]))
    assert result.effect == pytest.approx(raw_t - raw_c, abs=1e-6)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        cuped_adjust(np.zeros(10), np.zeros(10), np.zeros(5))


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
    assert result.variance_reduction == pytest.approx(1.0, abs=1e-6)


def test_adjusted_mean_close_to_raw_mean() -> None:
    outcomes, treatment, covariate = _make_data()
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.adjusted_mean == pytest.approx(float(np.mean(outcomes)), abs=1e-4)


def test_strong_correlation_gives_strong_reduction() -> None:
    rng = np.random.default_rng(0)
    covariate = rng.normal(size=80)
    outcomes = covariate * 2.0 + rng.normal(scale=0.01, size=80)
    treatment = np.array([0] * 40 + [1] * 40)
    result = cuped_adjust(outcomes, treatment, covariate)
    assert result.variance_reduction < 0.3
