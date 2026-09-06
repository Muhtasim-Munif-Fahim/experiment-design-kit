"""Tests for the A/B test simulator and significance checkers."""

from __future__ import annotations

import pytest

from experiment_design_kit import simulation as sim


def test_simulate_proportion_is_reproducible() -> None:
    a = sim.simulate_proportion(0.10, 0.14, 5000, seed=7)
    b = sim.simulate_proportion(0.10, 0.14, 5000, seed=7)
    assert a == b
    assert 0.0 <= a.control_rate <= 1.0
    assert 0.0 <= a.treatment_rate <= 1.0
    assert a.lift_absolute == a.treatment_rate - a.control_rate


def test_simulate_proportion_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        sim.simulate_proportion(1.5, 0.14, 100)


def test_chi_square_significant_table() -> None:
    result = sim.chi_square_significance(40, 100, 60, 100)
    assert result.dof == 1
    assert result.p_value < 0.05
    assert result.significant is True
    assert result.cramers_v > 0


def test_chi_square_null_table_not_significant() -> None:
    result = sim.chi_square_significance(50, 100, 50, 100)
    assert result.p_value == pytest.approx(1.0)
    assert result.significant is False
    assert result.cramers_v == pytest.approx(0.0)


def test_t_test_significance_clear_difference() -> None:
    control = [1.0, 2.0, 3.0, 4.0, 5.0]
    treatment = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = sim.t_test_significance(control, treatment)
    assert result.p_value < 0.05
    assert result.df == pytest.approx(8.0)
    assert result.se > 0


def test_t_test_identical_samples_not_significant() -> None:
    control = [1.0, 2.0, 3.0, 4.0]
    treatment = [1.0, 2.0, 3.0, 4.0]
    result = sim.t_test_significance(control, treatment)
    assert result.p_value == pytest.approx(1.0)


def test_t_test_significance_requires_min_size() -> None:
    with pytest.raises(ValueError):
        sim.t_test_significance([1.0], [2.0])


def test_run_proportion_ab_test_reproducible_and_significant() -> None:
    a = sim.run_proportion_ab_test(0.10, 0.14, 20000, seed=42)
    b = sim.run_proportion_ab_test(0.10, 0.14, 20000, seed=42)
    assert a == b
    assert a.metric == "proportion"
    assert a.n_per_group == 20000
    assert a.significant is True
    assert a.p_value < 0.05
    assert a.lift_relative > 0
    assert a.effect_size > 0


def test_run_proportion_ab_test_null_effect_not_significant() -> None:
    result = sim.run_proportion_ab_test(0.10, 0.10, 500, seed=1)
    assert result.significant is False


def test_run_continuous_ab_test_reproducible_and_significant() -> None:
    a = sim.run_continuous_ab_test(mean_control=100.0, sd=15.0, treatment_effect=10.0, n_per_group=2000, seed=3)
    b = sim.run_continuous_ab_test(mean_control=100.0, sd=15.0, treatment_effect=10.0, n_per_group=2000, seed=3)
    assert a == b
    assert a.metric == "continuous"
    assert a.significant is True
    assert a.lift_absolute > 0
    assert a.effect_size > 0
