"""Tests for Mann-Whitney power and sample-size planning."""

from __future__ import annotations

import math

import pytest

from experiment_design_kit.nonparametric import (
    mann_whitney_p_from_shift,
    mann_whitney_power,
    mann_whitney_sample_size,
)


def test_noether_closed_form_matches_known_plan():
    # For p=0.64, alpha=0.05, power=0.8: n ~= (1.96+0.84)^2 / (6*0.14^2) ≈ 66.7
    plan = mann_whitney_sample_size(0.64, alpha=0.05, power=0.8)
    assert plan.n_per_group == pytest.approx(67, abs=2)
    assert plan.achieved_power >= 0.8
    assert plan.n_total == 2 * plan.n_per_group


def test_power_recovers_at_planned_n():
    for p in (0.55, 0.60, 0.70):
        plan = mann_whitney_sample_size(p, power=0.8)
        assert mann_whitney_power(p, plan.n_per_group) >= 0.8
        # n-1 should sit at or just under the target (walk-up from closed form).
        if plan.n_per_group > 2:
            assert mann_whitney_power(p, plan.n_per_group - 1) < 0.8 + 1e-9


def test_power_at_null_is_alpha():
    # As p -> 0.5 power -> alpha; use a tiny effect and small n.
    assert mann_whitney_power(0.51, 10) < 0.1


def test_power_monotonic_in_n_and_p():
    assert mann_whitney_power(0.6, 50) < mann_whitney_power(0.6, 200)
    assert mann_whitney_power(0.55, 100) < mann_whitney_power(0.70, 100)


def test_p_from_shift_normal_matches_closed_form():
    # For N(0,1) vs N(d,1): P(X>Y) = Phi(d / sqrt(2))
    delta = 0.5
    expected = float(__import__("scipy").stats.norm.cdf(delta / math.sqrt(2)))
    estimated = mann_whitney_p_from_shift(delta, family="normal", n_mc=100_000, seed=1)
    assert estimated == pytest.approx(expected, abs=0.01)


def test_rejects_invalid_args():
    with pytest.raises(ValueError):
        mann_whitney_power(0.5, 100)
    with pytest.raises(ValueError):
        mann_whitney_power(0.6, 1)
    with pytest.raises(ValueError):
        mann_whitney_sample_size(0.6, power=1.5)
    with pytest.raises(ValueError):
        mann_whitney_p_from_shift(0.1, family="gamma")
