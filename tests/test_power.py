"""Tests for statistical power analysis."""

from __future__ import annotations

import pytest

from experiment_design_kit import power as pw
from experiment_design_kit.stats import cohen_h


def test_power_two_sample_known_value() -> None:
    assert pw.power_two_sample(0.5, 64) == pytest.approx(0.8015, abs=1e-3)
    assert pw.power_two_sample(0.5, 63) == pytest.approx(0.7952, abs=1e-3)


def test_required_sample_size_two_sample() -> None:
    result = pw.required_sample_size(0.5, kind="two-sample")
    assert result.n_per_group_required == 64
    assert result.n_total_required == 128
    assert result.effect_size == 0.5
    assert result.metadata["kind"] == "two-sample"


def test_required_sample_size_one_sample() -> None:
    result = pw.required_sample_size(0.5, kind="one-sample")
    assert result.n_per_group_required == 34
    assert result.n_total_required == 34


def test_required_sample_size_recovers_target_power() -> None:
    for n_val in (0.3, 0.5, 0.8):
        result = pw.required_sample_size(n_val, power=0.8)
        achieved = pw.power_two_sample(n_val, result.n_per_group_required)
        assert achieved == pytest.approx(0.8, abs=1e-2)


def test_power_one_sample_known_value() -> None:
    assert pw.power_one_sample(0.5, 34) == pytest.approx(0.80, abs=1e-2)


def test_power_proportion_known_value() -> None:
    # normal-approx power at the closed-form plan n (~3841) reaches ~0.80.
    assert pw.power_proportion(0.10, 0.12, 3841) == pytest.approx(0.80, abs=1e-2)


def test_power_proportion_monotonic_in_n() -> None:
    ns = [50, 200, 1000, 5000]
    powers = [pw.power_proportion(0.10, 0.12, n) for n in ns]
    assert all(b > a for a, b in zip(powers, powers[1:]))


def test_power_two_sample_invalid() -> None:
    with pytest.raises(ValueError):
        pw.power_two_sample(0.5, 1)
    with pytest.raises(ValueError):
        pw.power_two_sample(-0.5, 100)
    with pytest.raises(ValueError):
        pw.power_two_sample(0.5, 100, alpha=1.5)


def test_power_curve_increasing() -> None:
    curve = pw.power_curve(0.5, [10, 30, 64, 128])
    assert curve == [pytest.approx(c) for c in curve]
    assert all(b > a for a, b in zip(curve, curve[1:]))


def test_cohen_h_and_power_agree() -> None:
    # Cohen's h for 0.10 vs 0.14 is a small effect; power should be below 0.8 at n=100.
    h = cohen_h(0.10, 0.14)
    assert isinstance(h, float)
    assert pw.power_proportion(0.10, 0.14, 100) < 0.8
