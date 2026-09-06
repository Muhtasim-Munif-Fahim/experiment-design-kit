"""Tests for minimum detectable effect calculators."""

from __future__ import annotations

import pytest

from experiment_design_kit import mde as mde_mod
from experiment_design_kit.power import power_two_sample


def test_mde_two_sample_recovers_known_effect() -> None:
    result = mde_mod.minimum_detectable_effect(64, kind="two-sample")
    assert result.effect_size == pytest.approx(0.499, abs=0.02)
    assert result.n_per_group == 64
    assert result.kind == "two-sample"


def test_mde_one_sample_recovers_known_effect() -> None:
    result = mde_mod.minimum_detectable_effect(34, kind="one-sample")
    assert result.effect_size == pytest.approx(0.50, abs=0.02)


def test_mde_raw_scales_by_sigma() -> None:
    d = mde_mod.minimum_detectable_effect(64, kind="two-sample").effect_size
    assert mde_mod.minimum_detectable_effect_raw(64, sigma=10.0) == pytest.approx(d * 10.0, abs=1e-9)


def test_mde_two_sample_consistent_with_power() -> None:
    result = mde_mod.minimum_detectable_effect(64, kind="two-sample", power=0.8)
    achieved = power_two_sample(result.effect_size, 64)
    assert achieved == pytest.approx(0.8, abs=1e-2)


def test_mde_proportion_known_lift() -> None:
    result = mde_mod.minimum_detectable_effect_proportion(3841, 0.10, power=0.8)
    assert result.lift_absolute == pytest.approx(0.02, abs=0.003)
    assert result.lift_relative == pytest.approx(0.20, abs=0.05)
    assert 0.0 < result.effect_size_h < 0.1


def test_mde_proportion_too_small_n_raises() -> None:
    with pytest.raises(ValueError):
        mde_mod.minimum_detectable_effect_proportion(2, 0.10, power=0.99)


def test_mde_invalid_kind() -> None:
    with pytest.raises(ValueError):
        mde_mod.minimum_detectable_effect(64, kind="three-sample")


def test_mde_invalid_power() -> None:
    with pytest.raises(ValueError):
        mde_mod.minimum_detectable_effect(64, power=1.5)
