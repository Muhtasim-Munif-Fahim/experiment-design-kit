"""Tests for sample-size ratio optimization."""
from __future__ import annotations


import pytest

from experiment_design_kit.stats import sample_size_ratio


def test_ratio_equal_proportions_raises() -> None:
    with pytest.raises(ValueError, match="must differ"):
        sample_size_ratio(0.5, 0.5, 100)


def test_ratio_extreme_control() -> None:
    # Control rate near 0 -> variance p(1-p) small -> ratio small
    r = sample_size_ratio(0.01, 0.10, 1000)
    assert r > 0
    # p1(1-p1) = 0.0099, p2(1-p2) = 0.09 -> ratio = 0.0099/0.09 ≈ 0.11
    assert r == pytest.approx(0.11, abs=0.01)


def test_ratio_equal_variance_returns_one() -> None:
    r = sample_size_ratio(0.3, 0.7, 100)
    p1_var = 0.3 * 0.7
    p2_var = 0.7 * 0.3
    assert p1_var == p2_var
    assert r == pytest.approx(1.0)


def test_ratio_extreme_treatment() -> None:
    r = sample_size_ratio(0.5, 0.99, 500)
    assert r > 0
    # p1(1-p1) = 0.25, p2(1-p2) = 0.0099 -> ratio = 0.25/0.0099 ≈ 25.25
    assert r == pytest.approx(25.25, abs=0.1)


def test_ratio_negative_n1_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sample_size_ratio(0.1, 0.2, 0)


def test_ratio_invalid_p1() -> None:
    with pytest.raises(ValueError, match="p1"):
        sample_size_ratio(0.0, 0.1, 100)


def test_ratio_invalid_p2() -> None:
    with pytest.raises(ValueError, match="p2"):
        sample_size_ratio(0.1, 1.0, 100)
