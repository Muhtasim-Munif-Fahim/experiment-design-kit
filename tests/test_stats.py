"""Tests for effect-size and sample-size calculators."""

from __future__ import annotations

import math

import pytest

from experiment_design_kit import stats as st


def test_cohen_h_known_values() -> None:
    assert st.cohen_h(0.1, 0.2) == pytest.approx(0.2838, abs=1e-3)
    assert st.cohen_h(0.5, 0.6) == pytest.approx(0.2014, abs=1e-3)


def test_cohen_h_is_symmetric_and_nonnegative() -> None:
    assert st.cohen_h(0.1, 0.2) == st.cohen_h(0.2, 0.1)
    assert st.cohen_h(0.1, 0.2) >= 0


def test_cohen_h_invalid_proportion() -> None:
    with pytest.raises(ValueError):
        st.cohen_h(-0.1, 0.2)
    with pytest.raises(ValueError):
        st.cohen_h(1.5, 0.2)


def test_cohens_d_pooled_sd() -> None:
    d = st.cohens_d(5.0, 5.5, sd1=1.0, sd2=1.0)
    assert d == pytest.approx(0.5, abs=1e-6)


def test_two_proportion_z_known_value() -> None:
    result = st.two_proportion_z(0.5, 0.6, 100, 100)
    assert result.z == pytest.approx(-1.4213, abs=1e-3)
    assert result.p_value == pytest.approx(0.1552, abs=1e-3)
    assert result.pooled_p == pytest.approx(0.55, abs=1e-6)


def test_two_proportion_z_errors() -> None:
    with pytest.raises(ValueError):
        st.two_proportion_z(1.5, 0.6, 100, 100)
    with pytest.raises(ValueError):
        st.two_proportion_z(0.5, 0.6, 0, 100)


def test_welch_t_known_value() -> None:
    result = st.welch_t(5.0, 1.0, 30, 5.5, 1.0, 30)
    assert result.statistic == pytest.approx(-1.9365, abs=1e-3)
    assert result.df == pytest.approx(58.0, abs=1e-6)
    assert result.p_value == pytest.approx(0.0577, abs=1e-3)


def test_welch_t_unequal_variance_has_lower_df_than_pooled() -> None:
    welch = st.welch_t(5.0, 1.0, 10, 5.5, 9.0, 10)
    pooled = st.pooled_t(5.0, 1.0, 10, 5.5, 9.0, 10)
    assert welch.df < pooled.df == 18


def test_two_proportion_sample_size_known_value() -> None:
    result = st.two_proportion_sample_size(0.10, 0.12, power=0.8)
    assert result.n_per_group_required == 3841
    assert result.n_total_required == 7682
    assert result.effect_size == pytest.approx(0.0637, abs=1e-3)


def test_two_proportion_sample_size_ratio() -> None:
    # ratio n2/n1 = 2 -> group2 larger, group1 smaller.
    result = st.two_proportion_sample_size(0.10, 0.12, ratio=2.0)
    n1 = result.metadata["n_group1"]
    n2 = result.metadata["n_group2"]
    assert n2 > n1
    assert result.metadata["n_group2_required"] == math.ceil(n2)


def test_two_proportion_sample_size_errors() -> None:
    with pytest.raises(ValueError):
        st.two_proportion_sample_size(0.1, 0.1, power=0.8)
    with pytest.raises(ValueError):
        st.two_proportion_sample_size(1.5, 0.12, power=0.8)


def test_two_sample_t_sample_size_normal_approx() -> None:
    result = st.two_sample_t_sample_size(0.5)
    assert result.n_per_group_required == 63  # normal-approx planning value
    assert result.effect_size == 0.5
    assert result.n_total_required == 126
