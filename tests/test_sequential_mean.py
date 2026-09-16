"""Tests for sequential_mean_test."""
from __future__ import annotations

import pytest

from experiment_design_kit import sequential_mean_test, sequential_two_sample_mean_test


def test_significant_when_mean_far_from_null() -> None:
    report = sequential_mean_test(means=[5.0], sems=[0.1], null_mean=0.0)
    assert report.results[0].significant


def test_not_significant_when_mean_near_null() -> None:
    report = sequential_mean_test(means=[0.05], sems=[0.1], null_mean=0.0)
    assert not report.results[0].significant


def test_multiple_looks_accumulate() -> None:
    report = sequential_mean_test(
        means=[0.1, 0.2, 0.3],
        sems=[0.5, 0.3, 0.2],
        null_mean=0.0,
    )
    assert len(report.results) == 3
    assert report.results[2].cumulative_n == 3


def test_stop_on_significant() -> None:
    report = sequential_mean_test(
        means=[0.1, 5.0],
        sems=[0.5, 0.1],
        null_mean=0.0,
        stop_on_significant=True,
    )
    assert report.stopped
    assert report.stopped_at == 2


def test_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        sequential_mean_test(means=[0.0], sems=[0.1], alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        sequential_mean_test(means=[0.0], sems=[0.1], alpha=0.0)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        sequential_mean_test(means=[0.0], sems=[0.1, 0.2])


def test_zero_sem_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        sequential_mean_test(means=[0.0], sems=[0.0])


def test_negative_m_raises() -> None:
    with pytest.raises(ValueError, match="m must be positive"):
        sequential_mean_test(means=[0.0], sems=[0.1], m=-1.0)


def test_always_valid_pvalue_increases_with_n() -> None:
    report = sequential_mean_test(
        means=[0.5, 0.5, 0.5],
        sems=[0.1, 0.1, 0.1],
        null_mean=0.0,
    )
    avp = [r.always_valid_pvalue for r in report.results]
    assert avp[0] <= avp[1] <= avp[2]


def test_final_pvalue_reflects_last_look() -> None:
    report = sequential_mean_test(
        means=[0.1, 0.2, 0.3],
        sems=[0.5, 0.3, 0.2],
        null_mean=0.0,
    )
    assert report.final_pvalue == report.results[-1].always_valid_pvalue


def test_ns_overrides_look_index_as_sample_size() -> None:
    report = sequential_mean_test(
        means=[0.4, 0.4],
        sems=[0.1, 0.05],
        ns=[50, 200],
        null_mean=0.0,
    )
    assert report.results[0].cumulative_n == 50
    assert report.results[1].cumulative_n == 200


def test_two_sample_mean_null_not_significant() -> None:
    report = sequential_two_sample_mean_test(
        control_means=[0.0, 0.01, 0.0],
        control_sds=[1.0, 1.0, 1.0],
        control_ns=[80, 160, 240],
        treatment_means=[0.02, 0.0, 0.01],
        treatment_sds=[1.0, 1.0, 1.0],
        treatment_ns=[80, 160, 240],
        alpha=0.05,
    )
    assert len(report.results) == 3
    assert not report.is_significant


def test_two_sample_mean_large_effect_is_significant() -> None:
    report = sequential_two_sample_mean_test(
        control_means=[0.0, 0.0],
        control_sds=[1.0, 1.0],
        control_ns=[200, 400],
        treatment_means=[0.8, 0.8],
        treatment_sds=[1.0, 1.0],
        treatment_ns=[200, 400],
        alpha=0.05,
    )
    assert report.is_significant
    assert report.results[-1].always_valid_pvalue < 0.05


def test_two_sample_mean_obf_boundary() -> None:
    report = sequential_two_sample_mean_test(
        control_means=[0.0, 0.0],
        control_sds=[1.0, 1.0],
        control_ns=[50, 100],
        treatment_means=[0.05, 0.04],
        treatment_sds=[1.0, 1.0],
        treatment_ns=[50, 100],
        boundary="obrien-fleming",
        alpha=0.05,
    )
    assert report.boundary == "obrien-fleming"
    assert report.results[0].spending_increment < report.results[-1].spending_increment


def test_two_sample_mean_rejects_short_n() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        sequential_two_sample_mean_test(
            control_means=[0.0],
            control_sds=[1.0],
            control_ns=[1],
            treatment_means=[0.0],
            treatment_sds=[1.0],
            treatment_ns=[10],
        )
