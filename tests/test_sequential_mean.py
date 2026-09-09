"""Tests for sequential_mean_test."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import SequentialReport, sequential_mean_test


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
