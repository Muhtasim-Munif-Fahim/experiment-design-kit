"""Tests for sequential monitoring."""
from __future__ import annotations

import pytest

from experiment_design_kit.sequential import (
    SequentialReport,
    always_valid_pvalue,
    sequential_proportion_test,
)


def test_always_valid_pvalue_null() -> None:
    assert always_valid_pvalue(0.0, 100) == 1.0


def test_always_valid_pvalue_large_z() -> None:
    p = always_valid_pvalue(5.0, 100, m=0.01)
    assert p < 0.05
    assert 0.0 < p < 1.0


def test_always_valid_pvalue_zero_n() -> None:
    assert always_valid_pvalue(3.0, 0) == 1.0


def test_sequential_test_no_effect() -> None:
    # Per-look: 50% conversion every look -> null effect
    successes = [50, 100, 150]
    totals = [100, 200, 300]
    report = sequential_proportion_test(successes, totals, alpha=0.05)
    assert len(report.results) == 3
    assert not report.is_significant


def test_sequential_test_with_effect() -> None:
    # Per-look: strong cumulative effect (10% -> 20% -> 30% -> 50%)
    successes = [10, 90, 80, 100]
    totals = [100, 100, 100, 100]
    report = sequential_proportion_test(successes, totals, alpha=0.05)
    assert len(report.results) == 4
    assert report.results[-1].cumulative_n == 400


def test_sequential_test_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        sequential_proportion_test([10, 20], [100], alpha=0.05)


def test_sequential_test_early_stop() -> None:
    # Very strong effect -> should stop at look 1
    successes = [100, 100, 100, 100]
    totals = [100, 100, 100, 100]
    report = sequential_proportion_test(
        successes, totals, alpha=0.05, m=0.001, stop_on_significant=True
    )
    assert report.stopped
    assert report.stopped_at == 1
    assert len(report.results) == 1


def test_sequential_test_no_early_stop() -> None:
    successes = [100, 100, 100, 100]
    totals = [100, 100, 100, 100]
    report = sequential_proportion_test(
        successes, totals, alpha=0.05, m=0.001, stop_on_significant=False
    )
    assert len(report.results) == 4
    assert not report.stopped


def test_sequential_test_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        sequential_proportion_test([10], [100], alpha=0.0)


def test_sequential_test_invalid_m() -> None:
    with pytest.raises(ValueError, match="m must be positive"):
        sequential_proportion_test([10], [100], alpha=0.05, m=0)


def test_sequential_report_structure() -> None:
    successes = [10, 40]
    totals = [100, 100]
    report = sequential_proportion_test(successes, totals)
    assert isinstance(report, SequentialReport)
    assert len(report.results) == 2
    r = report.results[0]
    assert r.look == 1
    assert r.cumulative_n == 100
    assert 0.0 <= r.always_valid_pvalue <= 1.0


def test_sequential_test_zero_totals_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        sequential_proportion_test([10], [0], alpha=0.05)
