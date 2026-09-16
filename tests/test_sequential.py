"""Tests for sequential monitoring."""
from __future__ import annotations

import math

import pytest

from experiment_design_kit.sequential import (
    SequentialReport,
    alpha_spent,
    always_valid_pvalue,
    mixture_likelihood_ratio,
    sequential_proportion_test,
    sequential_two_proportion_test,
    simulate_peeking_fpr,
    spending_increment,
)


def test_always_valid_pvalue_null() -> None:
    assert always_valid_pvalue(0.0, 100) == 1.0


def test_always_valid_pvalue_large_z() -> None:
    p = always_valid_pvalue(5.0, 100, m=0.01)
    assert p < 0.05
    assert 0.0 < p < 1.0


def test_always_valid_pvalue_zero_n() -> None:
    assert always_valid_pvalue(3.0, 0) == 1.0


def test_always_valid_pvalue_matches_msprt_closed_form() -> None:
    z, n, m = 3.0, 50.0, 0.02
    nm = m * n
    expected = math.sqrt(1.0 + nm) * math.exp(-(z ** 2) * nm / (2.0 * (1.0 + nm)))
    assert always_valid_pvalue(z, n, m) == pytest.approx(min(1.0, expected))
    assert mixture_likelihood_ratio(z, n, m) == pytest.approx(1.0 / expected)


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


def test_always_valid_pvalue_process_is_nonincreasing() -> None:
    report = sequential_proportion_test([80, 80, 80], [100, 100, 100], m=0.01)
    avp = [r.always_valid_pvalue for r in report.results]
    assert avp == sorted(avp, reverse=True)


def test_alpha_spent_equals_alpha_at_the_end() -> None:
    assert alpha_spent(1.0, alpha=0.05, method="pocock") == pytest.approx(0.05)
    assert alpha_spent(1.0, alpha=0.05, method="obrien-fleming") == pytest.approx(0.05)
    assert alpha_spent(0.0, alpha=0.05, method="pocock") == 0.0


def test_obrien_fleming_spends_less_than_pocock_early() -> None:
    obf = alpha_spent(0.2, alpha=0.05, method="obrien-fleming")
    pocock = alpha_spent(0.2, alpha=0.05, method="pocock")
    assert 0.0 < obf < pocock < 0.05


def test_spending_increments_sum_to_alpha() -> None:
    ts = [0.25, 0.5, 0.75, 1.0]
    prev = 0.0
    total = 0.0
    for t in ts:
        total += spending_increment(prev, t, alpha=0.05, method="pocock")
        prev = t
    assert total == pytest.approx(0.05)


def test_two_proportion_null_not_significant() -> None:
    report = sequential_two_proportion_test(
        control_successes=[50, 50, 50],
        control_totals=[100, 100, 100],
        treatment_successes=[50, 50, 50],
        treatment_totals=[100, 100, 100],
        alpha=0.05,
    )
    assert len(report.results) == 3
    assert not report.is_significant
    assert report.results[-1].cumulative_n == 300


def test_two_proportion_large_effect_is_significant() -> None:
    report = sequential_two_proportion_test(
        control_successes=[10, 10, 10, 10],
        control_totals=[100, 100, 100, 100],
        treatment_successes=[40, 40, 40, 40],
        treatment_totals=[100, 100, 100, 100],
        alpha=0.05,
        m=0.01,
    )
    assert report.is_significant
    assert report.final_pvalue < 0.05


def test_two_proportion_pocock_boundary_uses_spending() -> None:
    report = sequential_two_proportion_test(
        control_successes=[10, 10],
        control_totals=[100, 100],
        treatment_successes=[12, 11],
        treatment_totals=[100, 100],
        boundary="pocock",
        alpha=0.05,
    )
    assert report.boundary == "pocock"
    assert report.results[0].alpha_spent < 0.05
    assert report.results[-1].alpha_spent == pytest.approx(0.05)


def test_two_proportion_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        sequential_two_proportion_test([10], [100], [10, 10], [100, 100])


def test_peeking_inflates_naive_fpr_always_valid_controls_proportion() -> None:
    result = simulate_peeking_fpr(
        n_looks=8,
        n_per_look=250,
        n_trials=400,
        alpha=0.05,
        metric="proportion",
        p_control=0.10,
        p_treatment=0.10,
        seed=7,
    )
    assert result.naive_fpr > result.alpha + 0.05
    assert result.always_valid_fpr <= result.alpha + 0.04
    assert result.pocock_fpr <= result.alpha + 0.04
    assert result.obrien_fleming_fpr <= result.alpha + 0.04
    assert result.always_valid_fpr < result.naive_fpr
    assert result.obrien_fleming_fpr <= result.pocock_fpr + 0.02


def test_peeking_inflates_naive_fpr_always_valid_controls_mean() -> None:
    result = simulate_peeking_fpr(
        n_looks=8,
        n_per_look=120,
        n_trials=400,
        alpha=0.05,
        metric="mean",
        mean_control=0.0,
        mean_treatment=0.0,
        sd=1.0,
        seed=11,
    )
    assert result.naive_fpr > result.alpha + 0.05
    assert result.always_valid_fpr <= result.alpha + 0.04
    assert result.pocock_fpr <= result.alpha + 0.04
    assert result.obrien_fleming_fpr <= result.alpha + 0.04


def test_simulate_peeking_fpr_validates_args() -> None:
    with pytest.raises(ValueError, match="n_looks"):
        simulate_peeking_fpr(n_looks=0)
    with pytest.raises(ValueError, match="metric"):
        simulate_peeking_fpr(metric="logit")
