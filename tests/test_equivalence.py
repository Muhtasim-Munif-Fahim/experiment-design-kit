"""Tests for non-inferiority and TOST equivalence tests and planning."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sp

from experiment_design_kit import (
    equivalence_power,
    equivalence_sample_size,
    noninferiority_means,
    noninferiority_power,
    noninferiority_proportions,
    noninferiority_sample_size,
    tost_means,
    tost_proportions,
)
from experiment_design_kit.cli import main


def test_tost_means_matches_ci_rule_and_welch():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 400), rng.normal(10.1, 2.5, 350)
    res = tost_means(a, b, margin=0.6)
    welch = sp.ttest_ind(b, a, equal_var=False)
    assert res.difference == pytest.approx(b.mean() - a.mean())
    assert res.difference / res.se == pytest.approx(welch.statistic)
    assert res.p_value == max(res.p_lower, res.p_upper)
    # equivalence <=> 90% CI inside the margins
    assert res.equivalent == (res.ci_low > -0.6 and res.ci_high < 0.6)
    assert res.equivalent


def test_tost_means_not_equivalent_when_underpowered_or_shifted():
    rng = np.random.default_rng(1)
    small = tost_means(rng.normal(0, 1, 15), rng.normal(0, 1, 15), margin=0.2)
    assert not small.equivalent  # absence of significance is not equivalence
    shifted = tost_means(rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000), margin=0.2)
    assert not shifted.equivalent and shifted.p_upper > 0.5


def test_asymmetric_margins():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 1000), rng.normal(0.15, 1, 1000)
    sym = tost_means(a, b, margin=0.2)
    asym = tost_means(a, b, margin=-0.1, upper_margin=0.4)
    assert (asym.lower_margin, asym.upper_margin) == (-0.1, 0.4)
    assert asym.p_upper < sym.p_upper
    with pytest.raises(ValueError, match="bracket zero"):
        tost_means(a, b, margin=0.1, upper_margin=0.4)


def test_tost_proportions_by_hand():
    res = tost_proportions(3000, 10000, 3050, 10000, margin=0.02)
    pc, pt = 0.30, 0.305
    se = math.sqrt(pc * (1 - pc) / 1e4 + pt * (1 - pt) / 1e4)
    assert res.se == pytest.approx(se)
    assert res.p_lower == pytest.approx(sp.norm.sf((0.005 + 0.02) / se))
    assert res.p_upper == pytest.approx(sp.norm.cdf((0.005 - 0.02) / se))
    assert res.equivalent and math.isinf(res.df)


def test_noninferiority_means_directions():
    rng = np.random.default_rng(3)
    control, treatment = rng.normal(100, 10, 500), rng.normal(99.5, 10, 500)
    res = noninferiority_means(control, treatment, margin=3.0)
    assert res.noninferior and res.confidence_bound > -3.0
    assert res.statistic == pytest.approx((res.difference + 3.0) / res.se)
    # latency-style metric: lower is better, treatment 4 units slower
    lat = noninferiority_means(control, control + 4 + rng.normal(0, 1, 500), margin=3.0,
                               higher_is_better=False)
    assert not lat.noninferior and lat.confidence_bound > 3.0


def test_noninferiority_proportions():
    ok = noninferiority_proportions(2000, 10000, 1980, 10000, margin=0.02)
    assert ok.noninferior
    bad = noninferiority_proportions(2000, 10000, 1800, 10000, margin=0.01)
    assert not bad.noninferior


def test_noninferiority_sample_size_closed_form():
    plan = noninferiority_sample_size(0.5, sd=2.0)
    z = sp.norm.ppf(0.975) + sp.norm.ppf(0.8)
    assert plan.metadata["n_control_exact"] == pytest.approx(z * z * 8 / 0.25)
    assert plan.n_control == 252 and plan.n_total == 504
    assert plan.achieved_power >= 0.8
    better = noninferiority_sample_size(0.5, sd=2.0, true_diff=0.25)
    assert better.n_control < plan.n_control
    unbalanced = noninferiority_sample_size(0.5, sd=2.0, ratio=2.0)
    assert unbalanced.n_treatment == 2 * unbalanced.n_control
    with pytest.raises(ValueError, match="exceed -margin"):
        noninferiority_sample_size(0.5, sd=2.0, true_diff=-0.6)


def test_equivalence_sample_size_textbook_values():
    # delta = 0: n = (z_{1-a} + z_{1-b/2})^2 * 2 sd^2 / m^2
    plan = equivalence_sample_size(0.5, sd=2.0)
    z = sp.norm.ppf(0.95) + sp.norm.ppf(0.9)
    assert plan.n_control == math.ceil(z * z * 8 / 0.25)
    assert plan.achieved_power >= 0.8
    assert equivalence_power(0.5, plan.n_control - 1, sd=2.0) < 0.8
    prop = equivalence_sample_size(0.05, p_control=0.3)
    assert prop.n_control == math.ceil(z * z * 0.42 / 0.0025)
    off_center = equivalence_sample_size(0.5, sd=2.0, true_diff=0.2)
    assert off_center.n_control > plan.n_control


def test_planned_power_holds_in_simulation():
    plan = equivalence_sample_size(0.3, sd=1.0, true_diff=0.05, power=0.8)
    rng = np.random.default_rng(5)
    hits = 0
    reps = 400
    for _ in range(reps):
        a = rng.normal(0, 1, plan.n_control)
        b = rng.normal(0.05, 1, plan.n_treatment)
        hits += tost_means(a, b, margin=0.3).equivalent
    assert abs(hits / reps - 0.8) < 0.07


def test_noninferiority_power_monotone():
    p1 = noninferiority_power(0.02, 1000, p_control=0.2)
    p2 = noninferiority_power(0.02, 4000, p_control=0.2)
    assert 0 < p1 < p2 < 1


@pytest.mark.parametrize("call, match", [
    (lambda: tost_means([1, 2, 3], [1, 2, 3], margin=0.0), "margin"),
    (lambda: tost_means([1, 2, 3], [2, 3, 4], margin=1, alpha=0.6), "alpha"),
    (lambda: tost_means([1], [2, 3], margin=1), "at least 2"),
    (lambda: tost_proportions(5, 4, 1, 4, margin=0.1), "between 0"),
    (lambda: equivalence_sample_size(0.1, sd=1, true_diff=0.2), "inside"),
    (lambda: equivalence_sample_size(0.1, sd=1, p_control=0.2), "not both"),
    (lambda: noninferiority_sample_size(0.1, sd=-1), "sd"),
    (lambda: noninferiority_sample_size(0.1, sd=1, power=1.0), "power"),
])
def test_validation(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_cli(capsys):
    assert main(["equivalence", "--margin", "0.5", "--sd", "2"]) == 0
    assert "n control: 252" in capsys.readouterr().out
    assert main(["equivalence", "--design", "equivalence", "--mode", "test", "--margin", "0.02",
                 "--counts", "3000,10000,3050,10000"]) == 0
    out = capsys.readouterr().out
    assert "EQUIVALENT" in out and "90% CI" in out
    assert main(["equivalence", "--mode", "power", "--margin", "0.5", "--sd", "2", "--n", "252"]) == 0
    assert "power: 0.80" in capsys.readouterr().out
    assert main(["equivalence", "--mode", "power", "--margin", "0.5"]) == 2
    assert "--n is required" in capsys.readouterr().err
