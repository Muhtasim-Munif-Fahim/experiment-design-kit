"""Tests for multiple-testing corrections and A/B/n sample-size planning."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest

from experiment_design_kit import (
    MultipleTestingResult,
    adjust_pvalues,
    corrected_alpha,
    family_wise_error_rate,
    multi_arm_sample_size,
    two_proportion_sample_size,
)
from experiment_design_kit.cli import main

P = [0.01, 0.04, 0.03, 0.005, 0.20, 0.049]


def test_hand_computed_holm_and_bh():
    p = [0.01, 0.04, 0.03, 0.005]
    holm = adjust_pvalues(p, "holm")
    # sorted 0.005,0.01,0.03,0.04 -> *4,*3,*2,*1 = 0.02,0.03,0.06,0.04 -> cummax
    assert holm.adjusted == pytest.approx((0.03, 0.06, 0.06, 0.02))
    assert holm.rejected == (True, False, False, True)
    bh = adjust_pvalues(p, "bh")
    # BH: p*m/rank = 0.005*4=0.02, 0.01*2=0.02, 0.03*4/3=0.04, 0.04 -> cummin from top
    assert bh.adjusted == pytest.approx((0.02, 0.04, 0.04, 0.02))
    assert bh.n_rejected == 4 and bh.controls == "FDR" and holm.controls == "FWER"


@pytest.mark.parametrize(
    "ours,theirs",
    [
        ("bonferroni", "bonferroni"),
        ("sidak", "sidak"),
        ("holm", "holm"),
        ("holm-sidak", "holm-sidak"),
        ("hochberg", "simes-hochberg"),
        ("bh", "fdr_bh"),
        ("by", "fdr_by"),
    ],
)
def test_matches_statsmodels(ours, theirs):
    smm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    for p in (P, rng.uniform(0, 0.2, 25), [0.5, 0.5, 0.01, 0.01, 1.0]):
        res = adjust_pvalues(p, ours, alpha=0.05)
        reject, adj, _, _ = smm.multipletests(p, alpha=0.05, method=theirs)
        np.testing.assert_allclose(res.adjusted, adj, rtol=1e-10, atol=1e-12)
        assert list(res.rejected) == list(reject)


def test_ordering_and_power_relations():
    bonf = np.array(adjust_pvalues(P, "bonferroni").adjusted)
    holm = np.array(adjust_pvalues(P, "holm").adjusted)
    hoch = np.array(adjust_pvalues(P, "hochberg").adjusted)
    bh = np.array(adjust_pvalues(P, "bh").adjusted)
    by = np.array(adjust_pvalues(P, "by").adjusted)
    assert np.all(holm <= bonf + 1e-15)
    assert np.all(hoch <= holm + 1e-15)
    assert np.all(bh <= hoch + 1e-15)
    assert np.all(by >= bh - 1e-15)
    assert np.all(np.array(P) <= bh + 1e-15)


def test_monotone_and_order_preserved():
    res = adjust_pvalues(P, "holm-sidak")
    order = np.argsort(P)
    adj = np.array(res.adjusted)[order]
    assert np.all(np.diff(adj) >= -1e-15)
    assert res.p_values == tuple(P)
    assert isinstance(res, MultipleTestingResult) and res.n_tests == 6


def test_aliases_and_single_test():
    assert adjust_pvalues(P, "fdr_bh").adjusted == adjust_pvalues(P, "BH").adjusted
    for method in ("bonferroni", "holm", "hochberg", "bh", "by", "sidak"):
        assert adjust_pvalues([0.03], method).adjusted == pytest.approx((0.03,))


def test_global_null_fwer_simulation():
    rng = np.random.default_rng(1)
    any_reject = {"holm": 0, "bh": 0, "none": 0}
    sims, m = 2000, 10
    for _ in range(sims):
        p = rng.uniform(size=m)
        any_reject["holm"] += adjust_pvalues(p, "holm").n_rejected > 0
        any_reject["bh"] += adjust_pvalues(p, "bh").n_rejected > 0
        any_reject["none"] += bool(np.any(p <= 0.05))
    assert any_reject["holm"] / sims < 0.065
    assert any_reject["bh"] / sims < 0.065  # FDR = FWER under the global null
    assert any_reject["none"] / sims == pytest.approx(family_wise_error_rate(0.05, m), abs=0.03)


def test_corrected_alpha_and_fwer():
    assert corrected_alpha(0.05, 4) == pytest.approx(0.0125)
    sid = corrected_alpha(0.05, 4, "sidak")
    assert family_wise_error_rate(sid, 4) == pytest.approx(0.05)
    assert sid > 0.0125
    assert corrected_alpha(0.05, 3, "none") == 0.05
    assert family_wise_error_rate(0.05, 10) == pytest.approx(1 - 0.95**10)


def test_multi_arm_sample_size():
    plan = multi_arm_sample_size(0.10, [0.12, 0.13], alpha=0.05, power=0.8)
    single = two_proportion_sample_size(0.10, 0.12, alpha=0.025, power=0.8)
    assert plan.n_per_arm_by_comparison[0] == single.n_per_group_required
    assert plan.n_per_arm_required == max(plan.n_per_arm_by_comparison)
    assert plan.n_per_arm_required == plan.n_per_arm_by_comparison[0]  # smaller lift is harder
    assert plan.n_arms == 3 and plan.n_total_required == 3 * plan.n_per_arm_required
    uncorrected = multi_arm_sample_size(0.10, [0.12, 0.13], correction="none")
    assert uncorrected.n_per_arm_required < plan.n_per_arm_required
    sidak = multi_arm_sample_size(0.10, [0.12, 0.13], correction="sidak")
    assert sidak.n_per_arm_required <= plan.n_per_arm_required


@pytest.mark.parametrize(
    "call",
    [
        lambda: adjust_pvalues([], "holm"),
        lambda: adjust_pvalues([0.1, 1.2], "holm"),
        lambda: adjust_pvalues([0.1, float("nan")], "holm"),
        lambda: adjust_pvalues([0.1], "tukey"),
        lambda: adjust_pvalues([0.1], "holm", alpha=0.0),
        lambda: corrected_alpha(0.05, 0),
        lambda: corrected_alpha(0.05, 2, "dunnett"),
        lambda: multi_arm_sample_size(0.1, []),
        lambda: family_wise_error_rate(0.05, 0),
    ],
)
def test_errors(call):
    with pytest.raises(ValueError):
        call()


def test_cli_adjust_and_plan():
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["multiple-testing", "--pvalues", "0.01,0.04,0.03,0.005", "--method", "holm"]) == 0
    text = out.getvalue()
    assert "rejected: 2 of 4" in text and "FWER" in text
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["multiple-testing", "--treatments", "0.12,0.13", "--control", "0.1"]) == 0
    assert "total (3 arms)" in out.getvalue()
    err = io.StringIO()
    with redirect_stderr(err):
        assert main(["multiple-testing", "--pvalues", "0.1,2", "--method", "bh"]) == 2
    assert "multiple-testing:" in err.getvalue()
