"""Tests for sample ratio mismatch (SRM) checks."""
from __future__ import annotations

import io
import math
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest
from scipy import stats as sp

from experiment_design_kit import (
    SRMResult,
    blocked_randomization,
    sample_ratio_mismatch,
    sequential_srm_test,
    srm_from_assignment,
    stratified_randomization,
)
from experiment_design_kit.cli import main


def _cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


# --- fixed-horizon -------------------------------------------------------


def test_matches_scipy_chisquare_for_equal_split() -> None:
    counts = [50421, 49579]
    result = sample_ratio_mismatch(counts)
    reference = sp.chisquare(counts)
    assert isinstance(result, SRMResult)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))
    assert result.dof == 1
    assert result.n == 100000
    assert result.expected == pytest.approx((50000.0, 50000.0))
    # p ~ 0.0078: suspicious, but not past the conventional 0.001 alarm.
    assert not result.mismatch
    assert sample_ratio_mismatch(counts, alpha=0.01).mismatch


def test_hand_computed_statistic_and_residuals() -> None:
    # ratio (3, 2) -> expected 60 and 40 of 100 units.
    result = sample_ratio_mismatch([70, 30], ratio=(3, 2))
    assert result.expected_share == pytest.approx((0.6, 0.4))
    assert result.expected == pytest.approx((60.0, 40.0))
    assert result.statistic == pytest.approx(100 / 60 + 100 / 40)
    assert result.residuals == pytest.approx((10 / math.sqrt(60), -10 / math.sqrt(40)))
    assert result.observed_share == pytest.approx((0.7, 0.3))
    assert result.worst_arm == "treatment"


def test_unequal_ratio_on_plan_is_not_flagged() -> None:
    # A 10% holdout that is exactly on plan.
    result = sample_ratio_mismatch([9000, 1000], ratio=(90, 10))
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.mismatch
    # The same counts tested against 50/50 are a gross mismatch.
    assert sample_ratio_mismatch([9000, 1000]).mismatch


def test_multi_arm_and_labels() -> None:
    result = sample_ratio_mismatch(
        [1000, 1000, 820], arm_labels=["A", "B", "C"]
    )
    assert result.dof == 2
    assert result.arm_labels == ("A", "B", "C")
    assert result.worst_arm == "C"
    assert result.mismatch
    default = sample_ratio_mismatch([10, 10, 10])
    assert default.arm_labels == ("arm_0", "arm_1", "arm_2")


def test_g_test_agrees_with_scipy_power_divergence() -> None:
    counts = [5200, 4800, 5050]
    ours = sample_ratio_mismatch(counts, method="g-test")
    ref = sp.power_divergence(counts, lambda_="log-likelihood")
    assert ours.statistic == pytest.approx(float(ref.statistic))
    assert ours.p_value == pytest.approx(float(ref.pvalue))
    assert ours.method == "g-test"


def test_g_test_handles_an_empty_arm() -> None:
    result = sample_ratio_mismatch([100, 0], method="g-test")
    assert math.isfinite(result.statistic)
    assert result.mismatch


def test_null_false_alarm_rate_is_near_alpha() -> None:
    rng = np.random.default_rng(0)
    alpha = 0.05
    flags = [
        sample_ratio_mismatch(rng.multinomial(2000, [0.5, 0.5]), alpha=alpha).mismatch
        for _ in range(2000)
    ]
    assert np.mean(flags) == pytest.approx(alpha, abs=0.015)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"counts": [10]}, "at least two arms"),
        ({"counts": [[1, 2], [3, 4]]}, "1-d"),
        ({"counts": [10, -1]}, "nonnegative"),
        ({"counts": [10, 2.5]}, "whole numbers"),
        ({"counts": [0, 0]}, "at least one unit"),
        ({"counts": [10, 10], "ratio": (1, 0)}, "positive"),
        ({"counts": [10, 10], "ratio": (1, 1, 1)}, "one positive weight per arm"),
        ({"counts": [10, 10], "alpha": 1.0}, "alpha"),
        ({"counts": [10, 10], "method": "fisher"}, "method"),
        ({"counts": [10, 10], "arm_labels": ["a"]}, "one label per arm"),
    ],
)
def test_input_validation(kwargs: dict, match: str) -> None:
    counts = kwargs.pop("counts")
    with pytest.raises(ValueError, match=match):
        sample_ratio_mismatch(counts, **kwargs)


# --- from assignments ----------------------------------------------------


def test_from_arm_id_array_counts_missing_arms() -> None:
    ids = np.array([0] * 30 + [2] * 30)
    result = srm_from_assignment(ids, n_arms=3)
    assert result.observed == (30, 0, 30)
    assert result.mismatch
    inferred = srm_from_assignment(np.zeros(5, dtype=int))
    assert inferred.observed == (5, 0)


def test_from_kit_assignment_objects() -> None:
    blocked = blocked_randomization(100, block_size=4, seed=0)
    result = srm_from_assignment(blocked)
    assert sum(result.observed) == 100
    assert result.arm_labels == blocked.arm_labels
    assert not result.mismatch

    rng = np.random.default_rng(1)
    strat = stratified_randomization(
        {"region": rng.choice(["n", "s", "e"], size=300)}, ratio=(1, 2), seed=1
    )
    on_plan = srm_from_assignment(strat, ratio=(1, 2))
    assert not on_plan.mismatch
    assert srm_from_assignment(strat).p_value < on_plan.p_value


def test_from_assignment_validation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        srm_from_assignment(np.array([0, 1, 2]), n_arms=2)
    with pytest.raises(ValueError, match="integers"):
        srm_from_assignment(np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="nonnegative"):
        srm_from_assignment(np.array([-1, 0]))
    with pytest.raises(ValueError, match="non-empty"):
        srm_from_assignment(np.array([], dtype=int))
    with pytest.raises(ValueError, match="n_arms does not match"):
        srm_from_assignment(blocked_randomization(8, block_size=4, seed=0), n_arms=3)


# --- sequential ----------------------------------------------------------


def test_bayes_factor_matches_closed_form() -> None:
    # Flat Dirichlet(1, 1) vs p = 1/2 with x = (3, 1):
    # BF = B(4, 2) / B(1, 1) / 0.5**4 = (3! 1! / 5!) * 16 = 0.8
    result = sequential_srm_test([[3, 1]], cumulative=True)
    assert result.prior == pytest.approx((1.0, 1.0))
    assert math.exp(result.looks[0].log_bayes_factor) == pytest.approx(0.8)
    assert result.final_p_value == pytest.approx(1.0)


def test_incremental_and_cumulative_inputs_agree() -> None:
    increments = [[510, 490], [495, 505], [530, 470]]
    cumulative = np.cumsum(increments, axis=0).tolist()
    a = sequential_srm_test(increments)
    b = sequential_srm_test(cumulative, cumulative=True)
    assert [look.counts for look in a.looks] == [tuple(row) for row in cumulative]
    assert [look.p_value for look in a.looks] == pytest.approx([look.p_value for look in b.looks])


def test_p_values_are_a_running_minimum() -> None:
    rng = np.random.default_rng(3)
    looks = rng.multinomial(200, [0.5, 0.5], size=25)
    result = sequential_srm_test(looks)
    p = np.array([look.p_value for look in result.looks])
    assert np.all(np.diff(p) <= 1e-15)
    assert np.all((p > 0) & (p <= 1))


def test_anytime_valid_under_continuous_peeking() -> None:
    rng = np.random.default_rng(0)
    alpha = 0.05
    flagged = 0
    trials = 400
    for _ in range(trials):
        looks = rng.multinomial(25, [0.5, 0.5], size=80)
        flagged += sequential_srm_test(looks, alpha=alpha).mismatch
    # Ville's inequality bounds the false-alarm rate by alpha at any number of peeks.
    assert flagged / trials <= alpha + 0.02


def test_detects_a_real_mismatch_and_reports_first_look() -> None:
    rng = np.random.default_rng(7)
    looks = rng.multinomial(1000, [0.53, 0.47], size=30)
    result = sequential_srm_test(looks, alpha=0.001)
    assert result.mismatch
    first = result.first_mismatch_look
    assert first is not None and first > 1
    assert all(look.mismatch for look in result.looks[first - 1 :])
    assert not any(look.mismatch for look in result.looks[: first - 1])


def test_unequal_ratio_and_three_arms() -> None:
    rng = np.random.default_rng(11)
    looks = rng.multinomial(300, [0.2, 0.4, 0.4], size=20)
    assert not sequential_srm_test(looks, ratio=(1, 2, 2), alpha=0.01).mismatch
    assert sequential_srm_test(looks, alpha=0.01).mismatch


def test_custom_prior_and_concentration() -> None:
    looks = [[520, 480]] * 5
    base = sequential_srm_test(looks, concentration=2.0)
    same = sequential_srm_test(looks, prior=(1.0, 1.0))
    assert base.prior == pytest.approx((1.0, 1.0))
    assert [lk.log_bayes_factor for lk in base.looks] == pytest.approx(
        [lk.log_bayes_factor for lk in same.looks]
    )
    tight = sequential_srm_test(looks, concentration=500.0)
    assert tight.prior == pytest.approx((250.0, 250.0))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"counts": []}, "at least one look"),
        ({"counts": [[1, 2], [1, 2, 3]]}, "same number of arms"),
        ({"counts": [[5, 5], [4, 6]], "cumulative": True}, "non-decreasing"),
        ({"counts": [[1, 2]], "prior": (1.0,)}, "one positive value per arm"),
        ({"counts": [[1, 2]], "prior": (1.0, 0.0)}, "positive"),
        ({"counts": [[1, 2]], "prior": (1.0, 1.0), "concentration": 2.0}, "either prior"),
        ({"counts": [[1, 2]], "concentration": 0.0}, "concentration"),
        ({"counts": [[1, 2]], "alpha": 0.0}, "alpha"),
    ],
)
def test_sequential_validation(kwargs: dict, match: str) -> None:
    counts = kwargs.pop("counts")
    with pytest.raises(ValueError, match=match):
        sequential_srm_test(counts, **kwargs)


# --- CLI ----------------------------------------------------------------


def test_cli_fixed_horizon() -> None:
    rc, out, _ = _cli(["srm", "--counts", "9000,1000"])
    assert rc == 0
    assert "SRM DETECTED" in out
    rc, out, _ = _cli(["srm", "--counts", "9000,1000", "--ratio", "90,10", "--method", "g-test"])
    assert rc == 0
    assert "verdict: no SRM" in out
    assert "g-test" in out


def test_cli_sequential() -> None:
    looks = ";".join(["560,440"] * 6)
    rc, out, _ = _cli(["srm", "--looks", looks])
    assert rc == 0
    assert "always-valid p" in out
    assert "SRM DETECTED at look" in out


def test_cli_reports_bad_input() -> None:
    rc, _, err = _cli(["srm", "--counts", "10,2.5"])
    assert rc == 2
    assert "whole numbers" in err
