"""Tests for stratified randomization and covariate balance diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiment_design_kit import (
    balance_report,
    quantile_bins,
    standardized_mean_difference,
    stratified_randomization,
)


def test_quantile_bins_are_ordered_and_cover_every_unit() -> None:
    values = np.arange(100, dtype=float)
    bins = quantile_bins(values, 4)
    assert set(np.unique(bins).tolist()) == {0, 1, 2, 3}
    counts = np.bincount(bins)
    assert counts.max() - counts.min() <= 1
    for left, right in zip(range(3), range(1, 4)):
        assert values[bins == left].max() <= values[bins == right].min()


def test_quantile_bins_collapse_a_constant_covariate() -> None:
    bins = quantile_bins(np.ones(10), 4)
    assert np.all(bins == 0)


def test_quantile_bins_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        quantile_bins(np.arange(5), 0)
    with pytest.raises(ValueError, match="numeric"):
        quantile_bins(np.array(["a", "b", "c"]), 2)
    with pytest.raises(ValueError, match="finite"):
        quantile_bins(np.array([1.0, np.nan, 3.0]), 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        quantile_bins(np.ones((2, 2)), 2)


def test_equal_allocation_balances_every_stratum() -> None:
    region = np.array(["a", "a", "b", "b"] * 20)
    device = np.array(["ios", "android"] * 40)
    result = stratified_randomization(
        {"region": region, "device": device}, seed=2
    )
    assert result.assignment.shape == (80,)
    assert result.n_strata == 4
    assert set(result.assignment.tolist()) == {0, 1}
    assert result.arm_counts == (40, 40)
    for row in result.stratum_arm_counts:
        assert sum(row) == 20
        assert max(row) - min(row) <= 1


def test_allocation_ratio_is_exact_in_a_single_stratum() -> None:
    group = np.array(["x"] * 30)
    result = stratified_randomization({"group": group}, ratio=(1, 2), seed=0)
    assert result.arm_counts == (10, 20)
    other = stratified_randomization({"group": group}, ratio=(1, 2), seed=1)
    assert other.arm_counts == (10, 20)
    assert not np.array_equal(result.assignment, other.assignment)


def test_three_arms_differ_by_at_most_one_within_a_stratum() -> None:
    result = stratified_randomization(
        {"group": np.zeros(10, dtype=int)}, n_arms=3, seed=0
    )
    assert sorted(result.arm_counts) == [3, 3, 4]
    assert result.arm_labels == ("arm_0", "arm_1", "arm_2")


def test_assignment_is_reproducible_for_a_fixed_seed() -> None:
    region = np.array(["us"] * 25 + ["eu"] * 15 + ["apac"] * 10)
    score = np.linspace(-1.0, 1.0, region.size)
    covariates = {"region": region, "score": score}
    first = stratified_randomization(covariates, n_bins={"score": 4}, seed=11)
    second = stratified_randomization(covariates, n_bins={"score": 4}, seed=11)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    np.testing.assert_array_equal(first.strata, second.strata)
    third = stratified_randomization(covariates, n_bins={"score": 4}, seed=12)
    assert not np.array_equal(first.assignment, third.assignment)


def test_integer_n_bins_bins_only_continuous_covariates() -> None:
    region = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    score = np.linspace(0.0, 1.0, 8)
    result = stratified_randomization(
        {"region": region, "score": score}, n_bins=2, seed=0
    )
    np.testing.assert_array_equal(result.factors["region"], region)
    assert set(np.unique(result.factors["score"]).tolist()).issubset({0, 1})
    assert "score" in result.bin_edges
    assert len(result.bin_edges["score"]) >= 2


def test_skewed_factor_is_balanced_inside_each_level() -> None:
    region = np.array(["a"] * 90 + ["b"] * 30 + ["c"] * 15)
    result = stratified_randomization({"region": region}, seed=1)
    for row in result.stratum_arm_counts:
        assert max(row) - min(row) <= 1
    report = balance_report({"region": region}, result.assignment)
    row = report.rows[0]
    assert row.kind == "categorical"
    assert row.p_value is not None and row.p_value > 0.5
    assert report.max_abs_smd < 0.2
    bad = np.array([0] * 90 + [1] * 45)
    bad_report = balance_report({"region": region}, bad)
    assert bad_report.rows[0].chi2 > row.chi2


def test_quantile_stratification_balances_bins_and_shrinks_raw_smd() -> None:
    score = np.linspace(-3.0, 3.0, 400)
    result = stratified_randomization({"score": score}, n_bins={"score": 4}, seed=0)
    binned = balance_report(
        {"score": result.factors["score"]},
        result.assignment,
        categorical=["score"],
    )
    assert binned.rows[0].kind == "categorical"
    assert binned.max_abs_smd < 0.15
    raw = balance_report({"score": score}, result.assignment)
    assert raw.rows[0].kind == "continuous"
    assert raw.rows[0].means is not None
    assert abs(raw.rows[0].smd_by_arm[1]) < 0.2
    assert len(result.bin_edges["score"]) == 5
    assert result.bin_edges["score"][0] == pytest.approx(-3.0)
    assert result.bin_edges["score"][-1] == pytest.approx(3.0)


def test_standardized_mean_difference_matches_austin_formula() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0, 1.0, 3.0])
    assignment = np.array([0, 0, 0, 0, 1, 1])
    expected = 0.5 / math.sqrt(11.0 / 6.0)
    assert standardized_mean_difference(values, assignment) == pytest.approx(expected)
    flipped = standardized_mean_difference(values, assignment, arm=0, reference_arm=1)
    assert flipped == pytest.approx(-expected)


def test_binary_smd_and_chi_square_on_a_known_table() -> None:
    category = np.array(["a"] * 50 + ["b"] * 50)
    assignment = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
    report = balance_report({"category": category}, assignment)
    row = report.rows[0]
    assert row.chi2 == pytest.approx(36.0)
    assert row.dof == 1
    assert row.p_value is not None and row.p_value < 1e-8
    assert row.max_abs_smd == pytest.approx(1.5)
    assert abs(row.smd_by_arm[1]) == pytest.approx(1.5)
    assert row.levels is not None
    assert row.levels[0][0] == "a"
    assert row.levels[0][1] == (40, 10)


def test_complete_separation_has_infinite_smd() -> None:
    category = np.array(["a"] * 20 + ["b"] * 20)
    assignment = np.array([0] * 20 + [1] * 20)
    report = balance_report({"category": category}, assignment)
    assert report.rows[0].chi2 == pytest.approx(40.0)
    assert math.isinf(report.max_abs_smd)
    assert "inf" in str(report)


def test_integer_codes_default_to_continuous_unless_marked_categorical() -> None:
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assignment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    continuous = balance_report({"group": group}, assignment)
    assert continuous.rows[0].kind == "continuous"
    categorical = balance_report({"group": group}, assignment, categorical=["group"])
    assert categorical.rows[0].kind == "categorical"
    assert categorical.rows[0].chi2 == pytest.approx(0.0)


def test_custom_arm_labels() -> None:
    result = stratified_randomization(
        {"group": np.array(["a"] * 10)},
        arm_labels=("holdout", "ship"),
        seed=0,
    )
    assert result.arm_labels == ("holdout", "ship")


def test_readme_example_balances_a_skewed_region() -> None:
    rng = np.random.default_rng(0)
    region = rng.choice(["us", "eu", "apac"], size=240, p=[0.5, 0.3, 0.2])
    pre_metric = rng.normal(size=240)
    assigned = stratified_randomization(
        {"region": region, "pre_metric": pre_metric},
        n_bins={"pre_metric": 4},
        seed=0,
    )
    assert assigned.n_strata > 1
    assert sum(assigned.arm_counts) == 240
    report = balance_report(
        {"region": region, "pre_metric": pre_metric},
        assigned.assignment,
        categorical=["region"],
    )
    region_row = next(row for row in report.rows if row.covariate == "region")
    assert region_row.kind == "categorical"
    assert region_row.max_abs_smd < 0.2
    assert "chi2=" in str(report)
    for level_name in ("us", "eu", "apac"):
        mask = region == level_name
        counts = np.bincount(assigned.assignment[mask], minlength=2)
        assert abs(int(counts[0] - counts[1])) <= 4


def test_stratified_randomization_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="at least one covariate"):
        stratified_randomization({})
    with pytest.raises(ValueError, match="same length"):
        stratified_randomization({"a": np.array(["x", "y"]), "b": np.array(["x"])})
    with pytest.raises(ValueError, match="looks continuous"):
        stratified_randomization({"score": np.linspace(0.0, 1.0, 8)})
    with pytest.raises(ValueError, match="enough distinct"):
        stratified_randomization({"score": np.ones(8)}, n_bins={"score": 4})
    with pytest.raises(ValueError, match="n_arms"):
        stratified_randomization({"group": np.array(["a", "b"])}, n_arms=1)
    with pytest.raises(ValueError, match="ratio"):
        stratified_randomization({"group": np.array(["a"] * 6)}, ratio=(1, 0))
    with pytest.raises(ValueError, match="not a covariate"):
        stratified_randomization({"group": np.array([1, 2, 3, 4])}, n_bins={"age": 2})
    with pytest.raises(ValueError, match="no continuous"):
        stratified_randomization({"group": np.array(["a", "b", "a", "b"])}, n_bins=4)
    with pytest.raises(ValueError, match="seed"):
        stratified_randomization({"group": np.array(["a", "b"])}, seed=True)


def test_balance_report_rejects_invalid_input() -> None:
    values = {"group": np.array(["a", "a", "b", "b"])}
    with pytest.raises(ValueError, match="at least two arms"):
        balance_report(values, np.array([0, 0, 0, 0]))
    with pytest.raises(ValueError, match="same length"):
        balance_report(values, np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="unknown covariate"):
        balance_report(values, np.array([0, 1, 0, 1]), categorical=["missing"])
    with pytest.raises(ValueError, match="both categorical and continuous"):
        balance_report(
            values,
            np.array([0, 1, 0, 1]),
            categorical=["group"],
            continuous=["group"],
        )
    with pytest.raises(ValueError, match="reference_arm"):
        balance_report(values, np.array([0, 1, 0, 1]), reference_arm=3)
