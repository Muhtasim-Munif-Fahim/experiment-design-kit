"""Tests for whole-cluster randomization and cluster-level balance."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import (
    ClusterAssignment,
    balance_report,
    cluster_balance_report,
    cluster_randomization,
)


def _assert_clusters_are_intact(result: ClusterAssignment) -> None:
    assert result.assignment.shape == result.cluster_ids.shape
    assert result.cluster_arms.shape == (result.n_clusters,)
    assert sum(result.cluster_counts) == result.n_clusters
    assert sum(result.arm_counts) == result.n
    assert len(result.arm_counts) == len(result.arm_labels)
    assert len(result.cluster_counts) == len(result.arm_labels)
    for label, arm in zip(result.cluster_labels, result.cluster_arms):
        mask = result.cluster_ids == label
        assert int(mask.sum()) >= 1
        assert np.all(result.assignment[mask] == arm)
        assert int(result.assignment[mask][0]) == int(arm)


def test_units_in_a_cluster_share_one_arm() -> None:
    ids = np.array([10, 10, 10, 7, 7, 3, 3, 3, 3])
    result = cluster_randomization(ids, seed=0)
    assert result.n == 9
    assert result.n_clusters == 3
    assert result.cluster_labels == (3, 7, 10)
    assert result.arm_labels == ("control", "treatment")
    assert result.seed == 0
    _assert_clusters_are_intact(result)
    assert sorted(result.cluster_counts) == [1, 2]


def test_string_clusters_and_list_input_follow_sorted_ids() -> None:
    result = cluster_randomization(["b", "a", "b", "c", "a"], seed=1)
    assert result.cluster_labels == ("a", "b", "c")
    assert result.n == 5
    _assert_clusters_are_intact(result)


def test_row_order_does_not_change_the_cluster_arm() -> None:
    ids = np.array([4, 1, 4, 2, 1, 2, 4, 3])
    base = cluster_randomization(ids, seed=7)
    perm = np.array([3, 0, 7, 1, 5, 2, 6, 4])
    moved = cluster_randomization(ids[perm], seed=7)
    np.testing.assert_array_equal(moved.assignment, base.assignment[perm])
    assert moved.cluster_labels == base.cluster_labels
    np.testing.assert_array_equal(moved.cluster_arms, base.cluster_arms)


def test_stored_ids_are_not_aliased_to_the_input() -> None:
    ids = np.array([1, 2, 1, 2])
    result = cluster_randomization(ids, seed=0)
    ids[0] = 99
    assert result.cluster_ids[0] == 1
    _assert_clusters_are_intact(result)


def test_ratio_allocates_clusters_not_units() -> None:
    sizes = [1] * 8 + [10]
    ids = np.repeat(np.arange(9), sizes)
    result = cluster_randomization(ids, ratio=(1, 2), seed=0)
    assert result.cluster_counts == (3, 6)
    assert sum(result.arm_counts) == 18
    _assert_clusters_are_intact(result)
    big = result.assignment[ids == 8]
    assert np.unique(big).size == 1
    scaled = cluster_randomization(ids, ratio=(2.0, 4.0), seed=0)
    np.testing.assert_array_equal(result.assignment, scaled.assignment)


def test_equal_sized_clusters_make_unit_counts_follow_the_cluster_ratio() -> None:
    ids = np.repeat(np.arange(12), 2)
    result = cluster_randomization(ids, n_arms=3, ratio=(1, 1, 2), seed=1)
    assert result.arm_labels == ("arm_0", "arm_1", "arm_2")
    assert result.cluster_counts == (3, 3, 6)
    assert result.arm_counts == (6, 6, 12)
    _assert_clusters_are_intact(result)


def test_assignment_is_reproducible_and_depends_on_seed() -> None:
    ids = np.repeat(np.arange(16), 2)
    first = cluster_randomization(ids, seed=11)
    second = cluster_randomization(ids, seed=11)
    third = cluster_randomization(ids, seed=12)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    np.testing.assert_array_equal(first.cluster_arms, second.cluster_arms)
    assert not np.array_equal(first.assignment, third.assignment)
    fresh = cluster_randomization(np.array([1, 1, 2, 2, 3, 3, 4, 4]), seed=None)
    assert fresh.seed is None
    assert fresh.cluster_counts == (2, 2)


def test_custom_arm_labels() -> None:
    result = cluster_randomization(
        np.array([1, 1, 2, 2]), arm_labels=("holdout", "ship"), seed=0
    )
    assert result.arm_labels == ("holdout", "ship")


@pytest.mark.parametrize("n_clusters", [2, 3, 4, 5, 9, 12])
def test_equal_cluster_counts_differ_by_at_most_one(n_clusters: int) -> None:
    ids = np.repeat(np.arange(n_clusters), 3)
    result = cluster_randomization(ids, seed=n_clusters)
    assert abs(result.cluster_counts[0] - result.cluster_counts[1]) <= 1
    assert sum(result.arm_counts) == ids.size
    _assert_clusters_are_intact(result)


def test_cluster_balance_uses_cluster_means_not_unit_means() -> None:
    ids = np.array([0, 0, 0, 1, 2, 3, 3, 3])
    score = np.array([0, 0, 0, 10, 0, 10, 10, 10], dtype=float)
    assignment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    report = cluster_balance_report({"score": score}, assignment, ids)
    unit = balance_report({"score": score}, assignment)
    assert report.arm_counts == (2, 2)
    assert report.rows[0].means == pytest.approx((5.0, 5.0))
    assert report.rows[0].max_abs_smd == pytest.approx(0.0)
    assert unit.rows[0].means == pytest.approx((2.5, 7.5))
    assert unit.rows[0].max_abs_smd > 0


def test_cluster_balance_keeps_a_constant_cluster_covariate() -> None:
    ids = np.array([0, 0, 0, 1, 2, 3, 3, 3])
    region = np.array(["e", "e", "e", "w", "e", "w", "w", "w"])
    score = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0])
    assignment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    report = cluster_balance_report(
        {"region": region, "score": score},
        assignment,
        ids,
        arm_labels=("holdout", "ship"),
        categorical=["region"],
        reference_arm=0,
    )
    assert report.arm_labels == ("holdout", "ship")
    assert report.arm_counts == (2, 2)
    assert report.rows[0].kind == "categorical"
    assert report.rows[0].max_abs_smd == pytest.approx(0.0)
    assert report.rows[1].kind == "continuous"
    assert report.rows[1].means == pytest.approx((1.5, 3.5))


def test_cluster_balance_rejects_a_split_or_a_varying_category() -> None:
    ids = np.array([1, 1, 2, 2])
    assignment = np.array([0, 0, 1, 1])
    varying = np.array(["e", "w", "e", "e"])
    with pytest.raises(ValueError, match="varies within a cluster"):
        cluster_balance_report({"region": varying}, assignment, ids)
    with pytest.raises(ValueError, match="splits a cluster"):
        cluster_balance_report(
            {"score": np.array([1.0, 2.0, 3.0, 4.0])},
            np.array([0, 1, 1, 1]),
            ids,
        )
    with pytest.raises(ValueError, match="same length"):
        cluster_balance_report(
            {"score": np.array([1.0, 2.0])},
            assignment,
            ids,
        )
    with pytest.raises(ValueError, match="at least one covariate"):
        cluster_balance_report({}, assignment, ids)


def test_cluster_randomization_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        cluster_randomization(np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="non-empty"):
        cluster_randomization(np.array([]))
    with pytest.raises(ValueError, match="one-dimensional"):
        cluster_randomization(np.array([[1, 2], [1, 2]]))
    with pytest.raises(ValueError, match="missing"):
        cluster_randomization(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="missing"):
        cluster_randomization(np.array(["a", None, "b"], dtype=object))
    with pytest.raises(ValueError, match="n_arms"):
        cluster_randomization(np.array([1, 2, 3, 4]), n_arms=1)
    with pytest.raises(ValueError, match="n_arms"):
        cluster_randomization(np.array([1, 2]), n_arms=True)
    with pytest.raises(ValueError, match="ratio"):
        cluster_randomization(np.array([1, 2, 3, 4]), ratio=(1, 0))
    with pytest.raises(ValueError, match="ratio"):
        cluster_randomization(np.array([1, 2, 3, 4]), ratio=(1,))
    with pytest.raises(ValueError, match="seed"):
        cluster_randomization(np.array([1, 2]), seed=True)
    with pytest.raises(ValueError, match="arm_labels"):
        cluster_randomization(np.array([1, 2]), arm_labels=("only",))
    with pytest.raises(ValueError, match="at least 3 clusters"):
        cluster_randomization(np.array([1, 1, 2, 2]), n_arms=3)
