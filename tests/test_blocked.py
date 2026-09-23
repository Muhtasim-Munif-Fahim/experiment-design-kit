"""Tests for fixed-size permuted-block randomization."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import blocked_randomization


def _block_counts(assignment: np.ndarray, block_size: int, n_arms: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for start in range(0, assignment.size, block_size):
        chunk = assignment[start : start + block_size]
        rows.append(np.bincount(chunk, minlength=n_arms).tolist())
    return rows


def test_equal_blocks_of_size_2k_match_exactly() -> None:
    result = blocked_randomization(20, block_size=4, seed=0)
    assert result.n == 20
    assert result.block_size == 4
    assert result.n_blocks == 5
    assert result.n_complete_blocks == 5
    assert result.arm_labels == ("control", "treatment")
    assert result.block_target == (2, 2)
    assert result.block_arm_counts == ((2, 2),) * 5
    assert result.arm_counts == (10, 10)
    assert _block_counts(result.assignment, 4, 2) == [[2, 2]] * 5
    np.testing.assert_array_equal(result.block_ids, np.arange(20) // 4)


def test_smallest_equal_block_alternates_one_of_each_arm() -> None:
    result = blocked_randomization(8, block_size=2, seed=3)
    assert result.block_target == (1, 1)
    assert result.arm_counts == (4, 4)
    assert all(row == (1, 1) for row in result.block_arm_counts)


def test_ratio_is_exact_inside_every_complete_block() -> None:
    result = blocked_randomization(18, block_size=6, ratio=(1, 2), seed=0)
    assert result.block_target == (2, 4)
    assert result.block_arm_counts == ((2, 4), (2, 4), (2, 4))
    assert result.arm_counts == (6, 12)
    scaled = blocked_randomization(18, block_size=6, ratio=(2.0, 4.0), seed=0)
    np.testing.assert_array_equal(result.assignment, scaled.assignment)


def test_three_arms_and_unequal_ratio() -> None:
    equal = blocked_randomization(12, block_size=3, n_arms=3, seed=0)
    assert equal.arm_labels == ("arm_0", "arm_1", "arm_2")
    assert equal.block_target == (1, 1, 1)
    assert equal.arm_counts == (4, 4, 4)

    uneven = blocked_randomization(
        8, block_size=4, n_arms=3, ratio=(1, 2, 1), seed=1
    )
    assert uneven.block_target == (1, 2, 1)
    assert uneven.block_arm_counts == ((1, 2, 1), (1, 2, 1))
    assert uneven.arm_counts == (2, 4, 2)


def test_assignment_is_reproducible_and_depends_on_seed() -> None:
    first = blocked_randomization(40, block_size=4, seed=11)
    second = blocked_randomization(40, block_size=4, seed=11)
    third = blocked_randomization(40, block_size=4, seed=12)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    np.testing.assert_array_equal(first.block_ids, second.block_ids)
    assert not np.array_equal(first.assignment, third.assignment)
    fresh = blocked_randomization(6, block_size=2, seed=None)
    assert fresh.seed is None
    assert fresh.arm_counts == (3, 3)


def test_within_block_order_is_not_a_fixed_pattern() -> None:
    result = blocked_randomization(40, block_size=4, seed=0)
    blocks = result.assignment.reshape(10, 4)
    assert len({tuple(row.tolist()) for row in blocks}) > 1


def test_final_block_keeps_equal_allocation_within_one_unit() -> None:
    result = blocked_randomization(10, block_size=4, seed=0)
    assert result.n_blocks == 3
    assert result.n_complete_blocks == 2
    assert result.block_arm_counts[0] == (2, 2)
    assert result.block_arm_counts[1] == (2, 2)
    assert result.block_arm_counts[2] == (1, 1)
    assert result.arm_counts == (5, 5)
    np.testing.assert_array_equal(result.block_ids, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]))


def test_odd_remainder_gives_one_arm_the_extra_unit() -> None:
    result = blocked_randomization(7, block_size=4, seed=2)
    assert result.block_arm_counts[0] == (2, 2)
    assert sorted(result.block_arm_counts[1]) == [1, 2]
    assert sorted(result.arm_counts) == [3, 4]


def test_short_study_uses_only_a_partial_block() -> None:
    result = blocked_randomization(4, block_size=6, ratio=(1, 2), seed=0)
    assert result.n_complete_blocks == 0
    assert result.n_blocks == 1
    assert result.block_target == (2, 4)
    assert result.arm_counts == (1, 3)
    assert result.block_ids.tolist() == [0, 0, 0, 0]


def test_single_unit_is_a_partial_block() -> None:
    result = blocked_randomization(1, block_size=2, seed=0)
    assert result.assignment.shape == (1,)
    assert result.n_blocks == 1
    assert result.n_complete_blocks == 0
    assert sum(result.block_arm_counts[0]) == 1
    assert int(result.assignment[0]) in (0, 1)


@pytest.mark.parametrize("block_size", [2, 4, 6, 10])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 15, 20, 21])
def test_equal_allocation_overall_imbalance_is_at_most_one(n: int, block_size: int) -> None:
    result = blocked_randomization(n, block_size=block_size, seed=n + block_size)
    assert sum(result.arm_counts) == n
    assert abs(result.arm_counts[0] - result.arm_counts[1]) <= 1
    for index, row in enumerate(result.block_arm_counts):
        assert sum(row) == (block_size if index < result.n_complete_blocks else n % block_size)
        if index < result.n_complete_blocks:
            assert row == result.block_target


@pytest.mark.parametrize("n", [3, 4, 5, 8, 9, 17])
def test_one_to_two_ratio_tracks_the_weights(n: int) -> None:
    result = blocked_randomization(n, block_size=6, ratio=(1, 2), seed=1)
    assert sum(result.arm_counts) == n
    assert abs(result.arm_counts[0] - n / 3) < 1
    assert abs(result.arm_counts[1] - 2 * n / 3) < 1
    for index, row in enumerate(result.block_arm_counts):
        if index < result.n_complete_blocks:
            assert row == (2, 4)


def test_custom_arm_labels() -> None:
    result = blocked_randomization(
        6, block_size=2, arm_labels=("holdout", "ship"), seed=0
    )
    assert result.arm_labels == ("holdout", "ship")


def test_blocked_randomization_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="2k"):
        blocked_randomization(10, block_size=3)
    with pytest.raises(ValueError, match="allocation period 3"):
        blocked_randomization(12, block_size=4, ratio=(1, 2))
    with pytest.raises(ValueError, match="block_size"):
        blocked_randomization(10, block_size=1)
    with pytest.raises(ValueError, match="block_size"):
        blocked_randomization(10, block_size=True)
    with pytest.raises(ValueError, match="n must"):
        blocked_randomization(0, block_size=2)
    with pytest.raises(ValueError, match="n must"):
        blocked_randomization(True, block_size=2)
    with pytest.raises(ValueError, match="n_arms"):
        blocked_randomization(6, block_size=2, n_arms=1)
    with pytest.raises(ValueError, match="ratio"):
        blocked_randomization(6, block_size=2, ratio=(1, 0))
    with pytest.raises(ValueError, match="ratio"):
        blocked_randomization(6, block_size=2, ratio=(1,))
    with pytest.raises(ValueError, match="seed"):
        blocked_randomization(6, block_size=2, seed=True)
    with pytest.raises(ValueError, match="arm_labels"):
        blocked_randomization(6, block_size=2, arm_labels=("only",))
