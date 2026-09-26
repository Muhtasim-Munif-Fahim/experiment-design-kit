"""Tests for switchback / time-based block randomization."""
from __future__ import annotations

import numpy as np
import pytest

from experiment_design_kit import SwitchbackAssignment, switchback_randomization


def _assert_blocks_intact(result: SwitchbackAssignment) -> None:
    assert result.assignment.shape == result.block_ids.shape
    assert result.block_arms.shape == (result.n_blocks,)
    assert sum(result.block_counts) == result.n_blocks
    assert sum(result.arm_counts) == result.n
    for block in range(result.n_blocks):
        mask = result.block_ids == block
        assert int(mask.sum()) >= 1
        assert np.all(result.assignment[mask] == result.block_arms[block])


def test_periods_in_a_block_share_one_arm() -> None:
    result = switchback_randomization(12, block_length=3, seed=0)
    assert result.n == 12
    assert result.n_blocks == 4
    assert result.block_length == 3
    assert result.arm_labels == ("control", "treatment")
    assert result.seed == 0
    _assert_blocks_intact(result)
    # Contiguous block ids 0,0,0,1,1,1,...
    np.testing.assert_array_equal(
        result.block_ids, np.repeat(np.arange(4), 3)
    )


def test_short_final_block() -> None:
    result = switchback_randomization(10, block_length=4, seed=1)
    assert result.n_blocks == 3
    assert int((result.block_ids == 2).sum()) == 2
    _assert_blocks_intact(result)


def test_ratio_allocates_blocks_not_periods() -> None:
    result = switchback_randomization(24, block_length=2, ratio=(1, 2), seed=0)
    assert result.n_blocks == 12
    assert result.block_counts == (4, 8)
    _assert_blocks_intact(result)
    scaled = switchback_randomization(24, block_length=2, ratio=(2.0, 4.0), seed=0)
    np.testing.assert_array_equal(result.assignment, scaled.assignment)


def test_assignment_is_reproducible_and_depends_on_seed() -> None:
    first = switchback_randomization(40, block_length=5, seed=11)
    second = switchback_randomization(40, block_length=5, seed=11)
    third = switchback_randomization(40, block_length=5, seed=12)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    np.testing.assert_array_equal(first.block_arms, second.block_arms)
    assert not np.array_equal(first.assignment, third.assignment)


def test_three_arms_and_custom_labels() -> None:
    result = switchback_randomization(
        18, block_length=3, n_arms=3, arm_labels=("a", "b", "c"), seed=0
    )
    assert result.arm_labels == ("a", "b", "c")
    assert result.block_counts == (2, 2, 2)
    _assert_blocks_intact(result)


def test_rejects_too_few_blocks_for_arms() -> None:
    with pytest.raises(ValueError, match="need at least 3 blocks"):
        switchback_randomization(4, block_length=2, n_arms=3, seed=0)


def test_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError, match="n must be an integer >= 1"):
        switchback_randomization(0, block_length=1)
    with pytest.raises(ValueError, match="block_length must be an integer >= 1"):
        switchback_randomization(10, block_length=0)
    with pytest.raises(ValueError, match="block_length must be <= n"):
        switchback_randomization(5, block_length=6)
