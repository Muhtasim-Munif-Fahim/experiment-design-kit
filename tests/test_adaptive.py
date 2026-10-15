"""Tests for Thompson sampling adaptive allocation."""

from __future__ import annotations

import pytest

from experiment_design_kit.adaptive import ThompsonAllocator, simulate_thompson_ab


def test_allocator_prefers_better_arm():
    res = simulate_thompson_ab([0.1, 0.4], n=400, arm_names=["A", "B"], seed=0)
    assert res.counts["B"] > res.counts["A"]
    assert res.prob_best["B"] > 0.7


def test_update_and_prob_best():
    alloc = ThompsonAllocator(["control", "treatment"], seed=1)
    for _ in range(30):
        alloc.update("treatment", 1.0)
        alloc.update("control", 0.0)
    pb = alloc.prob_best(n_draws=2000)
    assert pb["treatment"] > 0.9
    s = alloc.summary(n_draws=500)
    assert s.n == 60


def test_rejects_single_arm():
    with pytest.raises(ValueError, match="two arms"):
        ThompsonAllocator(["only"])


def test_rejects_bad_reward():
    alloc = ThompsonAllocator(["a", "b"], seed=0)
    with pytest.raises(ValueError, match="reward"):
        alloc.update("a", 1.5)
