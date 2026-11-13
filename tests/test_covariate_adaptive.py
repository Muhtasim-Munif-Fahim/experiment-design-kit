import numpy as np
import pandas as pd
import pytest

from experiment_design_kit import (
    efron_biased_coin,
    marginal_imbalance,
    pocock_simon_minimization,
)


def _units(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "sex": rng.choice(["F", "M"], size=n),
            "age": rng.choice(["<40", "40-60", ">60"], size=n, p=[0.3, 0.5, 0.2]),
            "site": rng.choice(list("ABCD"), size=n),
        }
    )


def test_efron_deterministic_limit_alternates():
    res = efron_biased_coin(51, p=1.0, seed=0)
    assert res.max_abs_imbalance == 1
    assert abs(res.final_imbalance) == 1
    np.testing.assert_array_equal(np.cumsum(2 * res.assignment - 1), res.imbalance_path)


def test_efron_keeps_imbalance_small_vs_coin_flip():
    efron = [abs(efron_biased_coin(400, seed=s).final_imbalance) for s in range(200)]
    coin = [abs(efron_biased_coin(400, p=0.5, seed=s).final_imbalance) for s in range(200)]
    assert np.mean(efron) < 3.0
    assert np.mean(coin) > 10.0  # ~ sqrt(2 n / pi) = 16
    assert np.mean(efron) < np.mean(coin) / 4


def test_minimization_balances_margins_better_than_randomization():
    units = _units()
    mini = [pocock_simon_minimization(units, seed=s).imbalance["total_range"] for s in range(20)]
    rand = [pocock_simon_minimization(units, p_best=0.5, seed=s).imbalance["total_range"] for s in range(20)]
    assert np.mean(mini) < np.mean(rand) / 2


def test_deterministic_minimization_two_arms_tight():
    units = _units(n=80, seed=1)
    res = pocock_simon_minimization(units, p_best=1.0, seed=0)
    assert res.imbalance["max_range"] <= 3
    assert res.imbalance["overall_range"] <= 2
    assert 0.0 < res.deterministic_fraction <= 1.0
    assert res.arm_sizes.sum() == 80


def test_first_unit_and_hand_example():
    # One factor; deterministic: units alternate within each level.
    units = pd.DataFrame({"g": ["a", "a", "b", "b", "a", "a"]})
    res = pocock_simon_minimization(units, p_best=1.0, seed=3)
    a = res.assignment
    assert a[0] != a[1] and a[2] != a[3] and a[4] != a[5]
    assert res.imbalance["factors"]["g"]["a"]["range"] == 0
    assert res.imbalance["factors"]["g"]["b"]["range"] == 0


def test_multi_arm_weights_measures_and_ratio():
    units = _units(n=150, seed=2)
    for measure in ("range", "variance", "sd"):
        res = pocock_simon_minimization(units, n_arms=3, measure=measure, weights={"site": 2.0}, seed=1)
        assert res.n_arms == 3 and res.arm_sizes.sum() == 150
        assert res.imbalance["overall_range"] <= 6
        assert res.weights == {"sex": 1.0, "age": 1.0, "site": 2.0}
    res = pocock_simon_minimization(units, allocation_ratio=(2, 1), p_best=0.9, seed=0)
    share = res.arm_sizes[0] / res.arm_sizes.sum()
    assert 0.6 < share < 0.73


def test_array_input_and_determinism():
    arr = _units(n=40).to_numpy()
    a = pocock_simon_minimization(arr, seed=5).assignment
    b = pocock_simon_minimization(arr, seed=5).assignment
    np.testing.assert_array_equal(a, b)


def test_marginal_imbalance_counts():
    units = pd.DataFrame({"x": ["u", "u", "v", "v", "v"]})
    out = marginal_imbalance(units, [0, 1, 0, 0, 1], n_arms=2)
    assert out["factors"]["x"]["u"] == {"counts": [1, 1], "range": 0}
    assert out["factors"]["x"]["v"] == {"counts": [2, 1], "range": 1}
    assert out["total_range"] == 1 and out["overall_range"] == 1


def test_validation():
    units = _units(n=10)
    with pytest.raises(ValueError):
        efron_biased_coin(0)
    with pytest.raises(ValueError):
        efron_biased_coin(10, p=0.3)
    with pytest.raises(ValueError):
        pocock_simon_minimization(units, n_arms=1)
    with pytest.raises(ValueError):
        pocock_simon_minimization(units, p_best=0.2)
    with pytest.raises(ValueError):
        pocock_simon_minimization(units, measure="mad")
    with pytest.raises(ValueError):
        pocock_simon_minimization(units, weights={"zzz": 1.0})
    with pytest.raises(ValueError):
        pocock_simon_minimization(units, weights=[1.0])
    with pytest.raises(ValueError):
        pocock_simon_minimization(units, allocation_ratio=(1, 0))
    with pytest.raises(ValueError):
        pocock_simon_minimization(units.assign(sex=None))
    with pytest.raises(ValueError):
        marginal_imbalance(units, [0, 1])
