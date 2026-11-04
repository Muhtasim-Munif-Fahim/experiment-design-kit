import numpy as np
import pytest

from experiment_design_kit import (
    ab_ba_crossover_analysis,
    carryover_balance,
    crossover_sample_size,
    is_carryover_balanced,
    williams_design,
)


@pytest.mark.parametrize("t", [2, 3, 4, 5, 6, 7, 8])
def test_williams_balanced(t):
    D = williams_design(t)
    assert D.shape == ((t if t % 2 == 0 else 2 * t), t)
    for row in D:
        assert sorted(row) == list(range(t))
    assert is_carryover_balanced(D)
    M = carryover_balance(D)
    assert np.all(np.diag(M) == 0)


def test_williams_t4_known_square():
    assert williams_design(4).tolist() == [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]


def test_cyclic_latin_square_is_not_carryover_balanced():
    t = 4
    cyclic = np.array([[(i + j) % t for j in range(t)] for i in range(t)])
    assert not is_carryover_balanced(cyclic)
    # Odd t: a single Williams square alone is not balanced.
    assert not is_carryover_balanced(williams_design(5)[:5])


def _simulate(rng, n, tau, period, carry, sb=2.0, sw=1.0):
    s_ab = rng.normal(0, sb, n)
    s_ba = rng.normal(0, sb, n)
    ab1 = 10 + s_ab + tau + rng.normal(0, sw, n)
    ab2 = 10 + s_ab + period + carry + rng.normal(0, sw, n)
    ba1 = 10 + s_ba + rng.normal(0, sw, n)
    ba2 = 10 + s_ba + period + tau + rng.normal(0, sw, n)
    return ab1, ab2, ba1, ba2


def test_analysis_recovers_effects():
    rng = np.random.default_rng(1)
    res = ab_ba_crossover_analysis(*_simulate(rng, 400, tau=1.5, period=-0.7, carry=0.0))
    assert res.treatment_effect == pytest.approx(1.5, abs=0.1)
    assert res.period_effect == pytest.approx(-0.7, abs=0.1)
    assert res.treatment_ci[0] < 1.5 < res.treatment_ci[1]
    assert res.treatment_p < 1e-10 and res.period_p < 1e-6
    assert res.within_subject_sd == pytest.approx(1.0, abs=0.08)
    assert res.df == 798


def test_hand_example_matches_closed_form():
    ab1, ab2 = [5.0, 7.0, 6.0], [3.0, 4.0, 4.0]
    ba1, ba2 = [4.0, 5.0, 3.0], [6.0, 6.0, 5.0]
    r = ab_ba_crossover_analysis(ab1, ab2, ba1, ba2)
    d_ab = np.array(ab1) - ab2
    d_ba = np.array(ba1) - ba2
    assert r.treatment_effect == pytest.approx((d_ab.mean() - d_ba.mean()) / 2)
    assert r.period_effect == pytest.approx(-(d_ab.mean() + d_ba.mean()) / 2)
    sp2 = (d_ab.var(ddof=1) + d_ba.var(ddof=1)) / 2
    assert r.treatment_se == pytest.approx(np.sqrt(sp2 * (2 / 3)) / 2)


def test_type1_error_and_carryover_detection():
    rng = np.random.default_rng(2)
    rej = sum(ab_ba_crossover_analysis(*_simulate(rng, 15, 0.0, 0.5, 0.0)).treatment_p < 0.05 for _ in range(600))
    assert abs(rej / 600 - 0.05) < 0.025
    flagged = ab_ba_crossover_analysis(*_simulate(rng, 300, 1.0, 0.0, carry=3.0, sb=1.0))
    assert flagged.carryover_flag and flagged.carryover_p < 0.01
    clean = [ab_ba_crossover_analysis(*_simulate(rng, 30, 1.0, 0.0, 0.0)).carryover_flag for _ in range(300)]
    assert abs(np.mean(clean) - 0.10) < 0.05


def test_sample_size_and_power_by_simulation():
    ss = crossover_sample_size(1.0, sigma_w=1.5, power=0.8, sigma_b=3.0)
    assert ss.achieved_power >= 0.8
    assert ss.n_total == 2 * ss.n_per_sequence
    assert ss.parallel_n_total > 3 * ss.n_total  # crossover is far cheaper
    smaller = crossover_sample_size(1.0, sigma_w=1.5, power=0.8)
    assert smaller.n_per_sequence == ss.n_per_sequence and smaller.parallel_n_total is None
    # m-1 per sequence is underpowered.
    rng = np.random.default_rng(3)
    m = ss.n_per_sequence
    hits = sum(
        ab_ba_crossover_analysis(*_simulate(rng, m, 1.0, 0.0, 0.0, sb=3.0, sw=1.5)).treatment_p < 0.05
        for _ in range(800)
    )
    assert abs(hits / 800 - ss.achieved_power) < 0.05


def test_validation():
    with pytest.raises(ValueError):
        williams_design(1)
    with pytest.raises(ValueError):
        ab_ba_crossover_analysis([1.0], [1.0], [1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ab_ba_crossover_analysis([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        crossover_sample_size(0.0, 1.0)
    with pytest.raises(ValueError):
        crossover_sample_size(1.0, 0.0)
    with pytest.raises(ValueError):
        carryover_balance([[0, 1, 2]][0])
