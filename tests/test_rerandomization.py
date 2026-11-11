import numpy as np
import pytest
from scipy import stats

from experiment_design_kit import (
    mahalanobis_balance,
    rerandomization_test,
    rerandomization_threshold,
    rerandomization_variance_reduction,
    rerandomize,
)


def _X(n=60, k=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, k))


def test_mahalanobis_matches_formula():
    X = _X()
    w = np.r_[np.ones(30, int), np.zeros(30, int)]
    d = X[w == 1].mean(0) - X[w == 0].mean(0)
    S = np.cov(X, rowvar=False)
    expected = 30 * 30 / 60 * d @ np.linalg.solve(S, d)
    assert mahalanobis_balance(X, w) == pytest.approx(expected)


def test_threshold_and_variance_reduction_closed_form():
    a = rerandomization_threshold(4, 0.05)
    assert stats.chi2.cdf(a, 4) == pytest.approx(0.05)
    red = rerandomization_variance_reduction(4, a)
    assert red == pytest.approx(1 - stats.chi2.cdf(a, 6) / 0.05)
    assert 0 < red < 1
    assert rerandomization_threshold(2, 1.0) == float("inf")
    assert rerandomization_variance_reduction(2, float("inf")) == 0.0
    # tighter acceptance -> more reduction
    assert rerandomization_variance_reduction(4, rerandomization_threshold(4, 0.001)) > red


def test_rerandomize_accepts_balanced_assignment_reproducibly():
    X = _X(n=80, k=4, seed=1)
    res = rerandomize(X, acceptance_prob=0.01, seed=7)
    assert res.assignment.sum() == 40 and res.n_treated == 40
    assert res.mahalanobis <= res.threshold
    assert res.mahalanobis == pytest.approx(mahalanobis_balance(X, res.assignment))
    assert res.n_draws >= 1 and res.acceptance_prob == 0.01
    assert res.variance_factor == pytest.approx(1 - res.variance_reduction)
    again = rerandomize(X, acceptance_prob=0.01, seed=7)
    np.testing.assert_array_equal(res.assignment, again.assignment)
    uneven = rerandomize(X, n_treated=20, threshold=2.0, seed=3)
    assert uneven.assignment.sum() == 20 and uneven.mahalanobis <= 2.0
    assert uneven.acceptance_prob == pytest.approx(stats.chi2.cdf(2.0, 4))


def test_empirical_variance_reduction_matches_theory():
    X = _X(n=200, k=2, seed=2)
    a = rerandomization_threshold(2, 0.1)
    rng = np.random.default_rng(0)
    plain, rerand = [], []
    for s in range(400):
        w = np.zeros(200, int)
        w[rng.permutation(200)[:100]] = 1
        plain.append(X[w == 1, 0].mean() - X[w == 0, 0].mean())
        r = rerandomize(X, threshold=a, seed=s)
        rerand.append(X[r.assignment == 1, 0].mean() - X[r.assignment == 0, 0].mean())
    ratio = np.var(rerand) / np.var(plain)
    v_a = 1 - rerandomization_variance_reduction(2, a)
    assert ratio == pytest.approx(v_a, rel=0.3)
    assert abs(np.mean(rerand)) < 0.02  # still unbiased (centered at 0)


def test_collinear_covariates_use_rank():
    X = _X(n=50, k=2, seed=4)
    X = np.c_[X, X[:, 0] + X[:, 1]]
    res = rerandomize(X, acceptance_prob=0.05, seed=0)
    assert res.threshold == pytest.approx(rerandomization_threshold(2, 0.05))


def test_randomization_test_detects_effect_and_controls_null():
    X = _X(n=60, k=2, seed=5)
    design = rerandomize(X, acceptance_prob=0.05, seed=1)
    rng = np.random.default_rng(3)
    y0 = X @ np.array([1.0, -0.5]) + rng.normal(size=60)
    y1 = y0 + 1.5 * design.assignment
    hit = rerandomization_test(y1, X, design.assignment, threshold=design.threshold, n_permutations=499, seed=0)
    assert hit.p_value < 0.01 and hit.estimate > 0.8 and hit.n_permutations == 499
    null = rerandomization_test(y0, X, design.assignment, threshold=design.threshold, n_permutations=499, seed=0)
    assert null.p_value > 10 * hit.p_value
    greater = rerandomization_test(y1, X, design.assignment, threshold=design.threshold, alternative="greater", n_permutations=199, seed=0)
    less = rerandomization_test(y1, X, design.assignment, threshold=design.threshold, alternative="less", n_permutations=199, seed=0)
    assert greater.p_value < 0.05 < less.p_value


def test_randomization_test_null_p_values_are_valid():
    X = _X(n=30, k=2, seed=6)
    a = rerandomization_threshold(2, 0.2)
    rng = np.random.default_rng(11)
    pvals = []
    for s in range(150):
        design = rerandomize(X, threshold=a, seed=1000 + s)
        y = X[:, 0] + rng.normal(size=30)
        pvals.append(rerandomization_test(y, X, design.assignment, threshold=a, n_permutations=99, seed=s).p_value)
    assert np.mean(np.array(pvals) <= 0.1) < 0.18


def test_validation():
    X = _X()
    with pytest.raises(ValueError, match="acceptance_prob"):
        rerandomize(X, acceptance_prob=0.0)
    with pytest.raises(ValueError, match="n_treated"):
        rerandomize(X, n_treated=60)
    with pytest.raises(ValueError, match="vary"):
        rerandomize(np.c_[X, np.ones(60)])
    with pytest.raises(ValueError, match="0/1"):
        mahalanobis_balance(X, np.full(60, 2))
    with pytest.raises(ValueError, match="each arm"):
        mahalanobis_balance(X, np.zeros(60, int))
    with pytest.raises(RuntimeError, match="max_draws"):
        rerandomize(X, threshold=1e-9, max_draws=50, seed=0)
    with pytest.raises(ValueError, match="alternative"):
        rerandomization_test(X[:, 0], X, np.r_[np.ones(30), np.zeros(30)], alternative="both")
