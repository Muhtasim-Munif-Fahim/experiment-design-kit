"""Tests for delta-method ratio metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiment_design_kit import (
    RatioMetricResult,
    delta_method_ratio_test,
    ratio_metric_variance,
)


def _users(n, ctr, rng, heterogeneity=2.0):
    """Per-user page views and clicks with user-level CTR heterogeneity."""
    views = rng.poisson(5, size=n) + 1
    a = ctr * heterogeneity * 5
    b = (1 - ctr) * heterogeneity * 5
    user_ctr = rng.beta(a, b, size=n)
    clicks = rng.binomial(views, user_ctr)
    return clicks.astype(float), views.astype(float)


def test_variance_matches_closed_form():
    y = np.array([1.0, 2.0, 0.0, 3.0, 4.0])
    x = np.array([2.0, 3.0, 1.0, 4.0, 5.0])
    ratio, var = ratio_metric_variance(y, x)
    mu_y, mu_x = y.mean(), x.mean()
    s = np.cov(y, x, ddof=1)
    expected = (s[0, 0] / mu_x**2 - 2 * mu_y * s[0, 1] / mu_x**3
                + mu_y**2 * s[1, 1] / mu_x**4) / len(y)
    assert ratio == pytest.approx(y.sum() / x.sum())
    assert var == pytest.approx(expected)


def test_constant_denominator_reduces_to_mean_variance():
    rng = np.random.default_rng(0)
    y = rng.normal(3.0, 1.0, size=500)
    x = np.full(500, 2.0)
    ratio, var = ratio_metric_variance(y, x)
    assert ratio == pytest.approx(y.mean() / 2.0)
    assert var == pytest.approx(y.var(ddof=1) / 4.0 / 500)


def test_delta_variance_matches_monte_carlo():
    rng = np.random.default_rng(1)
    ratios, variances = [], []
    for _ in range(400):
        y, x = _users(400, 0.1, rng)
        r, v = ratio_metric_variance(y, x)
        ratios.append(r)
        variances.append(v)
    assert np.mean(variances) == pytest.approx(np.var(ratios, ddof=1), rel=0.2)


def test_naive_event_level_se_is_too_small():
    rng = np.random.default_rng(2)
    y, x = _users(2000, 0.1, rng, heterogeneity=0.5)
    r, v = ratio_metric_variance(y, x)
    naive_var = r * (1 - r) / x.sum()  # treats every page view as independent
    assert v > 1.5 * naive_var


def test_null_false_positive_rate_near_alpha():
    rng = np.random.default_rng(3)
    rejections = 0
    sims = 400
    for _ in range(sims):
        yc, xc = _users(300, 0.1, rng)
        yt, xt = _users(300, 0.1, rng)
        rejections += delta_method_ratio_test(yc, xc, yt, xt).significant
    assert rejections / sims < 0.09


def test_detects_real_lift_and_relative_ci():
    rng = np.random.default_rng(4)
    yc, xc = _users(5000, 0.10, rng)
    yt, xt = _users(5000, 0.12, rng)
    res = delta_method_ratio_test(yc, xc, yt, xt, alpha=0.05)
    assert isinstance(res, RatioMetricResult)
    assert res.significant
    assert res.ci_low < res.absolute_diff < res.ci_high
    assert res.ci_low > 0
    assert res.relative_lift == pytest.approx(res.ratio_treatment / res.ratio_control - 1)
    assert res.relative_ci_low < 0.2 < res.relative_ci_high
    assert res.n_control == res.n_treatment == 5000
    assert res.z == pytest.approx(res.absolute_diff / res.se_diff)


def test_relative_se_formula():
    rng = np.random.default_rng(5)
    yc, xc = _users(800, 0.1, rng)
    yt, xt = _users(800, 0.1, rng)
    res = delta_method_ratio_test(yc, xc, yt, xt)
    rc, vc = ratio_metric_variance(yc, xc)
    rt, vt = ratio_metric_variance(yt, xt)
    assert res.se_relative == pytest.approx(math.sqrt(vt / rc**2 + rt**2 * vc / rc**4))


def test_validation():
    with pytest.raises(ValueError, match="same length"):
        ratio_metric_variance([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="two"):
        ratio_metric_variance([1], [1])
    with pytest.raises(ValueError, match="positive"):
        ratio_metric_variance([1, 2], [0, 0])
    with pytest.raises(ValueError, match="finite"):
        ratio_metric_variance([1, np.nan], [1, 1])
    with pytest.raises(ValueError, match="alpha"):
        delta_method_ratio_test([1, 2], [1, 2], [1, 2], [1, 2], alpha=1.0)
    with pytest.raises(ValueError, match="treatment"):
        delta_method_ratio_test([1, 2], [1, 2], [1], [1])
