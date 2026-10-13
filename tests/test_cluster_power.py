"""Tests for ICC-aware cluster-randomized power and sample size."""

from __future__ import annotations

import io
import math
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest
from scipy import stats as sp

from experiment_design_kit import (
    cluster_mde,
    cluster_power,
    cluster_sample_size,
    design_effect,
    effective_sample_size,
    estimate_icc,
    power_two_sample,
)
from experiment_design_kit.cli import main


def test_design_effect_classic_and_variable_sizes():
    assert design_effect(20, 0.05) == pytest.approx(1.95)
    assert design_effect(1, 0.3) == pytest.approx(1.0)
    assert design_effect(20, 0.0) == 1.0
    # Eldridge et al.: DE = 1 + ((cv^2 + 1) m - 1) icc
    assert design_effect(20, 0.05, cv=0.5) == pytest.approx(1 + (1.25 * 20 - 1) * 0.05)
    assert effective_sample_size(1000, 20, 0.05) == pytest.approx(1000 / 1.95)
    with pytest.raises(ValueError):
        design_effect(0.5, 0.1)
    with pytest.raises(ValueError):
        design_effect(10, 1.5)
    with pytest.raises(ValueError):
        design_effect(10, 0.1, cv=-1)


def test_normal_sample_size_matches_donner_klar_formula():
    d, m, icc = 0.3, 25, 0.02
    res = cluster_sample_size(m, icc, effect_size=d, method="normal")
    z = sp.norm.ppf(0.975) + sp.norm.ppf(0.8)
    n_ind = 2 * z**2 / d**2
    assert res.n_individual_per_arm == pytest.approx(n_ind)
    assert res.design_effect == pytest.approx(1 + 24 * 0.02)
    assert res.clusters_per_arm == math.ceil(n_ind * res.design_effect / m)
    assert res.total_clusters == 2 * res.clusters_per_arm
    assert res.individuals_per_arm == res.clusters_per_arm * m


def test_t_method_needs_at_least_as_many_clusters_and_hits_target():
    for icc in (0.01, 0.05, 0.2):
        n = cluster_sample_size(30, icc, effect_size=0.4, method="normal")
        t = cluster_sample_size(30, icc, effect_size=0.4, method="t")
        assert t.clusters_per_arm >= n.clusters_per_arm
        assert t.achieved_power >= 0.8
        assert cluster_power(t.clusters_per_arm - 1, 30, icc, effect_size=0.4) < 0.8 or t.clusters_per_arm == 2


def test_icc_zero_and_m_one_reduce_to_individual_t_test():
    # m = 1 => cluster-level t-test with k units per arm == individual two-sample t
    for k in (5, 12, 40):
        assert cluster_power(k, 1, 0.3, effect_size=0.5) == pytest.approx(power_two_sample(0.5, k))


def test_power_monotone_in_clusters_and_icc():
    p = [cluster_power(k, 20, 0.05, effect_size=0.3) for k in (4, 8, 16, 32)]
    assert all(a < b for a, b in zip(p, p[1:]))
    q = [cluster_power(10, 20, icc, effect_size=0.3) for icc in (0.0, 0.05, 0.2, 0.5)]
    assert all(a > b for a, b in zip(q, q[1:]))


def test_proportions_sample_size():
    res = cluster_sample_size(50, 0.01, p1=0.10, p2=0.15, kind="proportions", method="normal")
    z = sp.norm.ppf(0.975) + sp.norm.ppf(0.8)
    n_ind = z**2 * (0.1 * 0.9 + 0.15 * 0.85) / 0.05**2
    assert res.n_individual_per_arm == pytest.approx(n_ind)
    assert res.clusters_per_arm == math.ceil(n_ind * 1.49 / 50)
    t = cluster_sample_size(50, 0.01, p1=0.10, p2=0.15, kind="proportions")
    assert t.achieved_power >= 0.8


def test_mde_inverts_power():
    k, m, icc = 15, 20, 0.05
    d = cluster_mde(k, m, icc, method="normal")
    assert cluster_power(k, m, icc, effect_size=d, method="normal") == pytest.approx(0.8, abs=1e-6)
    d_t = cluster_mde(k, m, icc, method="t")
    assert d_t > d
    assert cluster_power(k, m, icc, effect_size=d_t, method="t") == pytest.approx(0.8, abs=0.01)


def test_validation():
    with pytest.raises(ValueError):
        cluster_power(1, 10, 0.1, effect_size=0.5)
    with pytest.raises(ValueError):
        cluster_power(5, 10, 0.1)
    with pytest.raises(ValueError):
        cluster_sample_size(10, 0.1, p1=0.2, p2=0.2, kind="proportions")
    with pytest.raises(ValueError):
        cluster_sample_size(10, 0.1, effect_size=0.5, kind="ratios")
    with pytest.raises(ValueError):
        cluster_sample_size(10, 0.1, effect_size=0.5, power=1.0)
    with pytest.raises(ValueError):
        cluster_power(5, 10, 0.1, effect_size=0.5, method="exact")


def test_estimate_icc_recovers_simulated_value():
    rng = np.random.default_rng(0)
    k, m, icc = 200, 15, 0.2
    u = rng.normal(0, math.sqrt(icc), size=k)
    e = rng.normal(0, math.sqrt(1 - icc), size=(k, m))
    y = (u[:, None] + e).ravel()
    g = np.repeat(np.arange(k), m)
    est = estimate_icc(y, g)
    assert est.icc == pytest.approx(icc, abs=0.04)
    assert est.ci_lower < icc < est.ci_upper
    assert est.n_clusters == k and est.n_obs == k * m
    assert est.m0 == pytest.approx(m)


def test_estimate_icc_balanced_hand_example_and_unbalanced():
    y = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    g = ["a", "a", "a", "b", "b", "b"]
    # MSB = 3 * (2-4)^2 * 2 / 1 = 24, MSW = (2 + 2) / 4 = 1, icc = 23 / (24 + 2) = 23/26
    est = estimate_icc(y, g)
    assert est.msb == pytest.approx(24.0)
    assert est.msw == pytest.approx(1.0)
    assert est.icc == pytest.approx(23 / 26)
    rng = np.random.default_rng(1)
    sizes = rng.integers(3, 30, size=60)
    g = np.repeat(np.arange(60), sizes)
    u = rng.normal(0, 0.5, size=60)
    y = u[g] + rng.normal(0, 1.0, size=g.size)
    est = estimate_icc(y, g)
    assert est.m0 < sizes.mean()
    assert est.icc == pytest.approx(0.25 / 1.25, abs=0.1)
    # independent data => icc near 0, truncated to be non-negative
    est0 = estimate_icc(rng.normal(size=600), np.repeat(np.arange(60), 10))
    assert 0.0 <= est0.icc < 0.08
    with pytest.raises(ValueError):
        estimate_icc([1, 2, 3], [0, 0, 0])
    with pytest.raises(ValueError):
        estimate_icc([1, 2], [0, 0, 1])


def test_cli_cluster_power_modes():
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["cluster-power", "--cluster-size", "20", "--icc", "0.05", "--d", "0.3"]) == 0
    out = buf.getvalue()
    assert "design effect: 1.9500" in out and "clusters per arm" in out
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main([
            "cluster-power", "--mode", "power", "--cluster-size", "20", "--icc", "0.05",
            "--p1", "0.1", "--p2", "0.15", "--clusters", "30",
        ]) == 0
    assert "power:" in buf.getvalue()
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["cluster-power", "--mode", "mde", "--cluster-size", "20", "--icc", "0.05", "--clusters", "10"]) == 0
    assert "MDE" in buf.getvalue()
    err = io.StringIO()
    with redirect_stderr(err):
        assert main(["cluster-power", "--mode", "power", "--cluster-size", "20", "--icc", "0.05", "--d", "0.3"]) == 2
    assert "--clusters" in err.getvalue()
