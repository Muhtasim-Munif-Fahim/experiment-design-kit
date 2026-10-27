"""Tests for mixture designs and Scheffé models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiment_design_kit import (
    best_blend,
    cox_effects,
    extreme_vertices,
    fit_scheffe,
    from_pseudocomponents,
    scheffe_model_matrix,
    scheffe_terms,
    simplex_centroid,
    simplex_lattice,
    to_pseudocomponents,
)
from experiment_design_kit.cli import main

# Cornell yarn-elongation data (Montgomery, Design and Analysis of Experiments, Ex. 11.5)
YARN_X = (
    [[1, 0, 0]] * 2 + [[0.5, 0.5, 0]] * 3 + [[0, 1, 0]] * 2
    + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 2 + [[0.5, 0, 0.5]] * 3
)
YARN_Y = [11.0, 12.4, 15.0, 14.8, 16.1, 8.8, 10.0, 10.0, 9.7, 11.8, 16.8, 16.0, 17.7, 16.4, 16.6]


@pytest.mark.parametrize("q, m", [(2, 1), (3, 2), (3, 3), (4, 2), (5, 3)])
def test_simplex_lattice_size_and_constraint(q, m):
    d = simplex_lattice(q, m)
    assert d.n_runs == math.comb(q + m - 1, m)
    np.testing.assert_allclose(d.matrix.sum(axis=1), 1.0)
    assert np.all(d.matrix >= 0)
    assert np.allclose(np.round(d.matrix * m), d.matrix * m)
    assert len({tuple(r) for r in d.matrix}) == d.n_runs
    assert d.point_types.count("vertex") == q


def test_simplex_centroid_and_augmentation():
    d = simplex_centroid(4)
    assert d.n_runs == 15 and d.point_types[-1] == "centroid"
    np.testing.assert_allclose(d.matrix[-1], 0.25)
    a = simplex_centroid(3, augment=True)
    assert a.n_runs == 7 + 3
    np.testing.assert_allclose(a.matrix[-3], [4 / 6, 1 / 6, 1 / 6])
    lat = simplex_lattice(3, 2, augment=True)
    assert lat.n_runs == 6 + 1 + 3 and lat.point_types.count("axial") == 3
    lat3 = simplex_lattice(3, 3, augment=True)  # {3,3} already has the centroid
    assert lat3.n_runs == 10 + 3


def test_extreme_vertices_hexagon():
    d = extreme_vertices([0.1, 0.1, 0.1], [0.6, 0.6, 0.6])
    verts = d.matrix[[t == "vertex" for t in d.point_types]]
    assert len(verts) == 6
    assert {tuple(np.round(sorted(v), 6)) for v in verts} == {(0.1, 0.3, 0.6)}
    assert d.point_types.count("edge-centroid") == 6
    np.testing.assert_allclose(d.matrix[-1], 1 / 3)
    np.testing.assert_allclose(d.matrix.sum(axis=1), 1.0)
    assert np.all(d.matrix >= 0.1 - 1e-12) and np.all(d.matrix <= 0.6 + 1e-12)


def test_extreme_vertices_lower_bounds_only_is_simplex():
    lo = [0.2, 0.1, 0.3]
    d = extreme_vertices(lo, centroids=False)
    assert d.n_runs == 3
    z = to_pseudocomponents(d.matrix, lo)
    np.testing.assert_allclose(np.sort(z, axis=0), np.sort(np.eye(3), axis=0), atol=1e-12)
    np.testing.assert_allclose(from_pseudocomponents(z, lo), d.matrix)
    with pytest.raises(ValueError):
        extreme_vertices([0.5, 0.6])
    with pytest.raises(ValueError):
        extreme_vertices(upper=[0.3, 0.3, 0.3])
    with pytest.raises(ValueError):
        to_pseudocomponents([[0.5, 0.5]], [0.5, 0.5])


def test_scheffe_terms_and_matrix():
    names = ["A", "B", "C"]
    assert scheffe_terms(names, "linear") == names
    assert len(scheffe_terms(names, "quadratic")) == 6
    assert scheffe_terms(names, "special_cubic")[-1] == "A*B*C"
    assert len(scheffe_terms(names, "cubic")) == 10
    X = scheffe_model_matrix([[0.5, 0.25, 0.25]], "cubic")
    assert X.shape == (1, 10)
    assert X[0, 3] == pytest.approx(0.125) and X[0, 6] == pytest.approx(0.5 * 0.25 * 0.25)
    with pytest.raises(ValueError):
        scheffe_terms(names, "quartic")


def test_fit_scheffe_yarn_example():
    fit = fit_scheffe(np.array(YARN_X), YARN_Y, model="quadratic")
    expected = [11.7, 9.4, 16.4, 19.0, 11.4, -9.6]
    reordered = [fit.coefficients[t] for t in ["x1", "x2", "x3", "x1*x2", "x1*x3", "x2*x3"]]
    np.testing.assert_allclose(reordered, expected, atol=1e-9)
    assert fit.resid_df == 9
    assert 0.9 < fit.r_squared < 1 and fit.adj_r_squared < fit.r_squared
    assert all(se > 0 for se in fit.std_errors.values())
    blend, best = best_blend(fit, resolution=40)
    assert blend[1] < 0.15 and blend[2] > blend[1]  # optimum is mostly x1/x3
    assert best >= fit.predict([[0.5, 0, 0.5]])[0] - 1e-9


def test_exact_recovery_and_rank_deficiency():
    d = simplex_centroid(3)
    true = np.array([2.0, 3.0, 1.0, 4.0, -2.0, 1.5, 30.0])
    y = scheffe_model_matrix(d.matrix, "special_cubic") @ true
    fit = fit_scheffe(d, y, model="special_cubic")
    np.testing.assert_allclose(fit.beta, true, atol=1e-9)
    assert fit.resid_df == 0 and math.isnan(fit.rmse)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_scheffe(simplex_lattice(3, 1), [1, 2, 3], model="quadratic")
    with pytest.raises(ValueError):
        fit_scheffe(np.array([[0.5, 0.6]]), [1.0], model="linear")
    with pytest.raises(ValueError):
        fit_scheffe(d, y[:-1])


def test_cox_effects_linear_model():
    d = simplex_lattice(3, 1)
    fit = fit_scheffe(d, [10.0, 4.0, 1.0], model="linear")
    eff = cox_effects(fit)
    # along Cox direction from centroid, x_i=0 gives the average of the other two
    assert eff["x1"]["total_effect"] == pytest.approx(10 - 2.5)
    assert eff["x3"]["total_effect"] == pytest.approx(1 - 7.0)
    assert np.all(np.diff(eff["x1"]["response"]) > 0)
    np.testing.assert_allclose(eff["x2"]["response"][0], 5.5)
    with pytest.raises(ValueError):
        cox_effects(fit, reference=[1, 0, 0])
    blend, best = best_blend(fit, upper=[0.5, 1, 1], resolution=10)
    np.testing.assert_allclose(blend, [0.5, 0.5, 0.0])
    assert best == pytest.approx(7.0)


def test_cli_mixture(capsys):
    y = ",".join(str(v) for v in [11.7, 9.4, 16.4, 15.3, 16.9, 10.5])
    assert main(["mixture", "--components", "3", "--degree", "2", "--responses", y]) == 0
    out = capsys.readouterr().out
    assert "simplex-lattice {3,2}" in out and "x1*x2: +19.0000" in out and "best blend (max)" in out
    assert main(["mixture", "--design", "extreme-vertices", "--lower", "0.1,0.1,0.1",
                 "--upper", "0.6,0.6,0.6"]) == 0
    out = capsys.readouterr().out
    assert "runs: 13" in out and "edge-centroid" in out
    assert main(["mixture", "--design", "centroid"]) == 2
    assert "--components is required" in capsys.readouterr().err
