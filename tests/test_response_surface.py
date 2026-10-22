"""Tests for Plackett-Burman screening and response-surface designs."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiment_design_kit import (
    PLACKETT_BURMAN_GENERATORS,
    box_behnken,
    central_composite,
    fit_quadratic,
    plackett_burman,
    quadratic_terms,
    scaled_prediction_variance,
    screening_effects,
)
from experiment_design_kit.cli import main

BBD_Y = [69, 75, 73, 80, 62, 70, 68, 76, 61, 72, 66, 74, 82, 81, 83]


@pytest.mark.parametrize("n", [4, 8, 12, 16, 20, 24, 32, 40, 44, 48])
def test_plackett_burman_is_orthogonal_and_balanced(n):
    from experiment_design_kit.response_surface import _hadamard

    H = _hadamard(n)
    np.testing.assert_allclose(H.T @ H, n * np.eye(n))
    k = min(n - 1, 25)  # factor letters A..Z without I
    d = plackett_burman(k, n)
    X = d.matrix
    assert X.shape == (n, k)
    assert set(np.unique(X)) == {-1.0, 1.0}
    np.testing.assert_allclose(X.T @ X, n * np.eye(k))
    np.testing.assert_allclose(X.sum(axis=0), 0.0)


def test_plackett_burman_12_uses_published_generator():
    d = plackett_burman(11)
    assert d.n_runs == 12
    first = "".join("+" if v > 0 else "-" for v in d.matrix[0])
    assert first == PLACKETT_BURMAN_GENERATORS[12]
    # cyclic: row i+1 is row i shifted right by one
    np.testing.assert_allclose(d.matrix[1], np.roll(d.matrix[0], 1))
    np.testing.assert_allclose(d.matrix[-1], -np.ones(11))


def test_plackett_burman_default_size_and_errors():
    assert plackett_burman(3).n_runs == 4
    assert plackett_burman(4).n_runs == 8
    assert plackett_burman(7).n_runs == 8
    assert plackett_burman(8).n_runs == 12
    assert plackett_burman(5, 12).matrix.shape == (12, 5)
    with pytest.raises(ValueError, match="multiple of 4"):
        plackett_burman(5, 10)
    with pytest.raises(ValueError, match="no Plackett-Burman"):
        plackett_burman(20, 28)
    with pytest.raises(ValueError):
        plackett_burman(0)


def test_screening_recovers_active_factors():
    rng = np.random.default_rng(3)
    d = plackett_burman(11)
    y = 50 + 4.0 * d.matrix[:, 0] - 3.0 * d.matrix[:, 4] + rng.normal(0, 0.2, 12)
    res = screening_effects(d, y)
    assert res.method == "lenth"
    assert set(res.significant) == {"A", "E"}
    assert res.effects["A"] == pytest.approx(8.0, abs=0.5)
    assert res.effects["E"] == pytest.approx(-6.0, abs=0.5)
    # Unsaturated: t tests on the residual
    d7 = plackett_burman(7, 12)
    y7 = 10 + 2 * d7.matrix[:, 1] + rng.normal(0, 0.1, 12)
    res7 = screening_effects(d7, y7)
    assert res7.method == "t-test" and res7.df == 4
    assert res7.significant == ("B",)


def test_ccd_rotatable_alpha_and_layout():
    d = central_composite(3)
    assert d.alpha == pytest.approx(8 ** 0.25)
    assert d.n_runs == 8 + 6 + 8
    assert d.point_types.count("axial") == 6 and d.point_types.count("center") == 8
    # rotatable: scaled prediction variance depends only on radius
    r = 1.2
    pts = [[r, 0, 0], [0, 0, -r], [r / np.sqrt(3)] * 3, [r / np.sqrt(2), -r / np.sqrt(2), 0]]
    v = scaled_prediction_variance(d, pts)
    np.testing.assert_allclose(v, v[0], rtol=1e-10)


def test_ccd_orthogonal_alpha_makes_quadratic_columns_orthogonal():
    for k, center in [(2, (4, 4)), (3, (2, 2)), (4, (3, 3))]:
        d = central_composite(k, alpha="orthogonal", center=center)
        Q = d.matrix ** 2
        Qc = Q - Q.mean(axis=0)
        G = Qc.T @ Qc
        np.testing.assert_allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-9)


def test_ccd_variants():
    faced = central_composite(3, face="faced")
    assert faced.alpha == 1.0 and np.abs(faced.matrix).max() == 1.0
    inscribed = central_composite(3, face="inscribed")
    assert np.abs(inscribed.matrix).max() == pytest.approx(1.0)
    assert np.abs(inscribed.matrix[:8]).max() == pytest.approx(1 / 8 ** 0.25)
    frac = central_composite(5, fraction=1, center=(1, 1))
    assert frac.n_factorial == 16 and frac.n_runs == 16 + 10 + 2
    custom = central_composite(2, alpha=1.5, center=(0, 1))
    assert custom.alpha == 1.5 and custom.n_runs == 9
    with pytest.raises(ValueError):
        central_composite(1)
    with pytest.raises(ValueError):
        central_composite(2, alpha="bogus")
    with pytest.raises(ValueError):
        central_composite(2, alpha=-1.0)
    with pytest.raises(ValueError):
        central_composite(2, face="spherical")


def test_box_behnken_structure():
    for k, edges in [(3, 12), (4, 24), (5, 40)]:
        d = box_behnken(k, center=3)
        assert d.n_runs == edges + 3
        body = d.matrix[:edges]
        assert np.all((body != 0).sum(axis=1) == 2)  # exactly two factors at ±1
        assert not np.any(np.all(np.abs(d.matrix) == 1, axis=1))  # no corners
        np.testing.assert_allclose(d.matrix.sum(axis=0), 0.0)
    with pytest.raises(ValueError):
        box_behnken(2)


def test_fit_quadratic_recovers_exact_surface_and_optimum():
    d = box_behnken(3)
    x = d.matrix

    def f(x):
        return (80 - 2 * (x[:, 0] - 0.3) ** 2 - 3 * (x[:, 1] + 0.2) ** 2
                - 1.5 * x[:, 2] ** 2 + x[:, 0] * x[:, 1])

    fit = fit_quadratic(d, f(x))
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.coefficients["AB"] == pytest.approx(1.0)
    assert fit.coefficients["C^2"] == pytest.approx(-1.5)
    # gradient = 0: -4(x0-.3)+x1 = 0, -6(x1+.2)+x0 = 0
    x1 = -0.9 / 5.75
    x0 = 0.3 + x1 / 4
    np.testing.assert_allclose(fit.stationary_point, [x0, x1, 0.0], atol=1e-10)
    assert fit.nature == "maximum"
    assert fit.stationary_response == pytest.approx(float(f(np.array([[x0, x1, 0.0]]))[0]))
    np.testing.assert_allclose(fit.predict(x), f(x), atol=1e-10)


def test_fit_quadratic_matches_least_squares_reference():
    d = box_behnken(3)
    fit = fit_quadratic(d, BBD_Y)
    x = d.matrix
    cols = [np.ones(15), *x.T, *(x.T ** 2)] + [x[:, i] * x[:, j] for i, j in itertools.combinations(range(3), 2)]
    beta = np.linalg.lstsq(np.column_stack(cols), BBD_Y, rcond=None)[0]
    np.testing.assert_allclose(list(fit.coefficients.values()), beta, atol=1e-10)
    # statsmodels OLS on the same model: R^2 = 0.986987, adj R^2 = 0.963563
    assert fit.r_squared == pytest.approx(0.986987, abs=1e-5)
    assert fit.adj_r_squared == pytest.approx(0.963563, abs=1e-5)
    assert fit.resid_df == 5
    assert list(fit.coefficients) == quadratic_terms("ABC")


def test_saddle_ridge_and_validation():
    d = central_composite(2)
    x = d.matrix
    saddle = fit_quadratic(d, x[:, 0] ** 2 - x[:, 1] ** 2 + 0.5 * x[:, 0])
    assert saddle.nature == "saddle"
    ridge = fit_quadratic(d, -x[:, 0] ** 2 + 0.3 * x[:, 1])
    assert ridge.nature == "ridge" and ridge.stationary_point is None
    with pytest.raises(ValueError, match="values"):
        fit_quadratic(d, [1.0, 2.0])
    with pytest.raises(ValueError, match="rank"):
        fit_quadratic(plackett_burman(3).matrix, [1.0, 2.0, 3.0, 4.0])
    arr = fit_quadratic(x, x[:, 0] ** 2, factors=["temp", "time"])
    assert "temp^2" in arr.coefficients


def test_cli_screening_and_response_surface(capsys):
    assert main(["screening", "--factors", "7", "--responses", "10,12,9,15,11,14,8,13"]) == 0
    out = capsys.readouterr().out
    assert "Plackett-Burman  runs: 8" in out and "Lenth PSE" in out
    assert main(["response-surface", "--factors", "3", "--design", "bbd",
                 "--responses", ",".join(map(str, BBD_Y))]) == 0
    out = capsys.readouterr().out
    assert "stationary point: maximum" in out
    assert main(["response-surface", "--factors", "2", "--alpha", "orthogonal"]) == 0
    assert "alpha: 1.4142" in capsys.readouterr().out
    assert main(["screening", "--factors", "5", "--runs", "10"]) == 2
