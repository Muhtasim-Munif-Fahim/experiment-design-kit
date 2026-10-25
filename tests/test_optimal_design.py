import itertools

import numpy as np
import pytest

from experiment_design_kit import (
    candidate_grid,
    centered_l2_discrepancy,
    d_efficiency,
    d_optimal_design,
    latin_hypercube,
    maximin_distance,
    maximin_latin_hypercube,
    model_matrix,
    phi_p,
)
from experiment_design_kit.cli import main


@pytest.mark.parametrize("centered", [False, True])
def test_latin_hypercube_one_point_per_stratum(centered):
    x = latin_hypercube(17, 4, seed=3, centered=centered)
    assert x.shape == (17, 4)
    assert np.all((x >= 0) & (x < 1))
    for col in x.T:
        assert sorted(np.floor(col * 17).astype(int)) == list(range(17))
    if centered:
        assert np.allclose((x * 17) % 1, 0.5)


def test_latin_hypercube_reproducible_and_validates():
    assert np.array_equal(latin_hypercube(5, 2, seed=1), latin_hypercube(5, 2, seed=1))
    with pytest.raises(ValueError):
        latin_hypercube(1, 2)
    with pytest.raises(ValueError):
        latin_hypercube(5, 0)


def test_centered_l2_discrepancy_matches_scipy():
    qmc = pytest.importorskip("scipy.stats.qmc")
    rng = np.random.default_rng(0)
    for n, k in [(10, 1), (25, 3), (40, 5)]:
        x = rng.random((n, k))
        assert centered_l2_discrepancy(x) == pytest.approx(qmc.discrepancy(x, method="CD"), rel=1e-10)
    with pytest.raises(ValueError):
        centered_l2_discrepancy([[1.5, 0.2]])


def test_maximin_distance_and_phi_p():
    sq = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert maximin_distance(sq) == pytest.approx(1.0)
    # phi_p -> 1 / min distance as p grows
    assert phi_p(sq, p=200) == pytest.approx(1.0, rel=0.02)
    assert phi_p([[0, 0], [0, 0]]) == float("inf")


def test_maximin_lhs_improves_spacing_and_stays_latin():
    n, k = 20, 3
    start = latin_hypercube(n, k, seed=11, centered=True)
    opt = maximin_latin_hypercube(n, k, seed=5, iterations=3000)
    for col in opt.T:
        assert sorted(np.floor(col * n).astype(int)) == list(range(n))
    assert maximin_distance(opt) > maximin_distance(start)
    assert phi_p(opt) < phi_p(start)


def test_candidate_grid_and_model_matrix_columns():
    g = candidate_grid(2, 3)
    assert g.shape == (9, 2)
    assert set(g.ravel()) == {-1.0, 0.0, 1.0}
    assert model_matrix(g, "linear").shape == (9, 3)
    assert model_matrix(g, "interaction").shape == (9, 4)
    m = model_matrix(g, "quadratic")
    assert m.shape == (9, 6)
    assert np.allclose(m[:, 3], g[:, 0] ** 2)
    assert np.allclose(m[:, 5], g[:, 0] * g[:, 1])
    with pytest.raises(ValueError):
        model_matrix(g, "cubic")


def test_d_optimal_two_level_recovers_full_factorial():
    d = d_optimal_design(8, candidate_grid(3, 2), model="interaction", allow_repeats=False, seed=0)
    assert d.indices == tuple(range(8))
    assert d.d_efficiency == pytest.approx(100.0)
    assert d_efficiency(candidate_grid(3, 2), "linear") == pytest.approx(100.0)


def test_d_optimal_matches_exhaustive_search():
    cand = candidate_grid(2, 3)
    f = model_matrix(cand, "quadratic")
    for n in (6, 7):
        best = max(np.linalg.slogdet(f[list(s)].T @ f[list(s)])[1]
                   for s in itertools.combinations(range(len(cand)), n))
        d = d_optimal_design(n, cand, model="quadratic", allow_repeats=False, seed=1)
        assert d.log_det == pytest.approx(best)
        assert len(set(d.indices)) == n


def test_d_optimal_beats_random_and_constraints_respected():
    cand = candidate_grid(3, 3)
    feasible = cand[cand.sum(axis=1) <= 1.0]  # drop a corner region
    d = d_optimal_design(12, feasible, model="quadratic", seed=2)
    assert d.n_runs == 12
    assert np.all(d.matrix.sum(axis=1) <= 1.0)
    rng = np.random.default_rng(0)
    f = model_matrix(feasible, "quadratic")
    for _ in range(50):
        idx = rng.choice(len(feasible), 12, replace=True)
        sign, ld = np.linalg.slogdet(f[idx].T @ f[idx])
        assert sign <= 0 or ld <= d.log_det + 1e-9


def test_d_optimal_validation():
    with pytest.raises(ValueError, match="model terms"):
        d_optimal_design(5, k=2, model="quadratic")
    with pytest.raises(ValueError, match="candidates or k"):
        d_optimal_design(6)
    with pytest.raises(ValueError, match="without repeats"):
        d_optimal_design(10, candidate_grid(2, 3), model="quadratic", allow_repeats=False)
    with pytest.raises(ValueError, match="not estimable"):
        d_optimal_design(6, candidate_grid(2, 2), model="quadratic")
    assert d_efficiency([[0, 0], [0, 0], [0, 0]], "linear") == 0.0


def test_cli_optimal_design(capsys):
    assert main(["optimal-design", "--factors", "3", "--runs", "10", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "D-efficiency" in out and "design: d-optimal" in out
    assert main(["optimal-design", "--factors", "2", "--runs", "12", "--kind", "maximin-lhs",
                 "--seed", "1", "--iterations", "200"]) == 0
    assert "centered L2 discrepancy" in capsys.readouterr().out
    assert main(["optimal-design", "--factors", "2", "--runs", "3", "--model", "quadratic"]) == 2
