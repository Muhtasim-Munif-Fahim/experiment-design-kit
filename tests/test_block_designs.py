import itertools

import numpy as np
import pytest

from experiment_design_kit.block_designs import (
    are_orthogonal,
    balanced_incomplete_block_design,
    bibd_intrablock_analysis,
    bibd_parameters,
    graeco_latin_square,
    is_latin_square,
    latin_square,
    latin_square_anova,
    mutually_orthogonal_latin_squares,
)
from experiment_design_kit.cli import main


def _dummies(codes, n):
    D = np.zeros((codes.size, n))
    D[np.arange(codes.size), codes] = 1
    return D[:, 1:]


def _rss(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.sum((y - X @ beta) ** 2))


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_latin_square_valid_and_randomised(n):
    L = latin_square(n, seed=0)
    assert is_latin_square(L)
    assert np.array_equal(latin_square(n, seed=3), latin_square(n, seed=3))
    if n == 9:
        assert not np.array_equal(latin_square(n, seed=1), latin_square(n, seed=2))
    assert is_latin_square(latin_square(n, randomize=False))


def test_is_latin_and_orthogonal_negative_cases():
    assert not is_latin_square([[0, 1], [0, 1]])
    assert not is_latin_square(np.zeros((2, 3)))
    A = latin_square(4, randomize=False)
    assert not are_orthogonal(A, A)


@pytest.mark.parametrize("q", [3, 4, 5, 7, 8, 9, 16, 25, 27])
def test_complete_sets_of_mols(q):
    squares = mutually_orthogonal_latin_squares(q)
    assert len(squares) == q - 1
    assert all(is_latin_square(s) for s in squares)
    assert all(are_orthogonal(a, b) for a, b in itertools.combinations(squares, 2))


def test_graeco_latin_square_and_impossible_orders():
    A, B = graeco_latin_square(5, seed=4)
    assert are_orthogonal(A, B)
    for n in (2, 6):
        with pytest.raises(ValueError, match="no Graeco-Latin"):
            graeco_latin_square(n)
    with pytest.raises(ValueError, match="prime power"):
        mutually_orthogonal_latin_squares(10)


def test_latin_square_anova_matches_regression():
    rng = np.random.default_rng(0)
    n = 5
    L = latin_square(n, seed=1)
    effects = np.array([0.0, 1.0, 2.0, -1.0, 0.5])
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    Y = 10 + rng.normal(size=n)[:, None] + rng.normal(size=n)[None, :] + effects[L] + rng.normal(0, 0.3, (n, n))
    res = latin_square_anova(Y, L)
    y = Y.ravel()
    one = np.ones((n * n, 1))
    R, C, T = _dummies(rows.ravel(), n), _dummies(cols.ravel(), n), _dummies(L.ravel(), n)
    rss_full = _rss(np.hstack([one, R, C, T]), y)
    rss_noT = _rss(np.hstack([one, R, C]), y)
    assert res.table["error"]["SS"] == pytest.approx(rss_full)
    assert res.table["treatment"]["SS"] == pytest.approx(rss_noT - rss_full)
    assert res.residual_df == 12
    tab = res.table
    assert tab["total"]["SS"] == pytest.approx(sum(tab[s]["SS"] for s in ("rows", "columns", "treatment", "error")))
    assert res.p_value < 1e-4
    np.testing.assert_allclose(res.treatment_means - res.treatment_means.mean(), effects - effects.mean(), atol=0.4)
    assert "treatment" in res.summary()


def test_latin_square_anova_null_p_values_roughly_uniform():
    rng = np.random.default_rng(1)
    ps = [latin_square_anova(rng.normal(size=(4, 4)), latin_square(4, seed=s)).p_value for s in range(400)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09


def test_latin_square_anova_validation():
    with pytest.raises(ValueError, match="Latin"):
        latin_square_anova(np.zeros((3, 3)), np.zeros((3, 3), dtype=int))
    with pytest.raises(ValueError, match="matching"):
        latin_square_anova(np.zeros((3, 3)), latin_square(4))


@pytest.mark.parametrize(
    "v,k,b,r,lam",
    [(9, 3, 12, 4, 1), (7, 3, 7, 3, 1), (13, 4, 13, 4, 1), (7, 4, 7, 4, 2), (16, 4, 20, 5, 1),
     (21, 5, 21, 5, 1), (6, 3, 20, 10, 4), (5, 2, 10, 4, 1)],
)
def test_bibd_constructions(v, k, b, r, lam):
    d = balanced_incomplete_block_design(v, k)
    assert (d.v, d.b, d.r, d.k, d.lam) == (v, b, r, k, lam)
    # basic BIBD identities
    assert b * k == v * r and lam * (v - 1) == r * (k - 1) and b >= v
    N = d.incidence()
    np.testing.assert_array_equal(N @ N.T, (r - lam) * np.eye(v, dtype=int) + lam)
    assert d.efficiency == pytest.approx(lam * v / (r * k))
    rd = d.randomized(seed=0)
    assert bibd_parameters(rd.blocks, v) == bibd_parameters(d.blocks, v)


def test_bibd_errors():
    with pytest.raises(ValueError, match="no construction"):
        balanced_incomplete_block_design(30, 7)
    with pytest.raises(ValueError):
        balanced_incomplete_block_design(4, 4)
    with pytest.raises(ValueError, match="pairs"):
        bibd_parameters([(0, 1), (0, 1), (2, 3), (2, 3)])
    with pytest.raises(ValueError, match="same size"):
        bibd_parameters([(0, 1), (1, 2, 0)])


def test_intrablock_analysis_matches_least_squares():
    rng = np.random.default_rng(2)
    d = balanced_incomplete_block_design(7, 3)
    tau = np.array([0, 1.5, -1, 0.5, 2, -0.5, 0.0])
    tau = tau - tau.mean()
    beta = rng.normal(0, 3, d.b)
    y = [[5 + beta[j] + tau[t] + rng.normal(0, 0.2) for t in blk] for j, blk in enumerate(d.blocks)]
    res = bibd_intrablock_analysis(d.blocks, y)
    # least squares with block + treatment dummies
    trt = np.array([t for blk in d.blocks for t in blk])
    blk = np.repeat(np.arange(d.b), d.k)
    yy = np.concatenate([np.asarray(r) for r in y])
    one = np.ones((yy.size, 1))
    X = np.hstack([one, _dummies(blk, d.b), _dummies(trt, 7)])
    rss_full = _rss(X, yy)
    rss_blocks = _rss(np.hstack([one, _dummies(blk, d.b)]), yy)
    assert res.table["error"]["SS"] == pytest.approx(rss_full)
    assert res.table["treatment (adj)"]["SS"] == pytest.approx(rss_blocks - rss_full)
    coef = np.linalg.lstsq(X, yy, rcond=None)[0][-6:]
    ls_tau = np.r_[0.0, coef]
    np.testing.assert_allclose(res.effects, ls_tau - ls_tau.mean(), atol=1e-8)
    np.testing.assert_allclose(res.effects, tau, atol=0.4)
    assert res.effects.sum() == pytest.approx(0.0, abs=1e-9)
    assert res.p_value < 1e-3
    assert res.efficiency == pytest.approx(7 / 9)
    assert res.se_difference > 0


def test_cli_block_design(capsys):
    assert main(["block-design", "latin", "--n", "4", "--seed", "1"]) == 0
    assert "Latin square" in capsys.readouterr().out
    assert main(["block-design", "graeco-latin", "--n", "5"]) == 0
    assert "Graeco-Latin" in capsys.readouterr().out
    assert main(["block-design", "bibd", "--v", "7", "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert "lambda=1" in out and "projective plane" in out
    assert main(["block-design", "graeco-latin", "--n", "6"]) == 2
