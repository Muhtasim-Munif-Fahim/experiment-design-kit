"""Tests for two-level full / fractional factorial designs and effect analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiment_design_kit import (
    MINIMUM_ABERRATION_GENERATORS,
    alias_structure,
    defining_relation,
    factorial_effects,
    fold_over,
    fractional_factorial,
    full_factorial,
    lenth_pse,
    main,
)

# Montgomery, Design and Analysis of Experiments, Example 6.2 (filtration rate,
# unreplicated 2^4, standard order).
FILTRATION = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]
FILTRATION_EFFECTS = {
    "A": 21.625, "B": 3.125, "C": 9.875, "D": 14.625,
    "AB": 0.125, "AC": -18.125, "AD": 16.625, "BC": 2.375, "BD": -0.375, "CD": -1.125,
    "ABC": 1.875, "ABD": 4.125, "ACD": -1.625, "BCD": -2.625, "ABCD": 1.375,
}


def _column(design, word):
    col = np.ones(design.n_runs, dtype=int)
    for ch in word:
        col = col * design.matrix[:, design.factors.index(ch)]
    return col


def test_full_factorial_standard_order_and_orthogonality():
    d = full_factorial(3)
    assert d.factors == ("A", "B", "C")
    assert d.matrix.tolist() == [
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
    ]
    d5 = full_factorial(5)
    X = np.column_stack([_column(d5, "".join(c)) for r in range(1, 6)
                         for c in itertools.combinations("ABCDE", r)])
    assert np.array_equal(X.T @ X, 32 * np.eye(31, dtype=int))
    assert d5.resolution is None and d5.resolution_roman == "full"


def test_effects_reproduce_montgomery_filtration_example():
    result = factorial_effects(full_factorial(4), FILTRATION)
    assert result.method == "lenth"
    for name, value in FILTRATION_EFFECTS.items():
        assert result.effects[name] == pytest.approx(value)
    assert result.intercept == pytest.approx(np.mean(FILTRATION))
    # Lenth (1989): s0 = 1.5 * median |c| = 3.1875 -> PSE = 1.5 * 1.75 = 2.625
    assert result.std_error == pytest.approx(2.625)
    assert result.df == pytest.approx(5.0)
    assert result.margin_of_error == pytest.approx(2.570582 * 2.625, rel=1e-5)
    assert result.simultaneous_margin > result.margin_of_error
    assert set(result.significant) == {"A", "C", "D", "AC", "AD"}
    assert [name for name, _ in result.ranked()[:3]] == ["A", "AC", "AD"]
    scores = result.half_normal_scores()
    assert scores[-1][0] == "A" and scores[0][1] <= scores[-1][1]
    assert all(b[2] > a[2] for a, b in zip(scores, scores[1:]))


def test_effect_equals_contrast_mean_difference():
    rng = np.random.default_rng(0)
    d = full_factorial(3)
    y = rng.normal(size=8)
    result = factorial_effects(d, y)
    for name, value in result.effects.items():
        col = _column(d, name)
        assert value == pytest.approx(y[col > 0].mean() - y[col < 0].mean())


def test_replicated_design_uses_t_tests():
    rng = np.random.default_rng(1)
    d = full_factorial(3)
    true = 10 + 3 * d.matrix[:, 0] / 2 + 0 * d.matrix[:, 1]  # A effect = 3
    y = true[:, None] + rng.normal(scale=0.5, size=(8, 4))
    result = factorial_effects(d, y)
    assert result.method == "t-test"
    assert result.df == pytest.approx(32 - 8)
    resid = y - y.mean(axis=1, keepdims=True)
    s2 = (resid**2).sum() / 24
    assert result.std_error == pytest.approx(2 * np.sqrt(s2 / 32))
    assert result.effects["A"] == pytest.approx(3, abs=0.5)
    assert "A" in result.significant and result.p_values["A"] < 1e-6
    assert result.p_values["B"] > 0.01
    limited = factorial_effects(d, y, max_order=1)
    assert set(limited.effects) == {"A", "B", "C"}
    assert limited.df == pytest.approx(32 - 4)


def test_fractional_generators_defining_relation_and_aliases():
    d = fractional_factorial(4, ["D=ABC"])
    assert d.n_runs == 8 and d.resolution == 4 and d.resolution_roman == "IV"
    assert d.defining_words == ("ABCD",)
    assert np.all(_column(d, "ABCD") == 1)
    assert d.aliases["A"] == ("BCD",) and d.aliases["AB"] == ("CD",)
    neg = fractional_factorial(4, ["D=-ABC"])
    assert neg.defining_words == ("-ABCD",)
    assert np.all(_column(neg, "ABCD") == -1)
    assert neg.aliases["A"] == ("-BCD",)

    d62 = fractional_factorial(6, ["E=ABC", "F=BCD"])
    assert d62.defining_words == ("ABCE", "ADEF", "BCDF")
    assert d62.word_length_pattern == (0, 3, 0, 0)
    for word in d62.defining_words:
        assert np.all(_column(d62, word) == 1)
    assert set(d62.aliases["AB"]) == {"CE", "BDEF", "ACDF"}


def test_every_word_in_defining_relation_is_constant_and_aliases_are_confounded():
    for (k, p), gens in MINIMUM_ABERRATION_GENERATORS.items():
        d = fractional_factorial(k, p=p)
        assert d.n_runs == 2 ** (k - p)
        assert len(d.defining_words) == 2**p - 1
        assert d.generators == gens
        for word in d.defining_words:
            sign = -1 if word.startswith("-") else 1
            assert np.all(_column(d, word.lstrip("-")) == sign)
        for effect, aliases in d.aliases.items():
            for alias in aliases:
                sign = -1 if alias.startswith("-") else 1
                assert np.array_equal(_column(d, effect), sign * _column(d, alias.lstrip("-")))
        assert len({tuple(r) for r in d.matrix}) == d.n_runs


def test_minimum_aberration_table_resolutions():
    expected = {(3, 1): 3, (4, 1): 4, (5, 1): 5, (5, 2): 3, (6, 1): 6, (6, 2): 4,
                (6, 3): 3, (7, 1): 7, (7, 2): 4, (7, 3): 4, (7, 4): 3, (8, 2): 5,
                (8, 3): 4, (8, 4): 4}
    for key, res in expected.items():
        assert fractional_factorial(*key[:1], p=key[1]).resolution == res
    # minimum aberration: 2^(7-2) with F=ABCD, G=ABDE has one length-4 word
    assert fractional_factorial(7, p=2).word_length_pattern[:2] == (0, 1)


def test_fold_over_dealiases_resolution_iii():
    base = fractional_factorial(7, p=4)
    assert base.resolution == 3
    assert abs(_column(base, "A") @ _column(base, "BD")) == 8  # A fully aliased with BD
    folded = fold_over(base)
    assert folded.n_runs == 16 and folded.resolution == 4
    assert len(folded.defining_words) == 7
    for f in folded.factors:
        for pair in itertools.combinations(folded.factors, 2):
            assert _column(folded, f) @ _column(folded, "".join(pair)) == 0
    single = fold_over(base, factor="A")
    for pair in itertools.combinations(single.factors, 2):
        if "A" in pair:
            other = [g for g in single.factors if g not in pair]
            for g in other:
                assert _column(single, "".join(pair)) @ _column(single, g) == 0
    assert all(w.lstrip("-").count("A") == 0 for w in single.defining_words)


def test_screening_recovers_active_factors_in_a_resolution_iv_fraction():
    rng = np.random.default_rng(5)
    d = fractional_factorial(6, p=2)
    y = 50 + 4 * _column(d, "A") + 3 * _column(d, "D") + rng.normal(scale=0.3, size=d.n_runs)
    result = factorial_effects(d, y)
    assert result.method == "lenth"
    assert set(result.significant) >= {"A", "D"}
    assert result.effects["A"] == pytest.approx(8, abs=0.6)
    assert len(result.effects) == 15
    assert "AB" in result.aliases


def test_to_levels_and_randomized_order():
    d = fractional_factorial(3, ["C=AB"])
    rows = d.to_levels({"A": ("old", "new"), "B": (10, 20), "C": ("x", "y")})
    assert rows[0] == {"A": "old", "B": 10, "C": "y"}
    order = d.randomized_order(seed=1)
    assert sorted(order.tolist()) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="missing"):
        d.to_levels({"A": (0, 1)})


def test_lenth_pse_and_helpers():
    assert lenth_pse([1.0, -1.0, 1.0, 20.0]) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="two"):
        lenth_pse([1.0])
    assert defining_relation(["D=AB", "E=AC"], 5) == ["ABD", "ACE", "BCDE"]
    assert alias_structure(["ABCD"], 4, max_order=1) == {
        "A": ("BCD",), "B": ("ACD",), "C": ("ABD",), "D": ("ABC",)}


def test_validation():
    with pytest.raises(ValueError, match="positive integer"):
        full_factorial(0)
    with pytest.raises(ValueError, match="look like"):
        fractional_factorial(4, ["DABC"])
    with pytest.raises(ValueError, match="last p factors"):
        fractional_factorial(4, ["C=AB"])
    with pytest.raises(ValueError, match="two distinct"):
        fractional_factorial(4, ["D=A"])
    with pytest.raises(ValueError, match="non-base"):
        fractional_factorial(5, ["D=AB", "E=AD"])
    with pytest.raises(ValueError, match="no tabulated"):
        fractional_factorial(9, p=5)
    with pytest.raises(ValueError, match="unknown factor"):
        fractional_factorial(4, ["D=ABX"])
    d = full_factorial(2)
    with pytest.raises(ValueError, match="rows"):
        factorial_effects(d, [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        factorial_effects(d, [1, 2, 3, float("nan")])
    with pytest.raises(ValueError, match="three effects"):
        factorial_effects(full_factorial(1), [1, 2])
    with pytest.raises(ValueError, match="alpha"):
        factorial_effects(d, [1, 2, 3, 4], alpha=0)
    with pytest.raises(ValueError, match="unknown factor"):
        fold_over(d, factor="Z")


def test_cli_factorial(capsys):
    rc = main(["factorial", "--factors", "5", "--fraction", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2^(5-1)" in out and "resolution: V" in out and "I = ABCDE" in out
    rc = main(["factorial", "--factors", "4", "--responses", ",".join(map(str, FILTRATION))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Lenth PSE: 2.6250" in out and "A: +21.6250" in out
    rc = main(["factorial", "--factors", "7", "--generators", "D=AB,E=AC,F=BC,G=ABC", "--fold-over"])
    out = capsys.readouterr().out
    assert rc == 0 and "runs: 16" in out and "resolution: IV" in out
    rc = main(["factorial", "--factors", "4", "--generators", "C=AB"])
    assert rc == 2
