"""Tests for Taguchi orthogonal arrays and signal-to-noise analysis."""

import numpy as np
import pytest

from experiment_design_kit import (
    TAGUCHI_ARRAYS,
    interaction_columns,
    is_orthogonal_array,
    orthogonal_array,
    signal_to_noise,
    taguchi_analysis,
    taguchi_array,
    taguchi_design,
)
from experiment_design_kit.cli import main


@pytest.mark.parametrize("name", sorted(TAGUCHI_ARRAYS))
def test_standard_arrays_are_strength_two(name):
    A = taguchi_array(name)
    assert is_orthogonal_array(A, 2)
    expected = {"L4": (4, 3), "L8": (8, 7), "L9": (9, 4), "L16": (16, 15), "L16b": (16, 5),
                "L18": (18, 8), "L25": (25, 6), "L27": (27, 13), "L32": (32, 31), "L49": (49, 8),
                "L64b": (64, 21), "L81": (81, 40)}[name]
    assert A.shape == expected


def test_l8_and_l9_match_published_tables():
    L8 = taguchi_array("L8") + 1
    np.testing.assert_array_equal(L8[:, 0], [1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(L8[:, 1], [1, 1, 2, 2, 1, 1, 2, 2])
    np.testing.assert_array_equal(L8[:, 2], [1, 1, 2, 2, 2, 2, 1, 1])
    np.testing.assert_array_equal(L8[:, 3], [1, 2, 1, 2, 1, 2, 1, 2])
    np.testing.assert_array_equal(L8[:, 6], [1, 2, 2, 1, 2, 1, 1, 2])
    L9 = (taguchi_array("L9") + 1).T
    np.testing.assert_array_equal(L9, [
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 2, 3, 2, 3, 1, 3, 1, 2],
        [1, 3, 2, 2, 1, 3, 3, 2, 1],
    ])


def test_two_level_arrays_have_strength_two_but_not_three():
    A = orthogonal_array(2, 3)
    assert not is_orthogonal_array(A, 3)  # col3 = col1 + col2
    assert is_orthogonal_array(A[:, [0, 1, 3]], 3)  # basic columns form a full factorial


def test_interaction_columns_triangular_table():
    # Taguchi L8 triangular table (1-based): 1x2 -> 3, 1x4 -> 5, 2x4 -> 6, 3x4 -> 7
    assert interaction_columns(2, 3, 0, 1) == [2]
    assert interaction_columns(2, 3, 0, 3) == [4]
    assert interaction_columns(2, 3, 1, 3) == [5]
    assert interaction_columns(2, 3, 2, 3) == [6]
    # three-level: interaction of columns 1 and 2 of L9 lives in columns 3 and 4
    assert interaction_columns(3, 2, 0, 1) == [2, 3]
    A = orthogonal_array(2, 3)
    np.testing.assert_array_equal(A[:, 2], (A[:, 0] + A[:, 1]) % 2)
    with pytest.raises(ValueError):
        interaction_columns(2, 3, 1, 1)


def test_taguchi_design_picks_smallest_array():
    assert taguchi_design([2, 2, 2])[0] == "L4"
    assert taguchi_design([2] * 5)[0] == "L8"
    assert taguchi_design([3, 3, 3, 3])[0] == "L9"
    name, D, cols = taguchi_design([2, 3, 3, 3])
    assert name == "L18" and cols[0] == 0 and is_orthogonal_array(D)
    assert taguchi_design([4, 4])[0] == "L16b"
    with pytest.raises(ValueError, match="no standard"):
        taguchi_design([6, 6])
    with pytest.raises(ValueError):
        taguchi_design([1])


def test_signal_to_noise_formulas():
    y = np.array([[2.0, 4.0], [1.0, 1.0]])
    np.testing.assert_allclose(signal_to_noise(y, "larger"),
                               -10 * np.log10([(1 / 4 + 1 / 16) / 2, 1.0]))
    np.testing.assert_allclose(signal_to_noise(y, "smaller"), -10 * np.log10([10.0, 1.0]))
    z = np.array([[9.0, 11.0, 10.0]])
    np.testing.assert_allclose(signal_to_noise(z, "nominal"), 10 * np.log10(100.0 / 1.0))
    np.testing.assert_allclose(signal_to_noise(z, "nominal_variance"), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="non-zero"):
        signal_to_noise([[0.0, 1.0]], "larger")
    with pytest.raises(ValueError, match="replicates"):
        signal_to_noise([1.0, 2.0], "nominal")
    with pytest.raises(ValueError, match="kind"):
        signal_to_noise([1.0], "bigger")


def test_analysis_recovers_additive_model_exactly():
    name, D, _ = taguchi_design([3, 3, 3])  # L9, column 4 free -> 2 error df
    eff = {0: [0.0, 2.0, 1.0], 1: [1.0, 0.0, 3.0], 2: [0.0, 0.5, 0.2]}
    sn_true = 10 + sum(np.array(eff[j])[D[:, j]] for j in range(3))
    y = 10 ** (sn_true / 20.0)  # one replicate: larger-the-better S/N = 20 log10 y
    res = taguchi_analysis(D, y, kind="larger", factors=["temp", "time", "press"])
    np.testing.assert_allclose(res.sn, sn_true)
    assert res.optimal_levels == {"temp": 1, "time": 2, "press": 1}
    assert res.predicted_sn == pytest.approx(10 + 2 + 3 + 0.5)
    assert res.rank["time"] == 1 and res.rank["press"] == 3
    rows = {r["source"]: r for r in res.anova}
    assert rows["error"]["SS"] == pytest.approx(0.0, abs=1e-9)
    assert sum(r["percent"] for r in res.anova if r["source"] != "total") == pytest.approx(100.0)
    assert "predicted S/N" in res.summary()


def test_analysis_with_noise_replicates_and_pooling():
    rng = np.random.default_rng(0)
    A = taguchi_array("L8")
    D = A[:, [0, 1, 3]]  # main effects on basic columns, interactions free
    mean = 20 + 5 * D[:, 0] - 3 * D[:, 1]
    spread = 1 + 2 * D[:, 2]  # factor C controls variability only
    Y = mean[:, None] + spread[:, None] * rng.normal(size=(8, 6))
    res = taguchi_analysis(D, Y, kind="nominal")
    assert res.optimal_levels["C"] == 0  # low-variance level wins nominal S/N
    assert res.mean_table["A"][1] > res.mean_table["A"][0]
    pooled = taguchi_analysis(D, Y, kind="nominal", pool=1)
    assert len(pooled.pooled) == 1
    err = [r for r in pooled.anova if r["source"] == "error"][0]
    assert err["df"] == 4 + 1  # 7 - 3 factor df, plus the pooled factor


def test_saturated_design_has_no_error_row():
    D = taguchi_array("L4")
    res = taguchi_analysis(D, [1.0, 2.0, 3.0, 5.0])
    assert all(r["source"] != "error" for r in res.anova)
    assert all(np.isnan(r["F"]) for r in res.anova)


def test_analysis_validation():
    D = taguchi_array("L4")
    with pytest.raises(ValueError, match="same number"):
        taguchi_analysis(D, [1.0, 2.0])
    bad = np.array([[0, 0], [0, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match="orthogonal"):
        taguchi_analysis(bad, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="name"):
        taguchi_analysis(D, [1.0, 2.0, 3.0, 4.0], factors=["a"])
    with pytest.raises(ValueError, match="unknown array"):
        taguchi_array("L7")


def test_cli_taguchi(capsys):
    assert main(["taguchi", "--array", "L8"]) == 0
    out = capsys.readouterr().out
    assert "8 runs x 7 columns" in out
    resp = "10,11;12,13;14,15;11,12;13,15;12,12.5;16,17;12,14;13,13.5"
    assert main(["taguchi", "--levels", "3,3,3", "--responses", resp]) == 0
    out = capsys.readouterr().out
    assert "L9" in out and "predicted S/N" in out and "error" in out
    assert main(["taguchi", "--levels", "7,7,7,7,7,7,7,7,7"]) == 2
