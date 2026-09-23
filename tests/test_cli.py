"""Tests for the experiment-design-kit CLI."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from experiment_design_kit import simulate_cuped_data
from experiment_design_kit.cli import main


def _run(argv: list[str]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    assert rc == 0, buf.getvalue()
    return buf.getvalue()


def test_sample_size_proportion() -> None:
    out = _run(["sample-size", "--method", "proportion", "--p1", "0.10", "--p2", "0.12"])
    assert "3,841" in out
    assert "Cohen's h" in out


def test_sample_size_two_sample_t() -> None:
    out = _run(["sample-size", "--method", "two-sample-t", "--d", "0.5"])
    assert "63" in out


def test_power_two_sample() -> None:
    out = _run(["power", "--method", "two-sample", "--d", "0.5", "--n", "64"])
    assert "0.80" in out


def test_power_one_sample() -> None:
    out = _run(["power", "--method", "one-sample", "--d", "0.5", "--n", "34"])
    assert "0.80" in out


def test_power_proportion() -> None:
    out = _run(["power", "--method", "proportion", "--p1", "0.10", "--p2", "0.12", "--n", "3841"])
    match = re.search(r"power \(proportion\): ([\d.]+)", out)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(0.80, abs=1e-2)


def test_mde_two_sample_with_sd() -> None:
    out = _run(["mde", "--method", "two-sample", "--n", "64", "--sd", "10"])
    assert "Cohen's d" in out
    assert "MDE raw effect" in out


def test_mde_proportion() -> None:
    out = _run(["mde", "--method", "proportion", "--n", "3841", "--p-control", "0.10"])
    assert "MDE lift (absolute)" in out
    assert re.search(r"0\.0\d", out)


def test_simulate_proportion() -> None:
    out = _run(["simulate", "--metric", "proportion", "--control", "0.10", "--treatment", "0.14", "--n", "2000", "--seed", "42"])
    assert "chi-square" in out
    assert "p-value" in out
    assert re.search(r"significant at alpha=0.05: (yes|no)", out)


def test_simulate_continuous() -> None:
    out = _run(["simulate", "--metric", "continuous", "--control", "100", "--sd", "15", "--effect", "10", "--n", "500", "--seed", "7"])
    assert "t-statistic" in out
    assert "p-value" in out


def test_report_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "demo_report.md"
    out = _run(["report", "--seed", "42", "--output", str(target)])
    assert "Wrote" in out
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "# Experiment design report" in text
    assert "Proportion scenario (conversion rate)" in text
    assert "Continuous scenario (revenue per user)" in text
    assert "Conclusion" in text


def test_report_default_path() -> None:
    out = _run(["report", "--seed", "1", "--output", "examples/output/demo_report.md"])
    assert "Wrote" in out
    assert Path("examples/output/demo_report.md").exists()


def test_cuped_synthetic() -> None:
    out = _run(["cuped", "--n", "200", "--correlation", "0.8", "--effect", "0.5", "--seed", "42"])
    assert "theta:" in out
    assert "variance reduction:" in out
    assert "CUPED effect:" in out
    assert "raw effect:" in out
    match = re.search(r"variance reduction: ([\d.]+)%", out)
    assert match is not None
    assert float(match.group(1)) > 40.0


def test_cuped_from_csv(tmp_path: Path) -> None:
    outcomes, treatment, covariates = simulate_cuped_data(
        80, correlation=0.75, treatment_effect=0.4, seed=1
    )
    csv_path = tmp_path / "metrics.csv"
    lines = ["outcome,treatment,covariate"]
    for y, t, x in zip(outcomes, treatment, covariates):
        lines.append(f"{y},{int(t)},{x}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = _run(["cuped", "--csv", str(csv_path), "--fit-on", "control"])
    assert "source: csv:" in out
    assert "fit_on=control" in out
    assert "CUPED effect:" in out


def test_sequential_peeking_fpr() -> None:
    out = _run(
        [
            "sequential",
            "--method",
            "peeking",
            "--looks",
            "6",
            "--n-per-look",
            "150",
            "--trials",
            "200",
            "--seed",
            "0",
        ]
    )
    assert "naive repeated testing" in out
    assert "always-valid mSPRT" in out
    assert "Pocock spending" in out
    assert "O'Brien-Fleming" in out
    naive = float(re.search(r"naive repeated testing : ([\d.]+)", out).group(1))
    av = float(re.search(r"always-valid mSPRT     : ([\d.]+)", out).group(1))
    assert naive > av


def test_sequential_two_proportion() -> None:
    out = _run(
        [
            "sequential",
            "--method",
            "two-proportion",
            "--control-successes",
            "10,10,10",
            "--control-totals",
            "100,100,100",
            "--treatment-successes",
            "20,25,30",
            "--treatment-totals",
            "100,100,100",
        ]
    )
    assert "always-valid p" in out
    assert "boundary: always-valid" in out
    assert "final always-valid p:" in out


def test_sequential_mean() -> None:
    out = _run(
        [
            "sequential",
            "--method",
            "mean",
            "--means",
            "0.1,0.5,1.2",
            "--sems",
            "0.4,0.2,0.1",
            "--boundary",
            "pocock",
        ]
    )
    assert "boundary: pocock" in out
    assert "naive p" in out


def test_sequential_missing_args() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = main(["sequential", "--method", "proportion"])
    assert rc == 2
    assert "successes" in err.getvalue()


def test_bayesian_power_threshold() -> None:
    out = _run(
        [
            "bayesian",
            "--method",
            "power",
            "--p1",
            "0.20",
            "--p2",
            "0.45",
            "--n",
            "250",
            "--threshold",
            "0.90",
            "--trials",
            "200",
            "--posterior-samples",
            "600",
            "--seed",
            "0",
        ]
    )
    assert "Bayesian power" in out
    assert "power (P(decision)):" in out
    assert "P(declare treatment better):" in out
    power = float(re.search(r"power \(P\(decision\)\): ([\d.]+)", out).group(1))
    treat = float(re.search(r"P\(declare treatment better\): ([\d.]+)", out).group(1))
    assert power > 0.8
    assert treat > 0.8


def test_bayesian_power_from_relative_lift() -> None:
    out = _run(
        [
            "bayesian",
            "--method",
            "power",
            "--p1",
            "0.20",
            "--lift",
            "1.0",
            "--n",
            "180",
            "--trials",
            "150",
            "--posterior-samples",
            "500",
            "--seed",
            "1",
        ]
    )
    assert "0.2000 / 0.4000" in out


def test_bayesian_power_rope() -> None:
    out = _run(
        [
            "bayesian",
            "--method",
            "power",
            "--p1",
            "0.25",
            "--p2",
            "0.25",
            "--n",
            "2000",
            "--decision",
            "rope",
            "--rope-lower",
            "-0.05",
            "--rope-upper",
            "0.05",
            "--threshold",
            "0.90",
            "--trials",
            "150",
            "--posterior-samples",
            "500",
            "--seed",
            "2",
        ]
    )
    assert "ROPE:" in out
    assert "P(declare equivalent):" in out
    equiv = float(re.search(r"P\(declare equivalent\): ([\d.]+)", out).group(1))
    assert equiv > 0.5


def test_bayesian_sample_size() -> None:
    out = _run(
        [
            "bayesian",
            "--method",
            "sample-size",
            "--p1",
            "0.25",
            "--p2",
            "0.45",
            "--power",
            "0.70",
            "--threshold",
            "0.90",
            "--trials",
            "150",
            "--posterior-samples",
            "500",
            "--seed",
            "3",
        ]
    )
    assert "Bayesian sample size" in out
    assert "sample size per group:" in out
    assert "achieved power:" in out
    achieved = float(re.search(r"achieved power: ([\d.]+)", out).group(1))
    assert achieved >= 0.70


def test_bayesian_power_requires_n() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = main(["bayesian", "--method", "power", "--p1", "0.1", "--p2", "0.2"])
    assert rc == 2
    assert "--n is required" in err.getvalue()


def test_bayesian_rejects_p2_and_lift() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = main(
            [
                "bayesian",
                "--method",
                "power",
                "--p1",
                "0.1",
                "--p2",
                "0.2",
                "--lift",
                "0.1",
                "--n",
                "100",
            ]
        )
    assert rc == 2
    assert "not both" in err.getvalue()


def test_randomize_synthetic_reports_balance() -> None:
    out = _run(["randomize", "--n", "240", "--seed", "0", "--bins", "4"])
    assert "source: synthetic" in out
    assert "strata:" in out
    assert "arm counts:" in out
    assert "covariate balance:" in out
    assert "region (categorical)" in out
    assert "score (continuous)" in out
    assert "quantile bins:" in out
    assert "quantile-bin balance:" in out
    assert "max |SMD|:" in out
    region_line = next(line for line in out.splitlines() if "region (categorical)" in line)
    region_smd = float(re.search(r"max\|SMD\|=([0-9.]+)", region_line).group(1))
    assert region_smd < 0.25


def test_randomize_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "units.csv"
    lines = ["id,region,pre_metric"]
    for index in range(40):
        region = "us" if index < 24 else "eu"
        lines.append(f"u{index},{region},{index / 10:.2f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = _run(
        [
            "randomize",
            "--csv",
            str(csv_path),
            "--bins",
            "pre_metric=4",
            "--categorical",
            "region",
            "--seed",
            "3",
        ]
    )
    assert f"source: csv:{csv_path}" in out
    assert "region (categorical)" in out
    assert "pre_metric (continuous)" in out
    assert "quantile-bin balance:" in out


def test_randomize_ratio_and_three_arms() -> None:
    out = _run(
        [
            "randomize",
            "--n",
            "90",
            "--arms",
            "3",
            "--ratio",
            "1,1,2",
            "--seed",
            "1",
            "--levels",
            "2",
        ]
    )
    assert "arm_0=" in out
    assert "arm_1=" in out
    assert "arm_2=" in out
    counts = [
        int(value)
        for value in re.findall(r"arm_\d=(\d+)", out.split("covariate balance:")[0])
    ]
    assert counts[2] > counts[0]


def test_randomize_requires_bins_for_continuous_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "units.csv"
    csv_path.write_text("score\n0.1\n0.4\n0.9\n1.2\n", encoding="utf-8")
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = main(["randomize", "--csv", str(csv_path), "--seed", "0"])
    assert rc == 2
    assert "continuous" in err.getvalue()


def test_block_equal_allocation_balances_every_block() -> None:
    out = _run(["block", "--n", "20", "--block-size", "4", "--seed", "0"])
    assert "block size: 4" in out
    assert "blocks: 5" in out
    assert "seed: 0" in out
    assert "per complete block: control=2, treatment=2" in out
    assert "arm counts: control=10, treatment=10" in out
    assert "final block:" not in out


def test_block_ratio_and_partial_final_block() -> None:
    out = _run(["block", "--n", "10", "--block-size", "4", "--seed", "0"])
    assert "blocks: 3" in out
    assert "arm counts: control=5, treatment=5" in out
    assert "final block: control=1, treatment=1" in out
    ratio = _run(
        ["block", "--n", "18", "--block-size", "6", "--ratio", "1,2", "--seed", "1"]
    )
    assert "per complete block: control=2, treatment=4" in ratio
    assert "arm counts: control=6, treatment=12" in ratio


def test_block_three_arms() -> None:
    out = _run(["block", "--n", "9", "--block-size", "3", "--arms", "3", "--seed", "0"])
    assert "arm_0=3" in out
    assert "arm_1=3" in out
    assert "arm_2=3" in out


def test_block_rejects_odd_size_for_equal_arms() -> None:
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = main(["block", "--n", "10", "--block-size", "3", "--seed", "0"])
    assert rc == 2
    assert "2k" in err.getvalue()


def test_cuped_missing_csv_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    buf = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        rc = main(["cuped", "--csv", str(bad)])
    assert rc == 2
    assert "outcome" in err.getvalue()
