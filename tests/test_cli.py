"""Tests for the experiment-design-kit CLI."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

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
