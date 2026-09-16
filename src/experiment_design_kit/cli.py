"""Command-line interface for experiment-design-kit."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .cuped import cuped_adjust, simulate_cuped_data
from .mde import minimum_detectable_effect, minimum_detectable_effect_proportion, minimum_detectable_effect_raw
from .power import power_one_sample, power_proportion, power_two_sample
from .reporting import compose_demo_report
from .sequential import (
    sequential_mean_test,
    sequential_proportion_test,
    sequential_two_proportion_test,
    sequential_two_sample_mean_test,
    simulate_peeking_fpr,
)
from .stats import cohen_h, two_proportion_sample_size, two_sample_t_sample_size
from .simulation import run_continuous_ab_test, run_proportion_ab_test


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiment-design-kit")
    sub = parser.add_subparsers(dest="command", required=True)

    ss = sub.add_parser("sample-size", help="Compute the required sample size for a planned effect")
    _add_sample_size_args(ss)

    pw = sub.add_parser("power", help="Compute achieved power for a given sample size")
    _add_power_args(pw)

    mde = sub.add_parser("mde", help="Compute the minimum detectable effect for a given n")
    _add_mde_args(mde)

    sim = sub.add_parser("simulate", help="Simulate an A/B test outcome and check significance")
    _add_simulate_args(sim)

    cuped = sub.add_parser(
        "cuped",
        help="Apply CUPED variance reduction to a continuous metric (synthetic or CSV)",
    )
    _add_cuped_args(cuped)

    seq = sub.add_parser(
        "sequential",
        help="Always-valid / alpha-spending sequential tests, or a peeking FPR simulation",
    )
    _add_sequential_args(seq)

    rep = sub.add_parser("report", help="Run the full demo workflow and write a Markdown report")
    rep.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    rep.add_argument(
        "--output", "-o", default="examples/output/demo_report.md",
        help="Output path for the Markdown report (default: examples/output/demo_report.md)",
    )
    return parser


def _add_sample_size_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", choices=["proportion", "two-sample-t"], required=True, help="Effect-test family")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--power", type=float, default=0.8, help="Target power (default: 0.8)")
    p.add_argument("--p1", type=float, default=0.10, help="Control proportion (proportion method)")
    p.add_argument("--p2", type=float, default=0.12, help="Treatment proportion (proportion method)")
    p.add_argument("--d", type=float, default=None, help="Cohen's d (two-sample-t method)")


def _add_power_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", choices=["two-sample", "one-sample", "proportion"], required=True)
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--n", type=int, required=True, help="Sample size per group")
    p.add_argument("--d", type=float, default=0.5, help="Cohen's d (t-test methods)")
    p.add_argument("--p1", type=float, default=0.10, help="Control proportion (proportion method)")
    p.add_argument("--p2", type=float, default=0.12, help="Treatment proportion (proportion method)")


def _add_mde_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", choices=["two-sample", "one-sample", "proportion"], required=True)
    p.add_argument("--n", type=int, required=True, help="Planned sample size per group")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--power", type=float, default=0.8, help="Target power (default: 0.8)")
    p.add_argument("--d", type=float, default=0.5, help="Cohen's d (t-test methods, unused for proportion)")
    p.add_argument("--p-control", type=float, default=0.10, help="Control proportion (proportion method)")
    p.add_argument("--sd", type=float, default=None, help="Per-group SD; with this, MDE is reported in raw units (t-test methods)")


def _add_simulate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metric", choices=["proportion", "continuous"], required=True)
    p.add_argument("--n", type=int, default=2000, help="Sample size per group (default: 2000)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--control", type=float, default=0.10, help="Control rate (proportion) or mean (continuous)")
    p.add_argument("--treatment", type=float, default=0.12, help="Treatment rate (proportion)")
    p.add_argument("--sd", type=float, default=15.0, help="Per-group SD (continuous)")
    p.add_argument("--effect", type=float, default=5.0, help="Treatment effect in raw units (continuous)")


def _add_cuped_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--csv",
        default=None,
        help="CSV with header columns outcome,treatment,covariate (skips synthetic data)",
    )
    p.add_argument("--n", type=int, default=500, help="Sample size per group for synthetic data (default: 500)")
    p.add_argument(
        "--correlation",
        type=float,
        default=0.7,
        help="Pre/post correlation for synthetic data (default: 0.7)",
    )
    p.add_argument("--effect", type=float, default=0.5, help="True treatment effect for synthetic data (default: 0.5)")
    p.add_argument("--sd", type=float, default=1.0, help="Outcome SD for synthetic data (default: 1.0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--fit-on",
        choices=["pooled", "control"],
        default="pooled",
        help="Estimate theta from pooled data (default) or control only",
    )


def _add_sequential_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--method",
        choices=["proportion", "two-proportion", "mean", "two-mean", "peeking"],
        required=True,
        help="One-sample proportion/mean, two-arm A/B, or a peeking FPR simulation",
    )
    p.add_argument(
        "--boundary",
        choices=["always-valid", "pocock", "obrien-fleming"],
        default="always-valid",
        help="Rejection boundary (default: always-valid mSPRT)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--m", type=float, default=0.01, help="mSPRT mixture parameter (default: 0.01)")
    p.add_argument(
        "--stop-on-significant",
        action="store_true",
        help="Stop after the first significant look",
    )
    p.add_argument("--null-p", type=float, default=0.5, help="Null proportion (proportion method)")
    p.add_argument("--null-mean", type=float, default=0.0, help="Null mean (mean method)")
    p.add_argument("--successes", default=None, help="Incremental successes per look (comma-separated)")
    p.add_argument("--totals", default=None, help="Incremental sample sizes per look (comma-separated)")
    p.add_argument("--means", default=None, help="Cumulative means per look (comma-separated)")
    p.add_argument("--sems", default=None, help="Cumulative SEMs per look (comma-separated)")
    p.add_argument("--ns", default=None, help="Optional cumulative sample sizes per look (comma-separated)")
    p.add_argument("--control-successes", default=None, help="Incremental control successes (comma-separated)")
    p.add_argument("--control-totals", default=None, help="Incremental control sample sizes (comma-separated)")
    p.add_argument("--treatment-successes", default=None, help="Incremental treatment successes (comma-separated)")
    p.add_argument("--treatment-totals", default=None, help="Incremental treatment sample sizes (comma-separated)")
    p.add_argument("--control-means", default=None, help="Cumulative control means (comma-separated)")
    p.add_argument("--control-sds", default=None, help="Cumulative control SDs (comma-separated)")
    p.add_argument("--control-ns", default=None, help="Cumulative control sample sizes (comma-separated)")
    p.add_argument("--treatment-means", default=None, help="Cumulative treatment means (comma-separated)")
    p.add_argument("--treatment-sds", default=None, help="Cumulative treatment SDs (comma-separated)")
    p.add_argument("--treatment-ns", default=None, help="Cumulative treatment sample sizes (comma-separated)")
    p.add_argument("--looks", type=int, default=8, help="Peeking simulation: number of looks (default: 8)")
    p.add_argument(
        "--n-per-look",
        type=int,
        default=200,
        help="Peeking simulation: observations per arm per look (default: 200)",
    )
    p.add_argument("--trials", type=int, default=400, help="Peeking simulation: Monte Carlo trials (default: 400)")
    p.add_argument("--seed", type=int, default=0, help="Peeking simulation: RNG seed (default: 0)")
    p.add_argument(
        "--metric",
        choices=["proportion", "mean"],
        default="proportion",
        help="Peeking simulation metric (default: proportion)",
    )


def _parse_ints(text: str, flag: str) -> list[int]:
    try:
        values = [int(float(part.strip())) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag}: expected comma-separated integers") from exc
    if not values:
        raise ValueError(f"{flag}: expected at least one integer")
    return values


def _parse_floats(text: str, flag: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag}: expected comma-separated numbers") from exc
    if not values:
        raise ValueError(f"{flag}: expected at least one number")
    return values


def _print_sequential_report(report) -> None:
    print(f"boundary: {report.boundary}")
    print(
        f"{'look':>4}  {'n':>8}  {'z':>8}  {'naive p':>10}  "
        f"{'always-valid p':>14}  {'spent':>8}  {'sig':>3}"
    )
    for r in report.results:
        sig = "yes" if r.significant else "no"
        print(
            f"{r.look:4d}  {r.cumulative_n:8d}  {r.z_statistic:8.3f}  "
            f"{r.p_value:10.4f}  {r.always_valid_pvalue:14.4f}  "
            f"{r.alpha_spent:8.4f}  {sig:>3}"
        )
    print(f"significant: {'yes' if report.is_significant else 'no'}")
    print(f"naive repeated testing significant: {'yes' if report.naive_significant else 'no'}")
    print(f"final always-valid p: {report.final_pvalue:.4f}")
    if report.stopped:
        print(f"stopped at look: {report.stopped_at}")


def cmd_sequential(args: argparse.Namespace) -> int:
    try:
        if args.method == "peeking":
            result = simulate_peeking_fpr(
                n_looks=args.looks,
                n_per_look=args.n_per_look,
                n_trials=args.trials,
                alpha=args.alpha,
                metric=args.metric,
                m=args.m,
                seed=args.seed,
            )
            print(f"Peeking false-positive rates under H0 (alpha={result.alpha})")
            print(
                f"metric: {result.metric}  looks: {result.n_looks}  "
                f"n/look/group: {result.n_per_look}  trials: {result.n_trials}  seed: {result.seed}"
            )
            print(f"naive repeated testing : {result.naive_fpr:.3f}   (typically inflated)")
            print(f"always-valid mSPRT     : {result.always_valid_fpr:.3f}")
            print(f"Pocock spending        : {result.pocock_fpr:.3f}")
            print(f"O'Brien-Fleming        : {result.obrien_fleming_fpr:.3f}")
            return 0

        if args.method == "proportion":
            if not args.successes or not args.totals:
                print("sequential: --successes and --totals are required", file=sys.stderr)
                return 2
            report = sequential_proportion_test(
                _parse_ints(args.successes, "--successes"),
                _parse_ints(args.totals, "--totals"),
                alpha=args.alpha,
                m=args.m,
                stop_on_significant=args.stop_on_significant,
                null_p=args.null_p,
                boundary=args.boundary,
            )
        elif args.method == "two-proportion":
            required = (
                args.control_successes,
                args.control_totals,
                args.treatment_successes,
                args.treatment_totals,
            )
            if not all(required):
                print(
                    "sequential: --control-successes, --control-totals, "
                    "--treatment-successes, and --treatment-totals are required",
                    file=sys.stderr,
                )
                return 2
            report = sequential_two_proportion_test(
                _parse_ints(args.control_successes, "--control-successes"),
                _parse_ints(args.control_totals, "--control-totals"),
                _parse_ints(args.treatment_successes, "--treatment-successes"),
                _parse_ints(args.treatment_totals, "--treatment-totals"),
                alpha=args.alpha,
                m=args.m,
                stop_on_significant=args.stop_on_significant,
                boundary=args.boundary,
            )
        elif args.method == "mean":
            if not args.means or not args.sems:
                print("sequential: --means and --sems are required", file=sys.stderr)
                return 2
            report = sequential_mean_test(
                _parse_floats(args.means, "--means"),
                _parse_floats(args.sems, "--sems"),
                null_mean=args.null_mean,
                alpha=args.alpha,
                m=args.m,
                stop_on_significant=args.stop_on_significant,
                ns=_parse_ints(args.ns, "--ns") if args.ns else None,
                boundary=args.boundary,
            )
        else:
            required = (
                args.control_means,
                args.control_sds,
                args.control_ns,
                args.treatment_means,
                args.treatment_sds,
                args.treatment_ns,
            )
            if not all(required):
                print(
                    "sequential: --control-means, --control-sds, --control-ns, "
                    "--treatment-means, --treatment-sds, and --treatment-ns are required",
                    file=sys.stderr,
                )
                return 2
            report = sequential_two_sample_mean_test(
                _parse_floats(args.control_means, "--control-means"),
                _parse_floats(args.control_sds, "--control-sds"),
                _parse_ints(args.control_ns, "--control-ns"),
                _parse_floats(args.treatment_means, "--treatment-means"),
                _parse_floats(args.treatment_sds, "--treatment-sds"),
                _parse_ints(args.treatment_ns, "--treatment-ns"),
                alpha=args.alpha,
                m=args.m,
                stop_on_significant=args.stop_on_significant,
                boundary=args.boundary,
            )
    except (OSError, ValueError) as exc:
        print(f"sequential: {exc}", file=sys.stderr)
        return 2

    _print_sequential_report(report)
    return 0


def cmd_sample_size(args: argparse.Namespace) -> int:
    if args.method == "proportion":
        result = two_proportion_sample_size(args.p1, args.p2, alpha=args.alpha, power=args.power)
        h = cohen_h(args.p1, args.p2)
        print(f"effect size (Cohen's h): {h:.4f}")
        print(f"sample size per group: {result.n_per_group_required:,}")
        print(f"total sample size: {result.n_total_required:,}")
    else:
        if args.d is None:
            print("sample-size: --d is required for method two-sample-t", file=sys.stderr)
            return 2
        result = two_sample_t_sample_size(args.d, alpha=args.alpha, power=args.power)
        print(f"effect size (Cohen's d): {args.d:.4f}")
        print(f"sample size per group: {result.n_per_group_required:,}")
        print(f"total sample size: {result.n_total_required:,}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    if args.method == "proportion":
        pwr = power_proportion(args.p1, args.p2, args.n, alpha=args.alpha)
        print(f"power (proportion): {pwr:.4f}")
    elif args.method == "two-sample":
        pwr = power_two_sample(args.d, args.n, alpha=args.alpha)
        print(f"power (two-sample t): {pwr:.4f}")
    else:
        pwr = power_one_sample(args.d, args.n, alpha=args.alpha)
        print(f"power (one-sample t): {pwr:.4f}")
    return 0


def cmd_mde(args: argparse.Namespace) -> int:
    if args.method == "proportion":
        result = minimum_detectable_effect_proportion(args.n, args.p_control, alpha=args.alpha, power=args.power)
        print(f"MDE lift (absolute): {result.lift_absolute:.4f}")
        print(f"MDE lift (relative): {result.lift_relative * 100:.1f}%")
        print(f"effect size (Cohen's h): {result.effect_size_h:.4f}")
    else:
        d = minimum_detectable_effect(args.n, alpha=args.alpha, power=args.power, kind=args.method).effect_size
        print(f"MDE effect size (Cohen's d): {d:.4f}")
        if args.sd is not None:
            raw = minimum_detectable_effect_raw(args.n, alpha=args.alpha, power=args.power, sigma=args.sd, kind=args.method)
            print(f"MDE raw effect (sd={args.sd}): {raw:.4f}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    if args.metric == "proportion":
        result = run_proportion_ab_test(args.control, args.treatment, args.n, alpha=args.alpha, seed=args.seed)
        print(f"control rate: {result.control_estimate:.4f}")
        print(f"treatment rate: {result.treatment_estimate:.4f}")
        print(f"lift: {result.lift_absolute:+.4f} ({result.lift_relative * 100:+.1f}%)")
        print(f"chi-square: {result.statistic:.3f}  p-value: {result.p_value:.4f}")
        print(f"significant at alpha={args.alpha}: {'yes' if result.significant else 'no'}")
    else:
        result = run_continuous_ab_test(args.control, args.sd, args.effect, args.n, alpha=args.alpha, seed=args.seed)
        print(f"control mean: {result.control_estimate:.4f}")
        print(f"treatment mean: {result.treatment_estimate:.4f}")
        print(f"lift: {result.lift_absolute:+.4f} ({result.lift_relative * 100:+.1f}%)")
        print(f"t-statistic: {result.statistic:.3f}  p-value: {result.p_value:.4f}")
        print(f"significant at alpha={args.alpha}: {'yes' if result.significant else 'no'}")
    return 0


def _load_cuped_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV is missing a header row")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        required = ("outcome", "treatment", "covariate")
        missing = [col for col in required if col not in fields]
        if missing:
            raise ValueError(
                "CSV must have columns outcome, treatment, covariate "
                f"(missing: {', '.join(missing)})"
            )
        outcomes: list[float] = []
        treatment: list[int] = []
        covariates: list[float] = []
        for row in reader:
            outcomes.append(float(row[fields["outcome"]]))
            treatment.append(int(float(row[fields["treatment"]])))
            covariates.append(float(row[fields["covariate"]]))
    return (
        np.asarray(outcomes, dtype=float),
        np.asarray(treatment, dtype=int),
        np.asarray(covariates, dtype=float),
    )


def cmd_cuped(args: argparse.Namespace) -> int:
    try:
        if args.csv:
            outcomes, treatment, covariates = _load_cuped_csv(Path(args.csv))
            source = f"csv:{args.csv}"
        else:
            outcomes, treatment, covariates = simulate_cuped_data(
                args.n,
                correlation=args.correlation,
                treatment_effect=args.effect,
                outcome_sd=args.sd,
                seed=args.seed,
            )
            source = f"synthetic n/group={args.n} correlation={args.correlation} seed={args.seed}"
        result = cuped_adjust(outcomes, treatment, covariates, fit_on=args.fit_on)
    except (OSError, ValueError) as exc:
        print(f"cuped: {exc}", file=sys.stderr)
        return 2

    print(f"source: {source}")
    print(f"n control/treatment: {result.n_control}/{result.n_treatment}")
    print(f"theta: {result.theta:.4f}  (fit_on={result.fit_on})")
    print(f"pre/post correlation: {result.correlation:.4f}")
    print(f"raw variance: {result.raw_variance:.4f}")
    print(f"adjusted variance: {result.adjusted_variance:.4f}")
    print(f"variance ratio: {result.variance_ratio:.4f}")
    print(f"variance reduction: {result.variance_reduction * 100:.1f}%")
    print(f"raw effect: {result.raw_effect:+.4f}")
    print(f"CUPED effect: {result.effect:+.4f}")
    print(f"adjusted control mean: {result.adjusted_control_mean:.4f}")
    print(f"adjusted treatment mean: {result.adjusted_treatment_mean:.4f}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = compose_demo_report(seed=args.seed)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"Wrote {target}")
    return 0


_COMMANDS = {
    "sample-size": cmd_sample_size,
    "power": cmd_power,
    "mde": cmd_mde,
    "simulate": cmd_simulate,
    "cuped": cmd_cuped,
    "sequential": cmd_sequential,
    "report": cmd_report,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
