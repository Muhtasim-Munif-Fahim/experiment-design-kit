"""Command-line interface for experiment-design-kit."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .bayesian import bayesian_power_proportion, required_bayesian_sample_size
from .cuped import cuped_adjust, simulate_cuped_data
from .mde import minimum_detectable_effect, minimum_detectable_effect_proportion, minimum_detectable_effect_raw
from .power import power_one_sample, power_proportion, power_two_sample
from .randomization import (
    balance_report,
    blocked_randomization,
    cluster_balance_report,
    cluster_randomization,
    stratified_randomization,
    switchback_randomization,
)
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

    bayes = sub.add_parser(
        "bayesian",
        help="Bayesian power or sample-size planning for conversion A/B tests",
    )
    _add_bayesian_args(bayes)

    rand = sub.add_parser(
        "randomize",
        help="Stratified randomization and covariate balance (SMD / chi-square)",
    )
    _add_randomize_args(rand)

    block = sub.add_parser(
        "block",
        help="Permuted-block randomization with a fixed block size (2k for equal arms)",
    )
    _add_block_args(block)

    cluster = sub.add_parser(
        "cluster",
        help="Cluster randomization: assign whole clusters to arms",
    )
    _add_cluster_args(cluster)

    switchback = sub.add_parser(
        "switchback",
        help="Switchback randomization: assign consecutive time blocks to arms",
    )
    _add_switchback_args(switchback)

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


def _add_bayesian_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--method",
        choices=["power", "sample-size"],
        required=True,
        help="Estimate decision power at a fixed n, or search for n",
    )
    p.add_argument("--p1", type=float, required=True, help="Assumed control conversion rate")
    p.add_argument("--p2", type=float, default=None, help="Assumed treatment conversion rate")
    p.add_argument(
        "--lift",
        type=float,
        default=None,
        help="Relative lift alternative to --p2 (p2 = p1 * (1 + lift))",
    )
    p.add_argument("--n", type=int, default=None, help="Sample size per control group (power method)")
    p.add_argument(
        "--power",
        type=float,
        default=0.8,
        help="Target Bayesian decision power for sample-size search (default: 0.8)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Posterior probability needed to decide (default: 0.95)",
    )
    p.add_argument(
        "--decision",
        choices=["threshold", "rope"],
        default="threshold",
        help="Decision rule: P(B>A) threshold or ROPE (default: threshold)",
    )
    p.add_argument("--rope-lower", type=float, default=None, help="ROPE lower bound (required for rope)")
    p.add_argument("--rope-upper", type=float, default=None, help="ROPE upper bound (required for rope)")
    p.add_argument(
        "--rope-scale",
        choices=["absolute", "relative"],
        default="absolute",
        help="ROPE is on θ_t-θ_c (absolute) or relative lift (default: absolute)",
    )
    p.add_argument("--prior-alpha", type=float, default=1.0, help="Beta prior alpha (default: 1)")
    p.add_argument("--prior-beta", type=float, default=1.0, help="Beta prior beta (default: 1)")
    p.add_argument("--ratio", type=float, default=1.0, help="n_treatment / n_control (default: 1)")
    p.add_argument("--trials", type=int, default=1000, help="Monte Carlo experiments (default: 1000)")
    p.add_argument(
        "--posterior-samples",
        type=int,
        default=2000,
        help="Posterior draws per experiment (default: 2000)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")


def _add_randomize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--csv",
        default=None,
        help="CSV of covariates, one row per unit (id/unit_id/user_id columns are ignored)",
    )
    p.add_argument("--n", type=int, default=240, help="Units in the synthetic example (default: 240)")
    p.add_argument("--arms", type=int, default=2, help="Number of arms (default: 2)")
    p.add_argument(
        "--ratio",
        default=None,
        help="Comma-separated allocation weights, one per arm (default: equal)",
    )
    p.add_argument(
        "--bins",
        default=None,
        help="Quantile bins: an integer for every continuous covariate, or name=bins pairs",
    )
    p.add_argument(
        "--categorical",
        default=None,
        help="Comma-separated names to treat as categories in the balance report",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Levels of the synthetic categorical covariate (default: 3)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")


def _add_cluster_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--csv",
        default=None,
        help="CSV with one row per unit (id/unit_id/user_id columns are ignored)",
    )
    p.add_argument(
        "--cluster",
        default="cluster",
        help="Cluster-id column in the CSV (default: cluster)",
    )
    p.add_argument("--n", type=int, default=120, help="Units in the synthetic example (default: 120)")
    p.add_argument(
        "--clusters",
        type=int,
        default=12,
        help="Clusters in the synthetic example (default: 12)",
    )
    p.add_argument("--arms", type=int, default=2, help="Number of arms (default: 2)")
    p.add_argument(
        "--ratio",
        default=None,
        help="Comma-separated allocation weights, one per arm (default: equal). Weights count clusters.",
    )
    p.add_argument(
        "--categorical",
        default=None,
        help="Comma-separated names to treat as categories in the cluster balance report",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")


def _add_block_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n", type=int, required=True, help="Number of units in enrollment order")
    p.add_argument(
        "--block-size",
        type=int,
        required=True,
        help="Fixed block size; equal treatment/control requires 2k (2, 4, 6, ...)",
    )
    p.add_argument("--arms", type=int, default=2, help="Number of arms (default: 2)")
    p.add_argument(
        "--ratio",
        default=None,
        help="Comma-separated allocation weights, one per arm (default: equal)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")


def _treatment_rate_from_args(args: argparse.Namespace) -> float:
    if args.p2 is not None and args.lift is not None:
        raise ValueError("provide --p2 or --lift, not both")
    if args.p2 is not None:
        return args.p2
    if args.lift is not None:
        p2 = args.p1 * (1.0 + args.lift)
        if not 0.0 <= p2 <= 1.0:
            raise ValueError("p1 * (1 + lift) must be in [0, 1]")
        return p2
    raise ValueError("--p2 or --lift is required")


def _rope_from_args(args: argparse.Namespace) -> tuple[float, float] | None:
    if args.decision != "rope":
        if args.rope_lower is not None or args.rope_upper is not None:
            raise ValueError("--rope-lower/--rope-upper are only used with --decision rope")
        return None
    if args.rope_lower is None or args.rope_upper is None:
        raise ValueError("--rope-lower and --rope-upper are required for --decision rope")
    return (args.rope_lower, args.rope_upper)


def _print_bayesian_power(result, heading: str) -> None:
    print(heading)
    print(f"control/treatment rates: {result.p_control:.4f} / {result.p_treatment:.4f}")
    print(
        f"n control/treatment: {result.n_control:,}/{result.n_treatment:,}  "
        f"threshold: {result.threshold:.3f}  decision: {result.decision}"
    )
    if result.decision == "rope":
        print(
            f"ROPE: [{result.rope_lower}, {result.rope_upper}] "
            f"on {result.rope_scale} lift"
        )
    print(
        f"trials: {result.n_trials:,}  posterior samples: {result.n_posterior_samples:,}  "
        f"seed: {result.seed}"
    )
    print(f"power (P(decision)): {result.power:.4f}")
    print(f"P(declare treatment better): {result.prob_treatment_wins:.4f}")
    print(f"P(declare control better): {result.prob_control_wins:.4f}")
    if result.decision == "rope":
        print(f"P(declare equivalent): {result.prob_equivalent:.4f}")
    print(f"P(inconclusive): {result.prob_inconclusive:.4f}")
    print(f"mean P(treatment > control): {result.mean_prob_treatment_better:.4f}")


def cmd_bayesian(args: argparse.Namespace) -> int:
    try:
        p_treatment = _treatment_rate_from_args(args)
        rope = _rope_from_args(args)
        if args.method == "power":
            if args.n is None:
                print("bayesian: --n is required for method power", file=sys.stderr)
                return 2
            result = bayesian_power_proportion(
                args.p1,
                p_treatment,
                args.n,
                threshold=args.threshold,
                decision=args.decision,
                rope=rope,
                rope_scale=args.rope_scale,
                prior_alpha=args.prior_alpha,
                prior_beta=args.prior_beta,
                ratio=args.ratio,
                n_trials=args.trials,
                n_posterior_samples=args.posterior_samples,
                random_state=args.seed,
            )
            _print_bayesian_power(result, "Bayesian power")
            return 0

        plan = required_bayesian_sample_size(
            args.p1,
            p_treatment,
            target_power=args.power,
            threshold=args.threshold,
            decision=args.decision,
            rope=rope,
            rope_scale=args.rope_scale,
            prior_alpha=args.prior_alpha,
            prior_beta=args.prior_beta,
            ratio=args.ratio,
            n_trials=args.trials,
            n_posterior_samples=args.posterior_samples,
            random_state=args.seed,
        )
        print("Bayesian sample size")
        print(f"control/treatment rates: {plan.p_control:.4f} / {plan.p_treatment:.4f}")
        print(f"target power: {plan.target_power:.2f}  threshold: {plan.threshold:.3f}  decision: {plan.decision}")
        if plan.decision == "rope":
            print(f"ROPE: [{plan.rope_lower}, {plan.rope_upper}] on {plan.rope_scale} lift")
        print(f"sample size per group: {plan.n_per_group:,}")
        print(f"total sample size: {plan.n_total:,}")
        print(f"achieved power: {plan.achieved_power:.4f}")
        print(f"P(declare treatment better): {plan.power_result.prob_treatment_wins:.4f}")
        print(f"trials: {plan.n_trials:,}  seed: {plan.seed}")
        return 0
    except ValueError as exc:
        print(f"bayesian: {exc}", file=sys.stderr)
        return 2


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


_ID_COLUMNS = {"id", "unit_id", "user_id"}


def _parse_bins(text: str) -> int | dict[str, int]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("--bins must be an integer or name=bins pairs")
    if "=" not in stripped:
        try:
            return int(stripped)
        except ValueError as exc:
            raise ValueError("--bins must be an integer or name=bins pairs") from exc
    spec: dict[str, int] = {}
    for part in stripped.split(","):
        if "=" not in part:
            raise ValueError("--bins must be an integer or name=bins pairs")
        name, raw = part.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--bins entries need a covariate name")
        try:
            spec[name] = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"--bins value for {name} must be an integer") from exc
    return spec


def _parse_name_list(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("expected at least one covariate name")
    return names


def _synthetic_covariates(n: int, levels: int, seed: int) -> dict[str, np.ndarray]:
    if n < 2:
        raise ValueError("--n must be at least 2")
    if levels < 2:
        raise ValueError("--levels must be at least 2")
    rng = np.random.default_rng(seed)
    names = [f"r{i}" for i in range(levels)]
    weights = np.linspace(1.0, 3.0, levels)
    weights = weights / weights.sum()
    region = rng.choice(names, size=n, p=weights)
    score = rng.normal(size=n)
    return {"region": region, "score": score}


def _parse_covariate_column(name: str, raw: list[str]) -> np.ndarray:
    if any(value == "" for value in raw):
        raise ValueError(f"column {name} contains empty values")
    try:
        numbers = [float(value) for value in raw]
    except ValueError:
        return np.asarray(raw)
    array = np.asarray(numbers, dtype=float)
    if np.all(array == np.floor(array)):
        return array.astype(int)
    return array


def _load_covariate_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is missing a header row")
        columns = [
            name for name in reader.fieldnames
            if name and name.strip() and name.strip().lower() not in _ID_COLUMNS
        ]
        if not columns:
            raise ValueError("CSV has no covariate columns")
        buckets: dict[str, list[str]] = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                value = row.get(name)
                buckets[name].append("" if value is None else value.strip())
    if not buckets[columns[0]]:
        raise ValueError("CSV has no data rows")
    return {name: _parse_covariate_column(name, values) for name, values in buckets.items()}


def cmd_randomize(args: argparse.Namespace) -> int:
    try:
        if args.csv:
            covariates = _load_covariate_csv(Path(args.csv))
            source = f"csv:{args.csv}"
            n_bins = _parse_bins(args.bins) if args.bins else None
        else:
            covariates = _synthetic_covariates(args.n, args.levels, args.seed)
            source = (
                f"synthetic n={args.n} levels={args.levels} seed={args.seed}"
            )
            n_bins = _parse_bins(args.bins) if args.bins else 4
        ratio = _parse_floats(args.ratio, "--ratio") if args.ratio else None
        categorical = _parse_name_list(args.categorical) if args.categorical else None
        result = stratified_randomization(
            covariates,
            n_arms=args.arms,
            ratio=ratio,
            n_bins=n_bins,
            seed=args.seed,
        )
        report = balance_report(
            covariates,
            result.assignment,
            arm_labels=result.arm_labels,
            categorical=categorical,
        )
        binned = {name: result.factors[name] for name in result.bin_edges}
        bin_report = None
        if binned:
            bin_report = balance_report(
                binned,
                result.assignment,
                arm_labels=result.arm_labels,
                categorical=tuple(binned),
            )
    except (OSError, ValueError) as exc:
        print(f"randomize: {exc}", file=sys.stderr)
        return 2

    counts = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.arm_counts)
    )
    print(f"source: {source}")
    print(f"n: {result.assignment.size}  strata: {result.n_strata}  seed: {result.seed}")
    print(f"arm counts: {counts}")
    if result.bin_edges:
        described = ", ".join(
            f"{name}:{len(edges) - 1}" for name, edges in result.bin_edges.items()
        )
        print(f"quantile bins: {described}")
    print("covariate balance:")
    print(report)
    if bin_report is not None:
        print("quantile-bin balance:")
        print(bin_report)
    return 0


def _synthetic_cluster_data(
    n: int, n_clusters: int, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("--n must be an integer >= 2")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError("--clusters must be an integer >= 2")
    if n < n_clusters:
        raise ValueError("--n must be at least --clusters")
    rng = np.random.default_rng(seed)
    sizes = rng.multinomial(n - n_clusters, np.full(n_clusters, 1.0 / n_clusters)) + 1
    cluster_ids = np.repeat(np.arange(n_clusters, dtype=int), sizes)
    regions = np.array(["north", "south", "east"])
    region = regions[np.arange(n_clusters) % 3][cluster_ids]
    shift = (np.arange(n_clusters) % 5) * 0.15
    score = rng.normal(loc=shift[cluster_ids], scale=1.0)
    return cluster_ids, {"region": region, "score": score}


def _load_cluster_csv(
    path: Path, cluster_column: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is missing a header row")
        fieldnames = [name.strip() for name in reader.fieldnames if name and name.strip()]
        if cluster_column not in fieldnames:
            raise ValueError(f"CSV is missing cluster column {cluster_column!r}")
        covariate_names = [
            name
            for name in fieldnames
            if name != cluster_column and name.lower() not in _ID_COLUMNS
        ]
        clusters: list[str] = []
        buckets: dict[str, list[str]] = {name: [] for name in covariate_names}
        for row in reader:
            raw_cluster = row.get(cluster_column)
            clusters.append("" if raw_cluster is None else raw_cluster.strip())
            for name in covariate_names:
                value = row.get(name)
                buckets[name].append("" if value is None else value.strip())
    if not clusters:
        raise ValueError("CSV has no data rows")
    if any(value == "" for value in clusters):
        raise ValueError(f"column {cluster_column} contains empty values")
    cluster_ids = _parse_covariate_column(cluster_column, clusters)
    covariates = {
        name: _parse_covariate_column(name, values) for name, values in buckets.items()
    }
    return cluster_ids, covariates


def cmd_cluster(args: argparse.Namespace) -> int:
    try:
        ratio = _parse_floats(args.ratio, "--ratio") if args.ratio else None
        categorical = _parse_name_list(args.categorical) if args.categorical else None
        if args.csv:
            cluster_ids, covariates = _load_cluster_csv(Path(args.csv), args.cluster)
            source = f"csv:{args.csv}"
        else:
            cluster_ids, covariates = _synthetic_cluster_data(args.n, args.clusters, args.seed)
            source = f"synthetic n={args.n} clusters={args.clusters} seed={args.seed}"
        result = cluster_randomization(
            cluster_ids,
            n_arms=args.arms,
            ratio=ratio,
            seed=args.seed,
        )
        report = None
        if covariates:
            report = cluster_balance_report(
                covariates,
                result.assignment,
                result.cluster_ids,
                arm_labels=result.arm_labels,
                categorical=categorical,
            )
    except (OSError, ValueError) as exc:
        print(f"cluster: {exc}", file=sys.stderr)
        return 2

    cluster_counts = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.cluster_counts)
    )
    arm_counts = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.arm_counts)
    )
    print(f"source: {source}")
    print(f"n: {result.n}  clusters: {result.n_clusters}  seed: {result.seed}")
    print(f"cluster counts: {cluster_counts}")
    print(f"arm counts: {arm_counts}")
    if report is not None:
        print("cluster-level covariate balance:")
        print(report)
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    try:
        ratio = _parse_floats(args.ratio, "--ratio") if args.ratio else None
        result = blocked_randomization(
            args.n,
            block_size=args.block_size,
            n_arms=args.arms,
            ratio=ratio,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"block: {exc}", file=sys.stderr)
        return 2

    counts = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.arm_counts)
    )
    target = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.block_target)
    )
    print(
        f"n: {result.n}  block size: {result.block_size}  "
        f"blocks: {result.n_blocks}  seed: {result.seed}"
    )
    print(f"per complete block: {target}")
    print(f"arm counts: {counts}")
    if result.n % result.block_size:
        final = result.block_arm_counts[-1]
        final_desc = ", ".join(
            f"{label}={count}" for label, count in zip(result.arm_labels, final)
        )
        print(f"final block: {final_desc}")
    return 0




def _add_switchback_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n", type=int, default=48, help="Number of sequential time periods (default: 48)")
    p.add_argument(
        "--block-length",
        type=int,
        default=4,
        help="Periods per switchback block (default: 4)",
    )
    p.add_argument("--arms", type=int, default=2, help="Number of arms (default: 2)")
    p.add_argument(
        "--ratio",
        default=None,
        help="Comma-separated allocation weights, one per arm (default: equal). Weights count blocks.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")


def cmd_switchback(args: argparse.Namespace) -> int:
    try:
        ratio = _parse_floats(args.ratio, "--ratio") if args.ratio else None
        result = switchback_randomization(
            args.n,
            block_length=args.block_length,
            n_arms=args.arms,
            ratio=ratio,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"switchback: {exc}", file=sys.stderr)
        return 2

    block_counts = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.block_counts)
    )
    arm_counts = ", ".join(
        f"{label}={count}" for label, count in zip(result.arm_labels, result.arm_counts)
    )
    print(
        f"n: {result.n}  block length: {result.block_length}  "
        f"blocks: {result.n_blocks}  seed: {result.seed}"
    )
    print(f"block counts: {block_counts}")
    print(f"arm counts: {arm_counts}")
    if result.n % result.block_length:
        final_arm = int(result.block_arms[-1])
        final_len = int(result.n - (result.n_blocks - 1) * result.block_length)
        print(
            f"final block: {result.arm_labels[final_arm]} "
            f"({final_len} period{'s' if final_len != 1 else ''})"
        )
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
    "bayesian": cmd_bayesian,
    "randomize": cmd_randomize,
    "block": cmd_block,
    "cluster": cmd_cluster,
    "switchback": cmd_switchback,
    "report": cmd_report,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
