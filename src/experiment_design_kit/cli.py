"""Command-line interface for experiment-design-kit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .mde import minimum_detectable_effect, minimum_detectable_effect_proportion, minimum_detectable_effect_raw
from .power import power_one_sample, power_proportion, power_two_sample
from .reporting import compose_demo_report
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
    "report": cmd_report,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _COMMANDS[args.command](args)
