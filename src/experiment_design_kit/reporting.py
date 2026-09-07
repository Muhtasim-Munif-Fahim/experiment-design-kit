"""Markdown report rendering and the end-to-end demo workflow.

The demo ties the four calculation areas together into a single planning
scenario: a conversion-rate (proportion) A/B test and a continuous-metric
(revenue-style) A/B test. For each scenario the report records the effect
size, the planned sample size, the achieved power, the minimum detectable
effect, and a seedable simulated outcome with its significance test.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mde import (
    MDEResult,
    ProportionMDE,
    minimum_detectable_effect,
    minimum_detectable_effect_proportion,
    minimum_detectable_effect_raw,
)
from .power import power_proportion, required_sample_size
from .simulation import ABTestResult, run_continuous_ab_test, run_proportion_ab_test
from .stats import cohen_h, cohens_d, two_proportion_sample_size


@dataclass(frozen=True)
class ProportionScenario:
    control: float
    treatment: float
    alpha: float
    power_target: float
    seed: int
    effect_size_h: float
    sample_size: int
    achieved_power: float
    mde: ProportionMDE
    simulation: ABTestResult


@dataclass(frozen=True)
class ContinuousScenario:
    control: float
    sd: float
    treatment_effect: float
    alpha: float
    power_target: float
    seed: int
    effect_size_d: float
    sample_size: int
    mde: MDEResult
    mde_raw: float
    simulation: ABTestResult


@dataclass(frozen=True)
class ReportData:
    proportion: ProportionScenario
    continuous: ContinuousScenario
    seed: int


def compose_demo_report(seed: int = 42) -> str:
    """Run the full design workflow and return a rendered Markdown report."""
    alpha = 0.05
    power_target = 0.8

    p_c, p_t = 0.10, 0.12
    h = cohen_h(p_c, p_t)
    n_p = two_proportion_sample_size(p_c, p_t, alpha=alpha, power=power_target).n_per_group_required
    prop = ProportionScenario(
        control=p_c,
        treatment=p_t,
        alpha=alpha,
        power_target=power_target,
        seed=seed,
        effect_size_h=h,
        sample_size=n_p,
        achieved_power=power_proportion(p_c, p_t, n_p, alpha=alpha),
        mde=minimum_detectable_effect_proportion(n_p, p_c, alpha=alpha, power=power_target),
        simulation=run_proportion_ab_test(p_c, p_t, n_p, alpha=alpha, seed=seed),
    )

    m_c, sd, eff = 100.0, 15.0, 5.0
    d = cohens_d(m_c + eff, m_c, sd, sd)
    n_t = required_sample_size(d, alpha=alpha, power=power_target, kind="two-sample").n_per_group_required
    cont = ContinuousScenario(
        control=m_c,
        sd=sd,
        treatment_effect=eff,
        alpha=alpha,
        power_target=power_target,
        seed=seed,
        effect_size_d=d,
        sample_size=n_t,
        mde=minimum_detectable_effect(n_t, alpha=alpha, power=power_target, kind="two-sample"),
        mde_raw=minimum_detectable_effect_raw(n_t, alpha=alpha, power=power_target, sigma=sd, kind="two-sample"),
        simulation=run_continuous_ab_test(m_c, sd, eff, n_t, alpha=alpha, seed=seed),
    )

    return render_report(ReportData(proportion=prop, continuous=cont, seed=seed))


def render_report(data: ReportData) -> str:
    """Render a :class:`ReportData` instance as a Markdown string."""
    lines: list[str] = []
    lines.append("# Experiment design report")
    lines.append("")
    lines.append(f"- Significance level (alpha): {data.proportion.alpha:.2f}")
    lines.append(f"- Target power (1 - beta): {data.proportion.power_target:.2f}")
    lines.append(f"- Random seed: {data.seed}")
    lines.append("")

    lines.append("## Proportion scenario (conversion rate)")
    lines.append("")
    p = data.proportion
    lines.append(f"- Control rate: {p.control:.3f}  ")
    lines.append(f"- Treatment rate: {p.treatment:.3f}  ")
    lines.append(f"- Absolute lift: {p.treatment - p.control:.3f}")
    lines.append(f"- Cohen's h (effect size): {p.effect_size_h:.4f}")
    lines.append(f"- Required sample size per group: {p.sample_size:,}")
    lines.append(f"- Achieved power at that n: {p.achieved_power:.4f}")
    lines.append(
        f"- Minimum detectable lift at n={p.sample_size:,}: "
        f"{p.mde.lift_absolute:.4f} "
        f"({p.mde.lift_relative * 100:.1f}% relative, h={p.mde.effect_size_h:.4f})"
    )
    lines.append("")
    lines.append("### Simulated outcome")
    lines.append("")
    sim = p.simulation
    lines.append("| Control rate | Treatment rate | Observed lift | chi-square | p-value | Significant |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    lines.append(
        f"| {sim.control_estimate:.4f} | {sim.treatment_estimate:.4f} | "
        f"{sim.lift_absolute:+.4f} | {sim.statistic:.3f} | {sim.p_value:.4f} | "
        f"{'yes' if sim.significant else 'no'} |"
    )
    lines.append("")

    lines.append("## Continuous scenario (revenue per user)")
    lines.append("")
    c = data.continuous
    lines.append(f"- Control mean: {c.control:.2f}  ")
    lines.append(f"- Per-group SD: {c.sd:.2f}  ")
    lines.append(f"- Planned treatment effect: {c.treatment_effect:.2f}")
    lines.append(f"- Cohen's d (effect size): {c.effect_size_d:.4f}")
    lines.append(f"- Required sample size per group: {c.sample_size:,}")
    lines.append(
        f"- Minimum detectable raw effect at n={c.sample_size:,}: {c.mde_raw:.4f} "
        f"(d={c.mde.effect_size:.4f})"
    )
    lines.append("")
    lines.append("### Simulated outcome")
    lines.append("")
    simc = c.simulation
    lines.append("| Control mean | Treatment mean | Observed lift | t-statistic | p-value | Significant |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    lines.append(
        f"| {simc.control_estimate:.4f} | {simc.treatment_estimate:.4f} | "
        f"{simc.lift_absolute:+.4f} | {simc.statistic:.3f} | {simc.p_value:.4f} | "
        f"{'yes' if simc.significant else 'no'} |"
    )
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    prop_sig = p.simulation.significant
    cont_sig = c.simulation.significant
    lines.append(
        f"- Proportion test (chi-square) {'detected' if prop_sig else 'did not detect'} "
        f"a significant difference at alpha={p.alpha:.2f} "
        f"(p={p.simulation.p_value:.4f})."
    )
    lines.append(
        f"- Continuous test (Welch t-test) {'detected' if cont_sig else 'did not detect'} "
        f"a significant difference at alpha={c.alpha:.2f} "
        f"(p={c.simulation.p_value:.4f})."
    )
    lines.append(
        "- Both scenarios were designed for ~80% power; observed significance "
        "depends on the realised random seed."
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ProportionScenario",
    "ContinuousScenario",
    "ReportData",
    "compose_demo_report",
    "render_report",
]
