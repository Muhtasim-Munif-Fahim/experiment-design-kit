"""Run a demo of experiment-design-kit and write a Markdown report."""

from __future__ import annotations

from pathlib import Path

from experiment_design_kit import (
    cohen_h,
    two_proportion_sample_size,
)
from experiment_design_kit.reporting import compose_demo_report


def main() -> None:
    seed = 42
    p_control, p_treatment = 0.10, 0.12
    h = cohen_h(p_control, p_treatment)
    n = two_proportion_sample_size(p_control, p_treatment, power=0.8).n_per_group_required
    print(f"proportion scenario: p_c={p_control} p_t={p_treatment} h={h:.4f} n/group={n:,}")

    report = compose_demo_report(seed=seed)
    out = Path("examples/output/demo_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
