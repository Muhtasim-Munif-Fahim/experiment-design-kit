# experiment-design-kit

A small, dependency-light Python toolkit for designing and evaluating A/B
experiments and controlled trials. It provides effect-size and sample-size
calculators (two-proportion z-test, two-sample and Welch's t-test),
statistical power analysis with configurable alpha/power, a minimum
detectable effect calculator, and a seedable A/B test outcome simulator
with significance checking (chi-square and t-test).

## Install

```bash
pip install -e .
```

## CLI quick start

```bash
# Required sample size per group for a conversion lift from 10% to 12%.
experiment-design-kit sample-size --method proportion --p1 0.10 --p2 0.12

# Achieved power for a Cohen's d = 0.5 at n = 64 per group.
experiment-design-kit power --method two-sample --d 0.5 --n 64

# Minimum detectable effect (raw units) for a continuous metric.
experiment-design-kit mde --method two-sample --n 64 --sd 10

# Simulate one conversion-rate A/B test and test it with chi-square.
experiment-design-kit simulate --metric proportion --control 0.10 --treatment 0.14 --n 3841

# Run the full demo workflow and write a Markdown report.
experiment-design-kit report -o examples/output/demo_report.md
```

## Library quick start

```python
from experiment_design_kit import (
    cohen_h,
    two_proportion_sample_size,
    power_proportion,
    minimum_detectable_effect_proportion,
    run_proportion_ab_test,
)

h = cohen_h(0.10, 0.12)
plan = two_proportion_sample_size(0.10, 0.12, power=0.8)
power = power_proportion(0.10, 0.12, plan.n_per_group_required)
mde = minimum_detectable_effect_proportion(plan.n_per_group_required, 0.10)
outcome = run_proportion_ab_test(0.10, 0.12, plan.n_per_group_required, seed=42)
```

See `examples/run_demo.py` for a complete end-to-end demo and `tests/` for
the unit-test contract.
