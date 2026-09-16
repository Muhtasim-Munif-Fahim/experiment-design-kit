# experiment-design-kit

A small, dependency-light Python toolkit for designing and evaluating A/B
experiments and controlled trials. It provides effect-size and sample-size
calculators (two-proportion z-test, two-sample and Welch's t-test),
statistical power analysis with configurable alpha/power, a minimum
detectable effect calculator, CUPED variance reduction for continuous
metrics, and a seedable A/B test outcome simulator with significance
checking (chi-square and t-test).

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

# CUPED on synthetic correlated pre/post data (reports theta and variance reduction).
experiment-design-kit cuped --n 500 --correlation 0.7 --effect 0.5

# CUPED on a CSV with columns outcome,treatment,covariate.
experiment-design-kit cuped --csv path/to/metrics.csv --fit-on pooled

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
    cuped_adjust,
    simulate_cuped_data,
)

h = cohen_h(0.10, 0.12)
plan = two_proportion_sample_size(0.10, 0.12, power=0.8)
power = power_proportion(0.10, 0.12, plan.n_per_group_required)
mde = minimum_detectable_effect_proportion(plan.n_per_group_required, 0.10)
outcome = run_proportion_ab_test(0.10, 0.12, plan.n_per_group_required, seed=42)

y, t, x = simulate_cuped_data(400, correlation=0.8, treatment_effect=0.5, seed=42)
cuped = cuped_adjust(y, t, x)
```

See `examples/run_demo.py` for a complete end-to-end demo and `tests/` for
the unit-test contract.

## CUPED (pre-experiment covariates)

For continuous metrics, CUPED (Controlled-experiment Using Pre-Experiment
Data) estimates `θ = Cov(Y, X) / Var(X)` from a pre-period covariate `X`
and forms the adjusted metric `Y_cv = Y - θ (X - E[X])`. The CUPED
treatment effect is the difference in adjusted means. When `X` is
correlated with `Y`, `Var(Y_cv) ≈ Var(Y) (1 - ρ²)`, so the reported
variance reduction is approximately `ρ²`.

```python
from experiment_design_kit import (
    adjust_metric,
    cuped_adjust,
    estimate_theta,
    simulate_cuped_data,
)

outcomes, treatment, pre_period = simulate_cuped_data(
    n_per_group=400,
    correlation=0.8,
    treatment_effect=0.5,
    seed=42,
)
theta = estimate_theta(outcomes, pre_period)
adjusted = adjust_metric(outcomes, pre_period, theta=theta)
result = cuped_adjust(outcomes, treatment, pre_period)
print(result.theta, result.variance_reduction, result.effect, result.raw_effect)
```

`cuped_adjust(..., fit_on="control")` estimates θ from the control arm
only. The default `fit_on="pooled"` is the usual choice when `X` is
pre-experiment and therefore independent of assignment. Pass a CSV with
columns `outcome`, `treatment`, and `covariate` to the `cuped` CLI to
run the same adjustment on your own data.

