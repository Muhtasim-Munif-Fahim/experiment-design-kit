# experiment-design-kit

A small, dependency-light Python toolkit for designing and evaluating A/B
experiments and controlled trials. It provides effect-size and sample-size
calculators (two-proportion z-test, two-sample and Welch's t-test),
statistical power analysis with configurable alpha/power, a minimum
detectable effect calculator, CUPED variance reduction for continuous
metrics, sequential always-valid inference (mSPRT p-values and
alpha-spending bounds) for optional stopping, Bayesian power and
sample-size planning for conversion tests (threshold or ROPE decisions
via simulation), and a seedable A/B test outcome simulator with
significance checking (chi-square and t-test).

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

# Sequential two-proportion A/B looks (incremental counts) with always-valid p-values.
experiment-design-kit sequential --method two-proportion \
  --control-successes 10,12,11 --control-totals 100,100,100 \
  --treatment-successes 18,20,19 --treatment-totals 100,100,100

# False-positive rates under peeking: naive repeated testing vs sequential bounds.
experiment-design-kit sequential --method peeking --looks 8 --n-per-look 200 --trials 400

# Bayesian power: chance of deciding P(B>A)>0.95 under an assumed 10% → 12% lift.
experiment-design-kit bayesian --method power --p1 0.10 --p2 0.12 --n 3841 --threshold 0.95

# Bayesian sample size for the same decision rule and a 80% decision rate.
experiment-design-kit bayesian --method sample-size --p1 0.10 --p2 0.12 --power 0.8 --threshold 0.95

# ROPE planning: probability of concluding practical equivalence (or a winner).
experiment-design-kit bayesian --method power --p1 0.10 --p2 0.10 --n 8000 \
  --decision rope --rope-lower -0.005 --rope-upper 0.005 --threshold 0.95

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
    bayesian_power_proportion,
    required_bayesian_sample_size,
)

h = cohen_h(0.10, 0.12)
plan = two_proportion_sample_size(0.10, 0.12, power=0.8)
power = power_proportion(0.10, 0.12, plan.n_per_group_required)
mde = minimum_detectable_effect_proportion(plan.n_per_group_required, 0.10)
outcome = run_proportion_ab_test(0.10, 0.12, plan.n_per_group_required, seed=42)

y, t, x = simulate_cuped_data(400, correlation=0.8, treatment_effect=0.5, seed=42)
cuped = cuped_adjust(y, t, x)

bayes = bayesian_power_proportion(0.10, 0.12, 3841, threshold=0.95, n_trials=1000, random_state=0)
plan = required_bayesian_sample_size(0.20, 0.28, target_power=0.8, threshold=0.95, random_state=0)
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

## Sequential testing (always-valid p-values and alpha-spending)

Peeking at a fixed-horizon p-value more than once inflates the Type I
error. Two lightweight corrections are included:

* **Always-valid p-values** from a mixture SPRT (Johari, Pekelis, Walsh).
  `p_av < alpha` at any look is a valid level-`alpha` rejection, even
  under continuous monitoring. The p-value process is the running minimum
  of `1/Λ_n`, where `Λ_n` is the normal-mixture likelihood ratio.
* **Lan-DeMets alpha-spending** for a *planned* sequence of looks.
  O'Brien-Fleming spends almost nothing early and almost all of `alpha`
  at the end; Pocock spends more evenly. A look is significant when the
  naive p-value is below the spending *increment* for that look.

```python
from experiment_design_kit import (
    sequential_two_proportion_test,
    sequential_two_sample_mean_test,
    simulate_peeking_fpr,
)

# Incremental conversion counts at three peeks (control vs treatment).
ab = sequential_two_proportion_test(
    control_successes=[10, 12, 11],
    control_totals=[100, 100, 100],
    treatment_successes=[18, 20, 19],
    treatment_totals=[100, 100, 100],
    alpha=0.05,
    boundary="always-valid",  # or "pocock" / "obrien-fleming"
)
print(ab.final_pvalue, ab.is_significant, ab.naive_significant)

# Snapshot means (cumulative mean, SD, n per arm).
means = sequential_two_sample_mean_test(
    control_means=[0.0, 0.02],
    control_sds=[1.0, 1.0],
    control_ns=[200, 400],
    treatment_means=[0.15, 0.18],
    treatment_sds=[1.0, 1.0],
    treatment_ns=[200, 400],
)

# Under H0, naive repeated testing over-rejects; sequential bounds do not.
fpr = simulate_peeking_fpr(
    n_looks=8, n_per_look=200, n_trials=400, metric="proportion", seed=0
)
print(fpr.naive_fpr, fpr.always_valid_fpr, fpr.pocock_fpr, fpr.obrien_fleming_fpr)
```

One-sample helpers `sequential_proportion_test` (vs `null_p`) and
`sequential_mean_test` (vs `null_mean`) cover non-A/B monitoring. Mixture
parameter `m` (default `0.01`) is the relative variance of the mixing
distribution: smaller `m` is more conservative. Spending functions are
evaluated at the information fraction `t = n / n_planned`.

## Bayesian power (conversion A/B tests)

Frequentist `power` / `sample-size` / `mde` answer "will a p-value cross
`alpha`?". Bayesian planning answers a different question: under an
assumed lift, how often will the Beta-Binomial analysis reach a *decision*?

Two rules are simulated (same Beta(1, 1) prior as `bayesian_ab_test`
unless you override it):

* **threshold** — decide when `P(θ_treatment > θ_control) ≥ threshold`
  or the reverse (default `threshold=0.95`).
* **ROPE** — decide when the posterior mass of the lift is at least
  `threshold` *above* the interval (treatment better), *below* it
  (control better), or *inside* it (practical equivalence). The interval
  is on the absolute difference `θ_t - θ_c` or on relative lift
  `(θ_t - θ_c) / θ_c`.

`power` is the Monte Carlo share of experiments that are not
inconclusive. The returned breakdown (`prob_treatment_wins`,
`prob_control_wins`, `prob_equivalent`) tells you *which* decision.

```python
from experiment_design_kit import (
    bayesian_ab_test,
    bayesian_power_proportion,
    required_bayesian_sample_size,
)

# Chance of P(B>A) > 0.95 (or P(A>B) > 0.95) at a planned n.
pwr = bayesian_power_proportion(
    0.10, 0.12, n_per_group=3841, threshold=0.95, n_trials=1000, random_state=0
)
print(pwr.power, pwr.prob_treatment_wins, pwr.prob_inconclusive)

# Smallest n per group with decision rate ≥ 0.8.
plan = required_bayesian_sample_size(
    0.10, 0.12, target_power=0.8, threshold=0.95, random_state=0
)
print(plan.n_per_group, plan.achieved_power)

# Equivalence planning: assumed lift of 0, ROPE of ±0.5 percentage points.
eq = bayesian_power_proportion(
    0.10,
    0.10,
    n_per_group=8000,
    decision="rope",
    rope=(-0.005, 0.005),
    threshold=0.95,
    n_trials=1000,
    random_state=0,
)
print(eq.prob_equivalent, eq.power)

# Same decision rule on observed counts.
observed = bayesian_ab_test(80, 1000, 110, 1000, random_state=0)
print(observed.prob_treatment_better, observed.expected_lift)
```

`--lift` on the CLI is a relative alternative to `--p2`
(`p2 = p1 * (1 + lift)`). Threshold-rule sample-size search requires a
non-zero assumed lift: under equal rates the chance of declaring a
winner goes to 0 as `n` grows. ROPE search is valid at a zero lift
because it can conclude equivalence.

