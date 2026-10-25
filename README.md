# experiment-design-kit

A small, dependency-light Python toolkit for designing and evaluating A/B
experiments and controlled trials. It provides effect-size and sample-size
calculators (two-proportion z-test, two-sample and Welch's t-test),
statistical power analysis with configurable alpha/power, a minimum
detectable effect calculator, CUPED variance reduction for continuous
metrics, sequential always-valid inference (mSPRT p-values and
alpha-spending bounds) for optional stopping, Bayesian power and
sample-size planning for conversion tests (threshold or ROPE decisions
via simulation), stratified randomization on categorical covariates or
quantile bins with covariate balance diagnostics (standardized mean
differences and chi-square), blocked (permuted-block) randomization
with a fixed block size, cluster randomization that assigns whole
clusters to arms, ICC-aware cluster-randomized power and sample size
(design effect, Donner & Klar clusters per arm, ANOVA ICC estimation),
switchback / time-based block randomization,
sample ratio mismatch (SRM) checks (fixed-horizon chi-square / G-test
and an anytime-valid sequential Bayes factor),
non-inferiority and TOST equivalence tests with sample-size and power
planning (means and proportions),
multiple-testing corrections (Bonferroni, Šidák, Holm, Hochberg,
Benjamini-Hochberg / Yekutieli) with A/B/n sample-size planning,
and a seedable A/B test outcome simulator
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

# Stratified assignment on a synthetic region factor and 4 quantile bins of a score.
experiment-design-kit randomize --n 240 --seed 0 --bins 4

# Same workflow on a CSV (one row per unit). Continuous columns need --bins.
experiment-design-kit randomize --csv path/to/units.csv --bins pre_metric=4 --categorical region

# Permuted blocks of size 4 (2k with k=2): each block has 2 control and 2 treatment.
experiment-design-kit block --n 100 --block-size 4 --seed 0

# 1:2 allocation. The block size must be a multiple of 3.
experiment-design-kit block --n 90 --block-size 6 --ratio 1,2 --seed 0

# Cluster randomization: whole clusters share an arm. The ratio counts clusters.
experiment-design-kit cluster --n 120 --clusters 12 --seed 0

# Switchback: consecutive time blocks share an arm. The ratio counts blocks.
experiment-design-kit switchback --n 48 --block-length 4 --seed 0

# Same workflow on a CSV. Balance is computed on one row per cluster.
experiment-design-kit cluster --csv path/to/units.csv --cluster school --categorical region

# Sample ratio mismatch: did a planned 50/50 split actually receive 50/50?
experiment-design-kit srm --counts 50421,49579

# Same check for a 10% holdout, or sequentially (incremental counts per look).
experiment-design-kit srm --counts 9000,1000 --ratio 90,10
experiment-design-kit srm --looks "510,490;530,470;560,440"

# Non-inferiority sample size: a 0.5-unit margin on a metric with SD 2.
experiment-design-kit equivalence --margin 0.5 --sd 2

# TOST equivalence on observed conversions with a ±2 pp margin.
experiment-design-kit equivalence --design equivalence --mode test --margin 0.02 --counts 3000,10000,3050,10000

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


## Switchback randomization

Switchback designs assign *time* rather than units. Periods `0 .. n-1` are
split into blocks of `block_length` consecutive periods; each block draws
one arm and every period inside the block shares it. `ratio` counts blocks
(the same largest-remainder rule as cluster randomization). A short final
block is allowed when `n` is not a multiple of `block_length`.

```python
from experiment_design_kit import switchback_randomization

assigned = switchback_randomization(48, block_length=4, ratio=(1, 1), seed=0)
print(assigned.block_counts)  # blocks per arm
print(assigned.arm_counts)    # periods per arm
print(assigned.assignment[:8])
```

## Sample ratio mismatch (SRM)

Before reading any treatment effect, check that each arm received the
traffic the design promised. A gap between the planned split and the
observed counts (an SRM) usually means assignment, logging, bot
filtering, or a redirect is losing units unevenly, which biases every
metric downstream.

* `sample_ratio_mismatch(counts, ratio=None, alpha=0.001)` — Pearson
  chi-square (or `method="g-test"`) goodness-of-fit of the observed
  counts to the planned ratio, for two or more arms. `alpha=0.001` is
  the conventional SRM alarm level. Pearson residuals per arm show which
  arm is drifting (`result.worst_arm`).
* `srm_from_assignment(assignment, ratio=None)` — the same check on an
  assignment object from this kit or a raw array of arm ids (e.g. the
  arm column of the units present in your analysis table).
* `sequential_srm_test(looks, ratio=None, alpha=0.05)` — anytime-valid
  monitoring with a Dirichlet-multinomial Bayes factor (Lindon & Malek,
  2020). The running `p = min(1, 1/BF)` may be checked after every batch
  of traffic without inflating the false-alarm rate (Ville's inequality).
  Looks are incremental counts unless `cumulative=True`.

```python
from experiment_design_kit import (
    sample_ratio_mismatch,
    sequential_srm_test,
    srm_from_assignment,
)

check = sample_ratio_mismatch([50421, 49579])          # planned 50/50
print(check.p_value, check.mismatch, check.worst_arm)  # 0.0078 False control

holdout = sample_ratio_mismatch([9000, 1000], ratio=(90, 10))
three = sample_ratio_mismatch([1000, 1000, 820], arm_labels=["A", "B", "C"])

monitor = sequential_srm_test([[510, 490], [530, 470], [560, 440]])
print(monitor.final_p_value, monitor.first_mismatch_look)
```

## Ratio metrics (delta method)

Click-through rate, revenue per session and average order value are ratios
of per-user sums. Users are randomized, but page views or sessions inside a
user are correlated, so a naive per-event standard error is too small and
inflates false positives. `delta_method_ratio_test` (Deng, Knoblich & Lu,
2018) linearises `R = mean(Y) / mean(X)` over randomization units and
returns the absolute difference with its delta-method SE, z, p-value and
CI, plus the relative lift `R_t / R_c - 1` with its own SE and CI.
`ratio_metric_variance(numerator, denominator)` gives the single-arm
`(ratio, variance)`.

```python
from experiment_design_kit import delta_method_ratio_test

# one entry per user: clicks and page views
res = delta_method_ratio_test(clicks_c, views_c, clicks_t, views_t, alpha=0.05)
print(res.ratio_control, res.ratio_treatment)
print(res.absolute_diff, (res.ci_low, res.ci_high), res.p_value)
print(res.relative_lift, (res.relative_ci_low, res.relative_ci_high))
```

## Non-inferiority and equivalence (TOST)

A non-significant superiority test is not evidence that two variants are
"the same". When the goal is to show that a cheaper model, a refactor or a
simplified flow is *not worse*, flip the burden of proof:

* **Non-inferiority** tests `H0: delta <= -margin` (one-sided, default
  `alpha = 0.025`), with `delta = treatment - control`. Use
  `higher_is_better=False` for latency / cost metrics.
* **Equivalence** uses Schuirmann's two one-sided tests: both
  `H0: delta <= lower` and `H0: delta >= upper` must be rejected at `alpha`,
  i.e. the `1 - 2*alpha` CI must sit inside the margins (symmetric
  `±margin`, or asymmetric via `upper_margin`).

Means use Welch's t; proportions use the unpooled Wald z. Planning uses the
normal approximation: closed form for non-inferiority, and the smallest `n`
whose TOST power `Phi((U - delta)/se - z) + Phi((delta - L)/se - z) - 1`
reaches the target for equivalence (`ratio = n_treatment / n_control`).

```python
from experiment_design_kit import (
    equivalence_sample_size, noninferiority_sample_size,
    noninferiority_means, tost_proportions,
)

noninferiority_sample_size(0.5, sd=2.0).n_control          # 252 per arm
equivalence_sample_size(0.05, p_control=0.30).n_control    # 1439 per arm

res = tost_proportions(3000, 10000, 3050, 10000, margin=0.02)
print(res.equivalent, (res.ci_low, res.ci_high), res.p_value)

ni = noninferiority_means(control_latency, new_latency, margin=5.0,
                          higher_is_better=False)
print(ni.noninferior, ni.confidence_bound)
```

## Multiple testing (many metrics, segments or variants)

Running ten independent tests at `alpha = 0.05` gives about a 40% chance of
at least one false positive (`family_wise_error_rate(0.05, 10)`).
`adjust_pvalues(p_values, method, alpha)` returns adjusted p-values in the
input order, with `rejected` meaning `adjusted <= alpha`:

| method | controls | notes |
| --- | --- | --- |
| `bonferroni`, `sidak` | FWER | single-step |
| `holm`, `holm-sidak` | FWER | step-down; never less powerful than the single-step versions |
| `hochberg` | FWER | step-up; needs independence / positive dependence |
| `bh` | FDR | Benjamini-Hochberg (independence / PRDS) |
| `by` | FDR | Benjamini-Yekutieli (any dependence) |

`multi_arm_sample_size(p_control, [p_t1, p_t2, ...], correction="bonferroni")`
plans an A/B/n test that compares each variant with control. It splits
`alpha` across the comparisons and gives every arm the largest
per-comparison sample size.

```python
from experiment_design_kit import adjust_pvalues, multi_arm_sample_size

res = adjust_pvalues([0.01, 0.04, 0.03, 0.005], method="holm")
print(res.adjusted, res.rejected)        # (0.03, 0.06, 0.06, 0.02) (True, False, False, True)

plan = multi_arm_sample_size(0.10, [0.12, 0.13], alpha=0.05, power=0.8)
print(plan.alpha_per_comparison, plan.n_per_arm_required, plan.n_total_required)
```

```bash
experiment-design-kit multiple-testing --pvalues 0.01,0.04,0.03,0.005 --method bh
experiment-design-kit multiple-testing --treatments 0.12,0.13 --control 0.10 --correction sidak
```

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

## Stratified randomization and covariate balance

Stratified randomization keeps arms aligned on baseline covariates.
Strata are the Cartesian product of categorical factors. A continuous
covariate is cut into quantile bins first and those bins become a
factor. Inside each stratum, units are shuffled into arms so the counts
follow the allocation ratio (equal by default) to the nearest unit.
Tied remainders are broken at random, and the whole draw is fixed by
`seed`.

Each observed combination of factors is balanced: arm counts in that
cell match the ratio to the nearest unit. With one factor, every level
is balanced that tightly. With several crossed factors, a one-way
margin can drift by up to one unit per combination inside the margin.
Quantile bins are balanced the same way. The raw continuous covariate
is not forced into exact balance; its standardized mean difference is
usually small after binning.

`balance_report` compares a realized assignment with the reference arm
(arm `0` by default):

* **Continuous** covariates use Austin's standardized mean difference,

  `(mean_arm - mean_reference) / sqrt((var_arm + var_reference) / 2)`.

* **Categorical** covariates (and any name passed in `categorical`,
  including integer codes) get a Pearson chi-square test of the
  level-by-arm table plus the SMD of each level indicator. The row's
  `max_abs_smd` is the largest absolute level SMD.

```python
import numpy as np

from experiment_design_kit import balance_report, stratified_randomization

rng = np.random.default_rng(0)
region = rng.choice(["us", "eu", "apac"], size=240, p=[0.5, 0.3, 0.2])
pre_metric = rng.normal(size=240)

assigned = stratified_randomization(
    {"region": region, "pre_metric": pre_metric},
    n_bins={"pre_metric": 4},
    seed=0,
)
print(assigned.arm_counts, assigned.n_strata)

report = balance_report(
    {"region": region, "pre_metric": pre_metric},
    assigned.assignment,
    categorical=["region"],
)
print(report)
```

`assigned.factors["pre_metric"]` is the bin index used for
stratification, and `assigned.bin_edges["pre_metric"]` is the cut
points. `assigned.stratum_arm_counts` is the arm counts inside each
stratum. Pass `ratio=(1, 2)` for a 1:2 allocation, or `n_arms=3` for
more than two arms.

The `randomize` CLI prints the same diagnostics. Without `--csv` it
builds a skewed categorical `region` and a normal `score`, then bins
`score` (default 4 bins). A CSV uses every column except `id`,
`unit_id`, and `user_id`. Integer-looking columns are categories unless
named in `--bins` (for example `--bins age=4`); pass `--categorical` to
force a chi-square test for numeric codes.

## Blocked randomization

Blocked randomization (permuted blocks) keeps the arm counts balanced
in enrollment order. Units are filled from index 0. Every complete
block has a fixed size and a fixed number of each arm; only the order
inside the block is random. `seed` fixes that order.

For equal treatment and control the block size is `2k`. A block of
size 2 (`k = 1`) is one control and one treatment. A block of size 4
(`k = 2`) is two of each. A block of size 6 is three of each. An odd
size is rejected because a complete block would not be able to split
evenly. Pass `ratio=(1, 2)` for a 1:2 scheme; the block size must then
be a multiple of 3 (`3`, `6`, `9`, ...), and each complete block holds
one share of control for every two shares of treatment. `n_arms=3`
with equal weights uses a multiple of 3.

After each complete block the cumulative counts sit on the ratio. If
`n` is not a multiple of the block size, the last block is shorter and
is filled with the same ratio, as evenly as integer counts allow. For
an equal allocation the overall arm counts then differ by at most one
unit.

Smaller blocks restore balance sooner. Larger blocks leave more
assignments inside the block unpredictable. Stratified randomization
is what aligns the covariate mix across arms. A blocked assignment
can still be passed to `balance_report` when those covariates are
available.

```python
import numpy as np

from experiment_design_kit import balance_report, blocked_randomization

assigned = blocked_randomization(100, block_size=4, seed=0)
print(assigned.arm_counts)            # (50, 50)
print(assigned.block_target)          # (2, 2) control, treatment
print(assigned.block_arm_counts[0])   # (2, 2) inside the first block

unequal = blocked_randomization(90, block_size=6, ratio=(1, 2), seed=1)
print(unequal.arm_counts)             # (30, 60)
print(unequal.block_target)           # (2, 4)

region = np.array(["us", "eu", "apac"] * 30)
print(balance_report({"region": region}, unequal.assignment))
```

The `block` CLI prints the same counts. `--block-size` is required.
`--ratio` is a comma-separated list of positive weights, and `--seed`
reproduces the within-block order.

## Cluster randomization

Cluster randomization assigns every unit in a cluster to the same arm.
`cluster_ids` is one id per unit (a school, a clinic, a market). The
distinct ids are the randomization units. They are sorted by id, then
shuffled into arms, and that arm is written onto every unit in the
cluster. Sorting by id keeps the cluster-to-arm mapping when the rows
are reordered. `seed` fixes the shuffle.

`ratio` counts clusters. `ratio=(1, 2)` puts about one cluster on
control for every two clusters on treatment. Unequal cluster sizes can
leave the unit counts uneven: `arm_counts` is units and `cluster_counts`
is clusters. Equal allocation keeps the cluster counts within one
cluster of each other.

`cluster_balance_report` checks covariate balance at the cluster level.
Each cluster is one row. A covariate that is constant within every
cluster keeps that value. A numeric covariate that varies within a
cluster is replaced by the within-cluster mean. A categorical covariate
that varies within a cluster is rejected. The report is the usual
standardized-mean-difference and chi-square table, and its arm counts
are cluster counts.

```python
import numpy as np

from experiment_design_kit import cluster_balance_report, cluster_randomization

cluster_ids = np.repeat(np.arange(12), 10)  # 12 clusters, 10 units each
assigned = cluster_randomization(cluster_ids, ratio=(1, 2), seed=0)
print(assigned.cluster_counts)  # (4, 8) clusters
print(assigned.arm_counts)      # (40, 80) units

region = np.array(["north", "south", "east"])[cluster_ids % 3]
print(cluster_balance_report(
    {"region": region},
    assigned.assignment,
    assigned.cluster_ids,
    categorical=["region"],
))
```

The `cluster` CLI prints the same counts and, when covariates are
present, the cluster-level balance report. Without `--csv` it builds
`n` units in `--clusters` groups, with a cluster-level `region` and a
unit-level `score` (balanced on the cluster mean). A CSV uses
`--cluster` for the id column (default `cluster`) and treats the other
columns, except `id`, `unit_id`, and `user_id`, as covariates.


## Cluster-randomized power and sample size (ICC-aware)

When whole clusters are randomized, correlated outcomes inside a cluster
inflate the variance by the design effect
`DE = 1 + ((cv² + 1)·m − 1)·ICC` (Donner & Klar 2000; Eldridge et al. 2006 for
variable cluster sizes). `cluster_sample_size` inflates the individually
randomized n by `DE` and converts it to clusters per arm. `method="normal"` is
the closed-form Donner & Klar count; `method="t"` (default) bumps it until the
noncentral-t power with `2k − 2` cluster-level df reaches the target, which
matters when only a few clusters are affordable.

```python
from experiment_design_kit import cluster_sample_size, cluster_power, cluster_mde, estimate_icc

plan = cluster_sample_size(cluster_size=20, icc=0.05, effect_size=0.3)
print(plan.design_effect, plan.clusters_per_arm, plan.individuals_per_arm, plan.achieved_power)

binary = cluster_sample_size(50, 0.01, p1=0.10, p2=0.15, kind="proportions")
print(cluster_power(12, 20, 0.05, effect_size=0.4), cluster_mde(12, 20, 0.05))

est = estimate_icc(values, cluster_ids)   # one-way ANOVA ICC + Searle F-based CI
print(est.icc, est.ci_lower, est.ci_upper)
```

```bash
experiment-design-kit cluster-power --cluster-size 20 --icc 0.05 --d 0.3
experiment-design-kit cluster-power --mode power --cluster-size 20 --icc 0.05 --p1 0.1 --p2 0.15 --clusters 30
experiment-design-kit cluster-power --mode mde --cluster-size 20 --icc 0.05 --clusters 10
```

## Two-level factorial designs (full, fractional, fold-over, Lenth analysis)

When several factors may matter and interact, a `2^k` factorial tests them all in one experiment. A `2^(k-p)` fraction runs `2^(k-p)` of the combinations, at the price of aliasing some effects with each other.

```python
from experiment_design_kit import factorial_effects, fold_over, fractional_factorial, full_factorial

design = fractional_factorial(6, p=2)       # minimum-aberration E=ABC, F=BCD
design.resolution, design.defining_words     # 4, ('ABCE', 'ADEF', 'BCDF')
design.aliases["AB"]                         # ('CE', 'ACDF', 'BDEF')
custom = fractional_factorial(5, ["D=AB", "E=-AC"])
folded = fold_over(fractional_factorial(7, p=4))   # resolution III -> IV, 16 runs
rows = design.to_levels({"A": ("old", "new"), ...})  # coded runs -> real settings
order = design.randomized_order(seed=1)

result = factorial_effects(full_factorial(4), y)   # y in standard order (or runs x replicates)
result.ranked()[:3], result.significant
result.std_error, result.margin_of_error, result.simultaneous_margin   # Lenth PSE / ME / SME
```

- Designs use coded `-1/+1` columns in standard (Yates) order. The factor letters are `A, B, C, ...`, skipping `I`.
- Generators are written like `D=ABC` or `E=-AC`. The design reports the full defining relation, the word-length pattern `(A3, A4, ...)`, the resolution, and the aliases of every main effect and two-factor interaction. With `p=` you get the standard minimum-aberration table for k ≤ 8.
- `factorial_effects` estimates one effect per alias class, where `effect = mean(y|+) - mean(y|-)`.
  - **Unreplicated designs:** it uses Lenth's (1989) pseudo standard error, and flags effects above the margin of error `ME = t(1-α/2, m/3) · PSE`. On Montgomery's filtration example (2^4) it reproduces PSE = 2.625 and flags A, C, D, AC and AD.
  - **Replicated designs, or with `max_order=`:** it uses t tests on the residual mean square.
  - `half_normal_scores()` gives the points for a Daniel plot.
- `fold_over(design)` appends the mirror-image runs, which breaks every main-effect/2FI alias. `fold_over(design, factor="A")` de-aliases A and all of its interactions.

```bash
experiment-design-kit factorial --factors 5 --fraction 1
experiment-design-kit factorial --factors 7 --generators D=AB,E=AC,F=BC,G=ABC --fold-over
experiment-design-kit factorial --factors 4 --responses 45,71,48,65,68,60,80,65,43,100,45,104,75,86,70,96
```

## Screening and response-surface designs (Plackett–Burman, CCD, Box–Behnken)

When many factors are candidates, a **Plackett–Burman** design screens up to `N − 1` of them in `N` runs, where `N` is a multiple of 4. Once the vital few are known, a **central composite** or **Box–Behnken** design supports a full second-order model, so you can locate an optimum.

```python
from experiment_design_kit import (box_behnken, central_composite, fit_quadratic,
                                   plackett_burman, scaled_prediction_variance, screening_effects)

pb = plackett_burman(11)                   # 12 runs, published cyclic generator, X'X = 12 I
screen = screening_effects(pb, y12)        # main effects; Lenth's method when saturated
ccd = central_composite(3)                 # rotatable: alpha = 8 ** 0.25 = 1.682, 8 + 6 + 8 centre runs
ccd_o = central_composite(3, alpha="orthogonal", center=(2, 2))
ccf = central_composite(3, face="faced")   # axial points on the cube faces
bbd = box_behnken(3)                       # 12 edge midpoints + 3 centre runs, no corners
fit = fit_quadratic(bbd, y15)
fit.coefficients["A^2"], fit.r_squared
fit.nature, fit.stationary_point, fit.stationary_response   # 'maximum', x_s, y(x_s)
fit.eigenvalues, fit.predict([[0.2, -0.1, 0.0]])
scaled_prediction_variance(ccd, points)    # constant on spheres for a rotatable design
```

- `plackett_burman(k, n_runs=None)` uses the published cyclic generators for N = 12, 20 and 24 (Plackett & Burman 1946). Other sizes use Paley's construction (N − 1 prime and ≡ 3 mod 4: 28 is not covered, 44 is) or Sylvester doubling (8, 16, 32, 40, 48, …). Every column is balanced and orthogonal.
- `central_composite` builds a `2^(k-p)` cube plus `2k` axial points plus centre runs. `alpha` can be:
  - `"rotatable"`: `F^(1/4)`;
  - `"orthogonal"`: `α² = (√(F·N) − F)/2`, which makes the pure quadratic columns orthogonal;
  - `"face"`;
  - a number.
  
  It also supports the circumscribed, inscribed and faced variants.
- `fit_quadratic` fits `y = b0 + x'b + x'Bx` by least squares. It reports R², adjusted R² and RMSE, then does canonical analysis: the stationary point `−B⁻¹b/2`, the eigenvalues of `B`, and whether the point is a maximum, minimum, saddle or ridge.

```bash
experiment-design-kit screening --factors 7 --responses 10,12,9,15,11,14,8,13
experiment-design-kit response-surface --factors 3 --design bbd --responses 69,75,73,80,62,70,68,76,61,72,66,74,82,81,83
experiment-design-kit response-surface --factors 2 --design ccd --alpha orthogonal
```

## Space-filling and D-optimal designs

`experiment_design_kit.optimal_design` covers run budgets and regions that
classical factorial/RSM families don't fit:

- `latin_hypercube(n, k)` — McKay–Beckman–Conover LHS on `[0, 1)^k` (one run per stratum per factor).
- `maximin_latin_hypercube(n, k)` — within-column swap search minimising the Morris–Mitchell `phi_p` criterion (≈ maximising the smallest inter-point distance).
- `centered_l2_discrepancy(x)` — Hickernell's CD² (matches `scipy.stats.qmc.discrepancy(method="CD")`), plus `maximin_distance` and `phi_p`.
- `d_optimal_design(n_runs, candidates, model=...)` — Fedorov exchange with random restarts for `linear`, `interaction` or `quadratic` models over any candidate set (drop infeasible points to impose constraints); reports `log det(X'X)` and D-efficiency.

```python
from experiment_design_kit import candidate_grid, d_optimal_design, maximin_latin_hypercube

d = d_optimal_design(10, candidate_grid(3, levels=3), model="quadratic", seed=0)
print(d.matrix, d.d_efficiency)
x = maximin_latin_hypercube(20, 3, seed=1)
```

```bash
experiment-design-kit optimal-design --factors 3 --runs 10 --model quadratic --seed 0
experiment-design-kit optimal-design --factors 2 --runs 12 --kind maximin-lhs --seed 1
```
