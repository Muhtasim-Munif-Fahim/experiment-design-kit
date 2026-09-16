"""Always-valid sequential monitoring for A/B testing.

Peeking at a fixed-horizon p-value more than once inflates the false-positive
rate. This module provides two practical corrections that stay
dependency-light (numpy + scipy only):

* **Always-valid p-values** from a mixture Sequential Probability Ratio Test
  (mSPRT; Robbins 1970, Johari et al. 2015/2017). ``p_av < alpha`` at *any*
  look is a valid level-``alpha`` rejection, even under continuous monitoring.
* **Lan-DeMets alpha-spending** bounds (O'Brien-Fleming or Pocock) for a
  planned sequence of looks. These spend a Type I error budget as a function
  of the information fraction ``t = n / n_planned``.

One-sample helpers test a proportion against ``null_p`` or a mean against
``null_mean``. Two-sample helpers compare control vs treatment — the usual
A/B setting. A Monte Carlo ``simulate_peeking_fpr`` reports the empirical
false-positive rate of naive repeated testing versus these sequential bounds
under the null.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
from scipy import stats as sp

from .stats import two_proportion_z

_ALWAYS_VALID = "always-valid"
_POCOCK = "pocock"
_OBF = "obrien-fleming"
_BOUNDARIES = {_ALWAYS_VALID, _POCOCK, _OBF}
_OBF_ALIASES = {"obf", "o'brien-fleming", "obrien_fleming"}


@dataclass
class SequentialResult:
    """Result of a single sequential look."""

    look: int
    z_statistic: float
    p_value: float
    always_valid_pvalue: float
    alpha: float
    significant: bool
    cumulative_n: int
    alpha_spent: float = 0.0
    spending_increment: float = 0.0
    information_fraction: float = 1.0
    boundary: str = _ALWAYS_VALID


@dataclass
class SequentialReport:
    """Full report from a sequential monitoring run."""

    results: List[SequentialResult] = field(default_factory=list)
    stopped: bool = False
    stopped_at: int = -1
    final_pvalue: float = 1.0
    boundary: str = _ALWAYS_VALID

    @property
    def is_significant(self) -> bool:
        return any(r.significant for r in self.results)

    @property
    def naive_significant(self) -> bool:
        """True if any look's *naive* two-sided p-value is below ``alpha``."""
        return any(r.p_value < r.alpha for r in self.results)


@dataclass(frozen=True)
class PeekingFPRResult:
    """Empirical false-positive rates under optional stopping (null true)."""

    n_looks: int
    n_per_look: int
    n_trials: int
    alpha: float
    metric: str
    naive_fpr: float
    always_valid_fpr: float
    pocock_fpr: float
    obrien_fleming_fpr: float
    seed: int


def normalize_boundary(boundary: str) -> str:
    """Map a boundary name to one of ``always-valid``, ``pocock``, ``obrien-fleming``."""
    key = boundary.strip().lower().replace("_", "-")
    if key in _OBF_ALIASES:
        return _OBF
    if key not in _BOUNDARIES:
        raise ValueError(
            "boundary must be one of 'always-valid', 'pocock', 'obrien-fleming'"
        )
    return key


def mixture_likelihood_ratio(z_statistic: float, n: float, m: float = 0.01) -> float:
    """Normal-mixture likelihood ratio ``Λ`` used by the mSPRT.

    For a normal mixing distribution on the effect with relative variance
    ``m`` and current sample size (or Fisher information) ``n``,

    .. math::
        \\Lambda = (1 + m n)^{-1/2}
        \\exp\\!\\left(\\frac{z^2 m n}{2(1 + m n)}\\right)

    Reject the null when ``Λ ≥ 1/alpha``. ``n <= 0`` returns ``1`` (no evidence).
    """
    if n <= 0 or m <= 0:
        return 1.0
    nm = m * n
    denom = 1.0 + nm
    # Compute in log space for stability when |z| is large.
    log_lambda = -0.5 * math.log(denom) + (z_statistic ** 2) * nm / (2.0 * denom)
    # Guard against overflow: Λ is only used as 1/Λ clipped to [0, 1].
    if log_lambda > 700.0:
        return math.inf
    return math.exp(log_lambda)


def always_valid_pvalue(z_statistic: float, n: int | float, m: float = 0.01) -> float:
    """Always-valid p-value for a z-statistic via the mSPRT.

    Uses the mixture sequential probability ratio test (Johari et al.):

    .. math::
        p_{av} = \\min\\!\\left(1,\\;
        \\sqrt{1 + m n}\\;
        \\exp\\!\\left(-\\frac{z^2 m n}{2(1 + m n)}\\right)\\right)

    ``m`` is the mixture parameter (relative variance of the mixing
    distribution) and ``n`` is the current sample size, or Fisher information
    ``1/\\mathrm{sem}^2`` when only a standard error is available.

    This is the p-value at a *single* look. The sequential tests below take
    the running minimum so the process stays a valid p-value process:
    ``p_n = min(p_{n-1}, 1/Λ_n)``.
    """
    if n <= 0 or m <= 0:
        return 1.0
    lam = mixture_likelihood_ratio(float(z_statistic), float(n), m)
    if lam <= 0.0 or not math.isfinite(lam):
        return 0.0 if lam > 1.0 else 1.0
    return min(1.0, max(0.0, 1.0 / lam))


def alpha_spent(
    information_fraction: float,
    alpha: float = 0.05,
    method: str = _OBF,
) -> float:
    """Lan-DeMets spending function ``α*(t)`` at information fraction ``t``.

    ``method`` is ``"obrien-fleming"`` (conservative early looks) or
    ``"pocock"`` (more even spending). At ``t = 1`` both return ``alpha``.
    """
    _require_alpha(alpha)
    if information_fraction <= 0.0:
        return 0.0
    t = min(1.0, float(information_fraction))
    method_n = normalize_boundary(method) if method != _ALWAYS_VALID else _OBF
    if method_n == _OBF:
        z_alpha = sp.norm.ppf(1.0 - alpha / 2.0)
        return float(2.0 * (1.0 - sp.norm.cdf(z_alpha / math.sqrt(t))))
    if method_n == _POCOCK:
        return float(alpha * math.log(1.0 + (math.e - 1.0) * t))
    raise ValueError("alpha-spending method must be 'pocock' or 'obrien-fleming'")


def spending_increment(
    t_prev: float,
    t: float,
    alpha: float = 0.05,
    method: str = _OBF,
) -> float:
    """Type I error spent between information fractions ``t_prev`` and ``t``."""
    spent_now = alpha_spent(t, alpha=alpha, method=method)
    spent_prev = alpha_spent(t_prev, alpha=alpha, method=method)
    return max(0.0, spent_now - spent_prev)


def sequential_proportion_test(
    successes: List[int],
    totals: List[int],
    alpha: float = 0.05,
    m: float = 0.01,
    stop_on_significant: bool = False,
    *,
    null_p: float = 0.5,
    boundary: str = _ALWAYS_VALID,
    planned_n: int | None = None,
) -> SequentialReport:
    """Sequential z-test for a single proportion with always-valid p-values.

    Parameters
    ----------
    successes
        Number of successes *at each look* (incremental, not cumulative).
        These are accumulated internally to form cumulative counts.
    totals
        Sample size *at each look* (incremental, not cumulative).
    alpha
        Nominal false-positive rate.
    m
        Mixture parameter (relative variance of the mixing distribution).
        Smaller values make the always-valid test more conservative.
    stop_on_significant
        If ``True``, stop processing after the first significant look.
        If ``False`` (default), all looks are processed.
    null_p
        Null hypothesis proportion (default 0.5).
    boundary
        ``"always-valid"`` (mSPRT), ``"pocock"``, or ``"obrien-fleming"``.
    planned_n
        Planned maximum sample size used as ``t = n / planned_n`` for
        spending bounds. Defaults to the total of ``totals``.
    """
    if len(successes) != len(totals):
        raise ValueError("successes and totals must have the same length")
    _require_alpha(alpha)
    _require_m(m)
    if not 0.0 <= null_p <= 1.0:
        raise ValueError("null_p must be in [0, 1]")
    boundary_n = normalize_boundary(boundary)

    zs: list[float] = []
    ps: list[float] = []
    ns: list[int] = []
    cumulative_successes = 0
    cumulative_total = 0
    for s, n in zip(successes, totals):
        if n <= 0:
            raise ValueError("totals must be positive")
        if s < 0:
            raise ValueError("successes must be non-negative")
        cumulative_successes += s
        cumulative_total += n
        if cumulative_successes > cumulative_total:
            raise ValueError("cumulative successes cannot exceed cumulative totals")
        p_hat = cumulative_successes / cumulative_total
        se = math.sqrt(null_p * (1.0 - null_p) / cumulative_total)
        z = abs(p_hat - null_p) / se if se > 0 else 0.0
        p_value = 2.0 * float(sp.norm.sf(z))
        zs.append(z)
        ps.append(p_value)
        ns.append(cumulative_total)

    return _assemble_report(
        zs,
        ps,
        ns,
        alpha=alpha,
        m=m,
        boundary=boundary_n,
        stop_on_significant=stop_on_significant,
        planned_n=planned_n,
    )


def sequential_two_proportion_test(
    control_successes: Sequence[int],
    control_totals: Sequence[int],
    treatment_successes: Sequence[int],
    treatment_totals: Sequence[int],
    *,
    alpha: float = 0.05,
    m: float = 0.01,
    stop_on_significant: bool = False,
    boundary: str = _ALWAYS_VALID,
    planned_n: int | None = None,
) -> SequentialReport:
    """Sequential two-proportion z-test (A/B) with always-valid / spending bounds.

    Counts are *incremental* per look, matching
    :func:`sequential_proportion_test`. The z-statistic at each look is the
    pooled two-proportion test on the cumulative counts. ``cumulative_n`` is
    the per-arm sample size ``min(n_control, n_treatment)`` used as ``n`` in
    the mSPRT.
    """
    c_s = list(control_successes)
    c_n = list(control_totals)
    t_s = list(treatment_successes)
    t_n = list(treatment_totals)
    n_looks = len(c_s)
    if not (len(c_n) == len(t_s) == len(t_n) == n_looks):
        raise ValueError("control/treatment successes and totals must have the same length")
    _require_alpha(alpha)
    _require_m(m)
    boundary_n = normalize_boundary(boundary)

    zs: list[float] = []
    ps: list[float] = []
    ns: list[int] = []
    cum_cs = cum_cn = cum_ts = cum_tn = 0
    for cs, cn, ts, tn in zip(c_s, c_n, t_s, t_n):
        if cn <= 0 or tn <= 0:
            raise ValueError("totals must be positive")
        if cs < 0 or ts < 0:
            raise ValueError("successes must be non-negative")
        cum_cs += cs
        cum_cn += cn
        cum_ts += ts
        cum_tn += tn
        if cum_cs > cum_cn or cum_ts > cum_tn:
            raise ValueError("cumulative successes cannot exceed cumulative totals")
        p1 = cum_cs / cum_cn
        p2 = cum_ts / cum_tn
        result = two_proportion_z(p1, p2, cum_cn, cum_tn, alpha=alpha)
        zs.append(abs(result.z))
        ps.append(float(result.p_value))
        ns.append(min(cum_cn, cum_tn))

    return _assemble_report(
        zs,
        ps,
        ns,
        alpha=alpha,
        m=m,
        boundary=boundary_n,
        stop_on_significant=stop_on_significant,
        planned_n=planned_n,
    )


def sequential_mean_test(
    means: List[float],
    sems: List[float],
    *,
    null_mean: float = 0.0,
    alpha: float = 0.05,
    m: float = 0.01,
    stop_on_significant: bool = False,
    ns: Sequence[int] | None = None,
    boundary: str = _ALWAYS_VALID,
    planned_n: int | None = None,
) -> SequentialReport:
    """Sequential z-test for means with always-valid p-values.

    Parameters
    ----------
    means:
        Sample mean at each look (cumulative, not incremental).
    sems:
        Standard error of the mean at each look (cumulative).
    null_mean:
        Null hypothesis mean.
    alpha:
        Nominal false-positive rate.
    m:
        Mixture parameter for the mSPRT correction. When ``ns`` is omitted,
        the mSPRT uses Fisher information ``1/sem^2`` as ``n``, so ``m`` is
        the mixing variance on the mean scale.
    stop_on_significant:
        Stop after the first significant look.
    ns:
        Optional cumulative sample size at each look. When omitted,
        ``cumulative_n`` is the look index (1, 2, ...) for backwards
        compatibility and the mSPRT uses ``1/sem^2``.
    boundary:
        ``"always-valid"``, ``"pocock"``, or ``"obrien-fleming"``.
    planned_n:
        Planned maximum sample size (or information) for spending bounds.
    """
    if len(means) != len(sems):
        raise ValueError("means and sems must have the same length")
    if ns is not None and len(ns) != len(means):
        raise ValueError("ns must have the same length as means")
    _require_alpha(alpha)
    _require_m(m)
    boundary_n = normalize_boundary(boundary)

    zs: list[float] = []
    ps: list[float] = []
    n_for_msprt: list[float] = []
    n_reported: list[int] = []
    for i, (mean, sem) in enumerate(zip(means, sems)):
        if sem <= 0:
            raise ValueError("sems must be strictly positive")
        z = abs(mean - null_mean) / sem
        p_value = 2.0 * float(sp.norm.sf(z))
        zs.append(z)
        ps.append(p_value)
        if ns is None:
            n_reported.append(i + 1)
            n_for_msprt.append(1.0 / (sem ** 2))
        else:
            n_i = int(ns[i])
            if n_i <= 0:
                raise ValueError("ns must be positive")
            n_reported.append(n_i)
            n_for_msprt.append(float(n_i))

    return _assemble_report(
        zs,
        ps,
        n_reported,
        alpha=alpha,
        m=m,
        boundary=boundary_n,
        stop_on_significant=stop_on_significant,
        planned_n=planned_n,
        n_for_msprt=n_for_msprt,
    )


def sequential_two_sample_mean_test(
    control_means: Sequence[float],
    control_sds: Sequence[float],
    control_ns: Sequence[int],
    treatment_means: Sequence[float],
    treatment_sds: Sequence[float],
    treatment_ns: Sequence[int],
    *,
    alpha: float = 0.05,
    m: float = 0.01,
    stop_on_significant: bool = False,
    boundary: str = _ALWAYS_VALID,
    planned_n: int | None = None,
) -> SequentialReport:
    """Sequential two-sample z-test for means (Welch SE) at snapshot looks.

    Each look is a *cumulative snapshot*: running mean, SD, and sample size
    per arm. The z-statistic uses the Welch standard error
    ``sqrt(s_c^2/n_c + s_t^2/n_t)``. ``cumulative_n`` is
    ``min(n_control, n_treatment)``.
    """
    c_m = list(control_means)
    c_s = list(control_sds)
    c_n = list(control_ns)
    t_m = list(treatment_means)
    t_s = list(treatment_sds)
    t_n = list(treatment_ns)
    n_looks = len(c_m)
    if not (len(c_s) == len(c_n) == len(t_m) == len(t_s) == len(t_n) == n_looks):
        raise ValueError("two-sample mean snapshots must all have the same length")
    _require_alpha(alpha)
    _require_m(m)
    boundary_n = normalize_boundary(boundary)

    zs: list[float] = []
    ps: list[float] = []
    ns: list[int] = []
    for mean_c, sd_c, n_c, mean_t, sd_t, n_t in zip(c_m, c_s, c_n, t_m, t_s, t_n):
        if n_c < 2 or n_t < 2:
            raise ValueError("group sample sizes must be at least 2")
        if sd_c < 0 or sd_t < 0:
            raise ValueError("standard deviations must be non-negative")
        se = math.sqrt(sd_c ** 2 / n_c + sd_t ** 2 / n_t)
        z = abs(mean_t - mean_c) / se if se > 0 else 0.0
        p_value = 2.0 * float(sp.norm.sf(z))
        zs.append(z)
        ps.append(p_value)
        ns.append(min(int(n_c), int(n_t)))

    return _assemble_report(
        zs,
        ps,
        ns,
        alpha=alpha,
        m=m,
        boundary=boundary_n,
        stop_on_significant=stop_on_significant,
        planned_n=planned_n,
    )


def simulate_peeking_fpr(
    n_looks: int = 8,
    n_per_look: int = 200,
    n_trials: int = 400,
    alpha: float = 0.05,
    *,
    metric: str = "proportion",
    m: float = 0.01,
    seed: int = 0,
    p_control: float = 0.10,
    p_treatment: float = 0.10,
    mean_control: float = 0.0,
    mean_treatment: float = 0.0,
    sd: float = 1.0,
) -> PeekingFPRResult:
    """Monte Carlo false-positive rates when peeking under the null.

    Each trial draws ``n_looks`` batches of ``n_per_look`` observations per
    arm under ``H0`` (equal rates or equal means). At every look the naive
    two-sided p-value, the mSPRT always-valid p-value, and Lan-DeMets
    spending increments (Pocock and O'Brien-Fleming) are computed. A method
    "rejects" if it is significant at *any* look — optional stopping.

    Under ``H0`` the naive FPR is typically well above ``alpha`` once there
    are several looks; always-valid and spending bounds stay near or below
    ``alpha``.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    if n_per_look < 2:
        raise ValueError("n_per_look must be at least 2")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    _require_alpha(alpha)
    _require_m(m)
    metric_n = metric.strip().lower()
    if metric_n not in {"proportion", "mean", "continuous"}:
        raise ValueError("metric must be 'proportion' or 'mean'")

    rng = np.random.default_rng(seed)
    n_cum = n_per_look * np.arange(1, n_looks + 1, dtype=float)
    t_frac = n_cum / n_cum[-1]
    pocock_inc = np.array(
        [
            spending_increment(
                0.0 if i == 0 else float(t_frac[i - 1]),
                float(t_frac[i]),
                alpha=alpha,
                method=_POCOCK,
            )
            for i in range(n_looks)
        ]
    )
    obf_inc = np.array(
        [
            spending_increment(
                0.0 if i == 0 else float(t_frac[i - 1]),
                float(t_frac[i]),
                alpha=alpha,
                method=_OBF,
            )
            for i in range(n_looks)
        ]
    )

    if metric_n == "proportion":
        if not 0.0 <= p_control <= 1.0 or not 0.0 <= p_treatment <= 1.0:
            raise ValueError("p_control and p_treatment must be in [0, 1]")
        c_batch = rng.binomial(n_per_look, p_control, size=(n_trials, n_looks))
        t_batch = rng.binomial(n_per_look, p_treatment, size=(n_trials, n_looks))
        c_cum = np.cumsum(c_batch, axis=1)
        t_cum = np.cumsum(t_batch, axis=1)
        p_c = c_cum / n_cum
        p_t = t_cum / n_cum
        pooled = (c_cum + t_cum) / (2.0 * n_cum)
        se = np.sqrt(np.maximum(pooled * (1.0 - pooled) * (2.0 / n_cum), 0.0))
        z = np.divide(p_t - p_c, se, out=np.zeros_like(p_c), where=se > 0)
    else:
        if sd <= 0:
            raise ValueError("sd must be positive")
        # Equal-sized batches: running mean of i.i.d. batch means.
        c_batch = rng.normal(
            mean_control, sd / math.sqrt(n_per_look), size=(n_trials, n_looks)
        )
        t_batch = rng.normal(
            mean_treatment, sd / math.sqrt(n_per_look), size=(n_trials, n_looks)
        )
        c_mean = np.cumsum(c_batch, axis=1) / np.arange(1, n_looks + 1)
        t_mean = np.cumsum(t_batch, axis=1) / np.arange(1, n_looks + 1)
        se = sd * np.sqrt(2.0 / n_cum)
        z = (t_mean - c_mean) / se

    p_naive = 2.0 * sp.norm.sf(np.abs(z))
    av = _always_valid_pvalue_array(z, n_cum, m)
    av_run = np.minimum.accumulate(av, axis=1)

    naive_fpr = float(np.mean(np.any(p_naive < alpha, axis=1)))
    av_fpr = float(np.mean(np.any(av_run < alpha, axis=1)))
    pocock_fpr = float(np.mean(np.any(p_naive < pocock_inc, axis=1)))
    obf_fpr = float(np.mean(np.any(p_naive < obf_inc, axis=1)))

    return PeekingFPRResult(
        n_looks=n_looks,
        n_per_look=n_per_look,
        n_trials=n_trials,
        alpha=alpha,
        metric="mean" if metric_n in {"mean", "continuous"} else "proportion",
        naive_fpr=naive_fpr,
        always_valid_fpr=av_fpr,
        pocock_fpr=pocock_fpr,
        obrien_fleming_fpr=obf_fpr,
        seed=seed,
    )


def _always_valid_pvalue_array(z: np.ndarray, n: np.ndarray, m: float) -> np.ndarray:
    """Vectorized :func:`always_valid_pvalue`."""
    nm = m * n
    denom = 1.0 + nm
    log_p = 0.5 * np.log(denom) - (z ** 2) * nm / (2.0 * denom)
    return np.clip(np.exp(log_p), 0.0, 1.0)


def _assemble_report(
    zs: Sequence[float],
    ps: Sequence[float],
    ns: Sequence[int],
    *,
    alpha: float,
    m: float,
    boundary: str,
    stop_on_significant: bool,
    planned_n: int | None,
    n_for_msprt: Sequence[float] | None = None,
) -> SequentialReport:
    report = SequentialReport(boundary=boundary)
    av_running = 1.0
    t_prev = 0.0
    planned = float(planned_n) if planned_n is not None else float(ns[-1] if ns else 1)
    if planned <= 0:
        raise ValueError("planned_n must be positive")
    msprt_n = n_for_msprt if n_for_msprt is not None else ns

    for i, (z, p, n, n_m) in enumerate(zip(zs, ps, ns, msprt_n)):
        t = min(1.0, float(n) / planned)
        av = always_valid_pvalue(z, n_m, m)
        av_running = min(av_running, av)
        if boundary == _ALWAYS_VALID:
            spent = alpha
            increment = alpha
            significant = av_running < alpha
        else:
            spent = alpha_spent(t, alpha=alpha, method=boundary)
            increment = spending_increment(t_prev, t, alpha=alpha, method=boundary)
            significant = p < increment and increment > 0.0

        report.results.append(
            SequentialResult(
                look=i + 1,
                z_statistic=float(z),
                p_value=float(p),
                always_valid_pvalue=float(av_running),
                alpha=alpha,
                significant=bool(significant),
                cumulative_n=int(n),
                alpha_spent=float(spent),
                spending_increment=float(increment),
                information_fraction=float(t),
                boundary=boundary,
            )
        )
        t_prev = t
        if significant and stop_on_significant and not report.stopped:
            report.stopped = True
            report.stopped_at = i + 1
            report.final_pvalue = float(av_running)
            break

    if report.results:
        report.final_pvalue = report.results[-1].always_valid_pvalue
    return report


def _require_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _require_m(m: float) -> None:
    if m <= 0:
        raise ValueError("m must be positive")


__all__ = [
    "PeekingFPRResult",
    "SequentialResult",
    "SequentialReport",
    "alpha_spent",
    "always_valid_pvalue",
    "mixture_likelihood_ratio",
    "normalize_boundary",
    "sequential_mean_test",
    "sequential_proportion_test",
    "sequential_two_proportion_test",
    "sequential_two_sample_mean_test",
    "simulate_peeking_fpr",
    "spending_increment",
]
