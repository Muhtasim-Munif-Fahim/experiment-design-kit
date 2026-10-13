"""ICC-aware power and sample size for cluster-randomized trials.

When whole clusters (schools, stores, cities, ...) are randomized, outcomes
inside a cluster are correlated and every individual carries less
information than in an individually randomized test. The inflation is the
*design effect* (Donner & Klar, 2000; Eldridge, Ashby & Kerry, 2006)::

    DE = 1 + ((cv^2 + 1) * m - 1) * icc

where ``m`` is the mean cluster size, ``icc`` the intracluster correlation
and ``cv`` the coefficient of variation of cluster sizes (``cv = 0`` gives
the classic ``1 + (m - 1) * icc``). Planning then multiplies the
individually randomized sample size by ``DE`` and converts it to clusters.

``method="normal"`` returns the closed-form Donner & Klar cluster count
``k = ceil(n_individual * DE / m)``. ``method="t"`` (default) searches for
the smallest ``k`` whose power under a noncentral t with ``2k - 2``
cluster-level degrees of freedom meets the target, which matters when
only a handful of clusters per arm are affordable.

``estimate_icc`` is the one-way ANOVA estimator for (possibly unbalanced)
clusters, with Searle's F-based confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp

__all__ = [
    "ClusterSampleSizeResult",
    "ICCEstimate",
    "cluster_mde",
    "cluster_power",
    "cluster_sample_size",
    "design_effect",
    "effective_sample_size",
    "estimate_icc",
]


@dataclass
class ClusterSampleSizeResult:
    """Clusters (and individuals) per arm for a cluster-randomized design."""

    clusters_per_arm: int
    cluster_size: float
    individuals_per_arm: int
    total_clusters: int
    total_individuals: int
    design_effect: float
    n_individual_per_arm: float
    achieved_power: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ICCEstimate:
    """One-way ANOVA intracluster correlation estimate."""

    icc: float
    ci_lower: float
    ci_upper: float
    msb: float
    msw: float
    n_clusters: int
    n_obs: int
    m0: float
    confidence: float


def _validate_icc(icc: float) -> float:
    icc = float(icc)
    if not 0.0 <= icc <= 1.0:
        raise ValueError("icc must be in [0, 1]")
    return icc


def _validate_alpha_power(alpha: float, power: float | None = None) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")


def design_effect(cluster_size: float, icc: float, cv: float = 0.0) -> float:
    """Variance inflation factor ``1 + ((cv^2 + 1) m - 1) icc``."""
    m = float(cluster_size)
    if m < 1:
        raise ValueError("cluster_size must be at least 1")
    icc = _validate_icc(icc)
    if cv < 0:
        raise ValueError("cv must be non-negative")
    return 1.0 + ((cv * cv + 1.0) * m - 1.0) * icc


def effective_sample_size(n_total: float, cluster_size: float, icc: float, cv: float = 0.0) -> float:
    """Number of independent observations a clustered sample is worth."""
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    return float(n_total) / design_effect(cluster_size, icc, cv)


def _variance_unit(kind: str, effect_size, p1, p2) -> tuple[float, float]:
    """Return (|effect|, per-individual variance sum across both arms)."""
    if kind == "means":
        if effect_size is None or effect_size <= 0:
            raise ValueError("effect_size (Cohen's d) must be positive")
        return float(effect_size), 2.0
    if kind == "proportions":
        if p1 is None or p2 is None:
            raise ValueError("p1 and p2 are required for kind='proportions'")
        for label, p in (("p1", p1), ("p2", p2)):
            if not 0.0 < p < 1.0:
                raise ValueError(f"{label} must be in (0, 1)")
        delta = abs(p1 - p2)
        if delta == 0:
            raise ValueError("p1 and p2 must differ")
        return delta, p1 * (1.0 - p1) + p2 * (1.0 - p2)
    raise ValueError("kind must be 'means' or 'proportions'")


def cluster_power(
    n_clusters_per_arm: int,
    cluster_size: float,
    icc: float,
    *,
    effect_size: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    kind: str = "means",
    alpha: float = 0.05,
    cv: float = 0.0,
    method: str = "t",
) -> float:
    """Two-sided power of a two-arm cluster-randomized comparison.

    ``kind="means"`` takes a standardized ``effect_size`` (Cohen's d on the
    total SD); ``kind="proportions"`` takes ``p1`` and ``p2``. ``method``
    is ``"t"`` (noncentral t, ``df = 2k - 2``) or ``"normal"``.
    """
    k = int(n_clusters_per_arm)
    if k < 2:
        raise ValueError("n_clusters_per_arm must be at least 2")
    _validate_alpha_power(alpha)
    de = design_effect(cluster_size, icc, cv)
    delta, var_sum = _variance_unit(kind, effect_size, p1, p2)
    se = math.sqrt(var_sum * de / (k * float(cluster_size)))
    ncp = delta / se
    if method == "normal":
        z = sp.norm.ppf(1.0 - alpha / 2.0)
        return float(sp.norm.sf(z - ncp) + sp.norm.cdf(-z - ncp))
    if method != "t":
        raise ValueError("method must be 't' or 'normal'")
    df = 2 * k - 2
    t_crit = sp.t.ppf(1.0 - alpha / 2.0, df)
    return float(sp.nct.sf(t_crit, df, ncp) + sp.nct.cdf(-t_crit, df, ncp))


def cluster_sample_size(
    cluster_size: float,
    icc: float,
    *,
    effect_size: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    kind: str = "means",
    alpha: float = 0.05,
    power: float = 0.8,
    cv: float = 0.0,
    method: str = "t",
    max_clusters: int = 100_000,
) -> ClusterSampleSizeResult:
    """Clusters per arm needed to reach ``power`` (Donner & Klar).

    ``n_individual_per_arm`` is the unclustered normal-approximation size;
    it is inflated by the design effect and divided by ``cluster_size``.
    With ``method="t"`` the count is then raised until the exact
    noncentral-t power (``2k - 2`` df) reaches the target.
    """
    _validate_alpha_power(alpha, power)
    if method not in ("t", "normal"):
        raise ValueError("method must be 't' or 'normal'")
    m = float(cluster_size)
    de = design_effect(m, icc, cv)
    delta, var_sum = _variance_unit(kind, effect_size, p1, p2)
    z = sp.norm.ppf(1.0 - alpha / 2.0) + sp.norm.ppf(power)
    n_ind = z * z * var_sum / (delta * delta)
    k = max(2, math.ceil(n_ind * de / m - 1e-9))
    common = dict(effect_size=effect_size, p1=p1, p2=p2, kind=kind, alpha=alpha, cv=cv)
    if method == "t":
        while cluster_power(k, m, icc, method="t", **common) < power:
            k += 1
            if k > max_clusters:
                raise ValueError("no feasible design within max_clusters")
    achieved = cluster_power(k, m, icc, method=method, **common)
    per_arm = int(math.ceil(k * m - 1e-9))
    return ClusterSampleSizeResult(
        clusters_per_arm=k,
        cluster_size=m,
        individuals_per_arm=per_arm,
        total_clusters=2 * k,
        total_individuals=2 * per_arm,
        design_effect=de,
        n_individual_per_arm=float(n_ind),
        achieved_power=achieved,
        metadata={"kind": kind, "icc": float(icc), "cv": float(cv), "alpha": alpha,
                  "target_power": power, "method": method},
    )


def cluster_mde(
    n_clusters_per_arm: int,
    cluster_size: float,
    icc: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    cv: float = 0.0,
    method: str = "t",
) -> float:
    """Minimum detectable standardized effect (Cohen's d) for a fixed design.

    ``method="t"`` uses ``t_{2k-2}`` quantiles (Bloom 1995 style),
    ``"normal"`` the z quantiles.
    """
    k = int(n_clusters_per_arm)
    if k < 2:
        raise ValueError("n_clusters_per_arm must be at least 2")
    _validate_alpha_power(alpha, power)
    de = design_effect(cluster_size, icc, cv)
    if method == "normal":
        mult = sp.norm.ppf(1.0 - alpha / 2.0) + sp.norm.ppf(power)
    elif method == "t":
        df = 2 * k - 2
        mult = sp.t.ppf(1.0 - alpha / 2.0, df) + sp.t.ppf(power, df)
    else:
        raise ValueError("method must be 't' or 'normal'")
    return float(mult * math.sqrt(2.0 * de / (k * float(cluster_size))))


def estimate_icc(values, clusters, confidence: float = 0.95, truncate: bool = True) -> ICCEstimate:
    """One-way random-effects ANOVA estimate of the ICC.

    ``icc = (MSB - MSW) / (MSB + (m0 - 1) MSW)`` with the unbalanced-size
    adjustment ``m0 = (N - sum n_j^2 / N) / (k - 1)``. The CI inverts the
    ``F = MSB / MSW`` statistic (Searle 1971; exact for balanced data).
    Negative estimates are truncated to 0 when ``truncate``.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    g = np.asarray(clusters).ravel()
    if y.shape != g.shape:
        raise ValueError("values and clusters must have the same length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    labels, inv = np.unique(g, return_inverse=True)
    k = labels.size
    N = y.size
    if k < 2:
        raise ValueError("need at least two clusters")
    if N - k < 1:
        raise ValueError("need at least one cluster with two or more observations")
    counts = np.bincount(inv).astype(np.float64)
    sums = np.bincount(inv, weights=y)
    means = sums / counts
    grand = y.mean()
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - float(np.sum(counts**2)) / N) / (k - 1)
    if msw == 0:
        icc = 1.0 if msb > 0 else 0.0
        lo = hi = icc
    else:
        icc = (msb - msw) / (msb + (m0 - 1.0) * msw)
        F = msb / msw
        a = 1.0 - confidence
        f_hi = sp.f.ppf(1.0 - a / 2.0, k - 1, N - k)
        f_lo = sp.f.ppf(a / 2.0, k - 1, N - k)
        fl, fu = F / f_hi, F / f_lo
        lo = (fl - 1.0) / (fl + m0 - 1.0)
        hi = (fu - 1.0) / (fu + m0 - 1.0)
    if truncate:
        icc, lo, hi = (max(0.0, v) for v in (icc, lo, hi))
    return ICCEstimate(
        icc=float(min(icc, 1.0)),
        ci_lower=float(min(lo, 1.0)),
        ci_upper=float(min(hi, 1.0)),
        msb=msb,
        msw=msw,
        n_clusters=int(k),
        n_obs=int(N),
        m0=float(m0),
        confidence=confidence,
    )
