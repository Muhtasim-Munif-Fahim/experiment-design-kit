"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

CUPED (Deng, Xu, Kohavi, Walker 2013) uses a pre-experiment covariate ``X``
to form an adjusted continuous metric

    Y_cv = Y - θ (X - E[X])

where ``θ = Cov(Y, X) / Var(X)`` is the OLS slope of the outcome on the
covariate. The adjusted metric is unbiased for the same mean as ``Y``
(when ``X`` is pre-period and independent of assignment) and has variance
``Var(Y) * (1 - ρ²)``, so the variance reduction is approximately the
squared pre/post correlation.

This module stays numpy-only: estimate ``θ``, form the adjusted metric,
and report the CUPED treatment effect plus the realised variance reduction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CUPEDResult:
    """CUPED-adjusted means, effect, and variance-reduction diagnostics."""

    theta: float
    correlation: float
    adjusted_mean: float
    adjusted_variance: float
    raw_variance: float
    variance_ratio: float
    variance_reduction: float
    effect: float
    raw_effect: float
    control_mean: float
    treatment_mean: float
    adjusted_control_mean: float
    adjusted_treatment_mean: float
    n_control: int
    n_treatment: int
    fit_on: str


def estimate_theta(outcomes: np.ndarray, covariates: np.ndarray) -> float:
    """OLS slope ``θ = Cov(Y, X) / Var(X)`` used as the CUPED coefficient.

    Parameters
    ----------
    outcomes:
        Post-period (or in-experiment) continuous metric.
    covariates:
        Pre-experiment covariate, same length as ``outcomes``.

    Returns
    -------
    float
        Estimated θ. Returns ``0.0`` when ``X`` has zero variance (no
        adjustment is possible).
    """
    y, x = _as_paired_vectors(outcomes, covariates, "outcomes", "covariates")
    if y.size < 2:
        raise ValueError("need at least 2 observations to estimate theta")
    x_c = x - float(np.mean(x))
    y_c = y - float(np.mean(y))
    var_x = float(np.dot(x_c, x_c))
    if var_x == 0.0:
        return 0.0
    return float(np.dot(x_c, y_c) / var_x)


def adjust_metric(
    outcomes: np.ndarray,
    covariates: np.ndarray,
    theta: float | None = None,
    covariate_mean: float | None = None,
) -> np.ndarray:
    """Return the CUPED-adjusted metric ``Y - θ (X - E[X])``.

    If ``theta`` is omitted it is estimated from ``outcomes`` and
    ``covariates``. ``covariate_mean`` defaults to the sample mean of ``X``
    so that the adjustment is mean-preserving: ``E[Y_cv] = E[Y]``.
    """
    y, x = _as_paired_vectors(outcomes, covariates, "outcomes", "covariates")
    if theta is None:
        theta = estimate_theta(y, x)
    mu_x = float(np.mean(x)) if covariate_mean is None else float(covariate_mean)
    return y - float(theta) * (x - mu_x)


def simulate_cuped_data(
    n_per_group: int,
    *,
    correlation: float = 0.7,
    treatment_effect: float = 0.0,
    outcome_mean: float = 0.0,
    outcome_sd: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate correlated pre-period ``X`` and post-period ``Y`` with a known effect.

    Returns ``(outcomes, treatment, covariates)`` ready for :func:`cuped_adjust`.
    ``treatment`` is a 0/1 indicator with ``n_per_group`` units in each arm.
    The covariate is standard normal and independent of assignment; the
    residual is scaled so that ``Corr(Y, X) ≈ correlation`` when the
    treatment effect is zero.

    Parameters
    ----------
    n_per_group:
        Units per arm (control and treatment are equal-sized).
    correlation:
        Target pre/post correlation in ``[-1, 1]``.
    treatment_effect:
        Additive shift applied to the treatment arm after drawing ``Y``.
    outcome_mean, outcome_sd:
        Location and scale of the untreated outcome.
    seed:
        RNG seed.
    """
    if n_per_group < 1:
        raise ValueError("n_per_group must be at least 1")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    if outcome_sd <= 0:
        raise ValueError("outcome_sd must be positive")

    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    covariates = rng.normal(size=n)
    noise = rng.normal(size=n)
    residual_scale = math.sqrt(max(0.0, 1.0 - correlation ** 2))
    outcomes = outcome_mean + outcome_sd * (correlation * covariates + residual_scale * noise)
    treatment = np.concatenate(
        [np.zeros(n_per_group, dtype=int), np.ones(n_per_group, dtype=int)]
    )
    outcomes = outcomes.copy()
    outcomes[treatment == 1] += treatment_effect
    return outcomes, treatment, covariates


def cuped_adjust(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    control_label: int = 0,
    fit_on: str = "pooled",
) -> CUPEDResult:
    """Apply CUPED and report the adjusted effect plus variance reduction.

    θ is estimated as the OLS slope of ``Y`` on ``X``. By default the slope
    is pooled across arms (standard CUPED when ``X`` is pre-experiment).
    Pass ``fit_on="control"`` to estimate θ from the control arm only,
    which is safer if the covariate could be affected by treatment.

    The CUPED effect is the difference in adjusted means. In a randomised
    experiment this is unbiased for the same ATE as the raw difference but
    typically has lower sampling variance when ``X`` is correlated with
    ``Y``. ``variance_reduction`` is ``1 - Var(Y_cv) / Var(Y)`` (the
    fraction of variance removed); ``variance_ratio`` is the complementary
    ``Var(Y_cv) / Var(Y)``.

    Parameters
    ----------
    outcomes:
        Observed post-period outcome values.
    treatment:
        Assignment indicator (``control_label`` for control, anything else
        for treatment).
    covariates:
        Pre-experiment covariate, same length as ``outcomes``.
    control_label:
        Value in ``treatment`` that identifies the control group.
    fit_on:
        ``"pooled"`` (default) or ``"control"``.
    """
    if fit_on not in {"pooled", "control"}:
        raise ValueError("fit_on must be 'pooled' or 'control'")

    y, x = _as_paired_vectors(outcomes, covariates, "outcomes", "covariates")
    t = np.asarray(treatment).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("outcomes, treatment, and covariates must have the same length")
    if not np.all(np.isfinite(np.asarray(t, dtype=float))):
        raise ValueError("treatment must contain only finite values")
    if y.size == 0:
        raise ValueError("cannot adjust an empty dataset")

    control_mask = t == control_label
    n_control = int(np.sum(control_mask))
    n_treatment = int(y.size - n_control)
    if n_control == 0 or n_treatment == 0:
        raise ValueError("treatment must contain both control and treatment observations")

    if fit_on == "control":
        theta = estimate_theta(y[control_mask], x[control_mask])
    else:
        theta = estimate_theta(y, x)

    adjusted = adjust_metric(y, x, theta=theta, covariate_mean=float(np.mean(x)))

    raw_control = y[control_mask]
    raw_treatment = y[~control_mask]
    adj_control = adjusted[control_mask]
    adj_treatment = adjusted[~control_mask]

    raw_var = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    adj_var = float(np.var(adjusted, ddof=1)) if adjusted.size > 1 else 0.0
    if raw_var > 0:
        variance_ratio = adj_var / raw_var
    else:
        variance_ratio = 1.0
    variance_reduction = 1.0 - variance_ratio

    correlation = _safe_corr(y, x)

    c_mean = float(np.mean(raw_control))
    t_mean = float(np.mean(raw_treatment))
    adj_c_mean = float(np.mean(adj_control))
    adj_t_mean = float(np.mean(adj_treatment))

    return CUPEDResult(
        theta=theta,
        correlation=correlation,
        adjusted_mean=float(np.mean(adjusted)),
        adjusted_variance=adj_var,
        raw_variance=raw_var,
        variance_ratio=variance_ratio,
        variance_reduction=variance_reduction,
        effect=adj_t_mean - adj_c_mean,
        raw_effect=t_mean - c_mean,
        control_mean=c_mean,
        treatment_mean=t_mean,
        adjusted_control_mean=adj_c_mean,
        adjusted_treatment_mean=adj_t_mean,
        n_control=n_control,
        n_treatment=n_treatment,
        fit_on=fit_on,
    )


def _as_paired_vectors(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str,
    b_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError(f"{a_name} and {b_name} must have the same length")
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        raise ValueError(f"{a_name} and {b_name} must contain only finite values")
    return aa, bb


def _safe_corr(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    if float(np.var(y, ddof=0)) == 0.0 or float(np.var(x, ddof=0)) == 0.0:
        return 0.0
    return float(np.corrcoef(y, x)[0, 1])


__all__ = [
    "CUPEDResult",
    "adjust_metric",
    "cuped_adjust",
    "estimate_theta",
    "simulate_cuped_data",
]
