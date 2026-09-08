"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CUPEDResult:
    """Adjusted mean, variance reduction, and estimated treatment effect."""

    adjusted_mean: float
    adjusted_variance: float
    variance_reduction: float
    effect: float
    control_mean: float
    treatment_mean: float


def cuped_adjust(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    *,
    control_label: int = 0,
) -> CUPEDResult:
    """Apply a CUPED adjustment to reduce outcome variance using covariates.

    CUPED fits a linear model ``outcome ~ covariate`` on the control group and
    uses the fitted coefficient to compute an adjusted outcome. The ratio of
    adjusted to unadjusted variance is returned as the variance-reduction
    factor. When the covariate is uncorrelated with the outcome the reduction
    factor approaches 1.0 (no benefit).

    Parameters
    ----------
    outcomes:
        Observed outcome values (e.g. revenue, engagement score).
    treatment:
        Binary indicator: 0 = control, 1 = treatment.
    covariates:
        Pre-experiment covariate with the same length as ``outcomes``.
    control_label:
        Value in ``treatment`` that identifies the control group.

    Returns
    -------
    CUPEDResult
        Adjusted statistics and the estimated treatment effect.
    """
    outcomes = np.asarray(outcomes, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    covariates = np.asarray(covariates, dtype=float)
    if outcomes.shape != covariates.shape or outcomes.shape != treatment.shape:
        raise ValueError("outcomes, treatment, and covariates must have the same length")
    if outcomes.size == 0:
        raise ValueError("cannot adjust an empty dataset")

    mask = treatment == control_label
    if not np.any(mask) or not np.any(~mask):
        raise ValueError("treatment must contain both control and treatment observations")

    c_outcomes = outcomes[mask]
    t_outcomes = outcomes[~mask]
    c_cov = covariates[mask]
    t_cov = covariates[~mask]

    c_mean = float(np.mean(c_outcomes))
    t_mean = float(np.mean(t_outcomes))
    effect = t_mean - c_mean

    cov_mean = float(np.mean(covariates))
    cov_centered = covariates - cov_mean
    outcome_centered = outcomes - np.mean(outcomes)
    cov_var = float(np.mean(cov_centered**2))
    if cov_var == 0.0:
        adj_outcomes = outcomes.copy()
    else:
        theta = float(np.mean(cov_centered * outcome_centered) / cov_var)
        adj_outcomes = outcomes - theta * cov_centered

    adj_c = adj_outcomes[mask]
    adj_var = float(np.var(adj_outcomes, ddof=0))
    raw_var = float(np.var(outcomes, ddof=0))
    reduction = adj_var / raw_var if raw_var > 0 else 1.0

    return CUPEDResult(
        adjusted_mean=float(np.mean(adj_outcomes)),
        adjusted_variance=adj_var,
        variance_reduction=reduction,
        effect=effect,
        control_mean=c_mean,
        treatment_mean=t_mean,
    )


__all__ = ["CUPEDResult", "cuped_adjust"]
