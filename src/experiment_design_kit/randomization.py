"""Stratified randomization and covariate balance diagnostics.

Stratified randomization assigns units to arms *inside* covariate strata so
the arms have nearly the same mix of those covariates. Strata are the
Cartesian product of one or more categorical factors. A continuous
covariate is turned into a factor by cutting it into quantile bins first.

Within each stratum, arm counts follow the requested allocation ratio
(equal by default) as closely as integer counts allow. The units that
receive each arm are a random permutation, so the assignment is
reproducible given ``seed``.

Balance diagnostics compare the realized arms:

* **Standardized mean difference** (Austin 2009) for a numeric covariate,

  ``(mean_arm - mean_reference) / sqrt((var_arm + var_reference) / 2)``

  with the unbiased sample variance. Binary indicators use the Bernoulli
  form ``p(1-p)`` in place of the sample variance.
* **Pearson chi-square** on the level-by-arm table for a categorical
  covariate, together with the per-level indicator SMDs. ``max_abs_smd``
  is the largest absolute SMD across levels and non-reference arms.

These checks describe the assignment. They are not a substitute for the
outcome analysis.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp


@dataclass(frozen=True)
class StratifiedAssignment:
    """Within-stratum random assignment of units to arms.

    ``factors`` holds the values that were actually stratified on: original
    categories, or quantile-bin indices when a covariate was binned.
    ``bin_edges`` records the cut points ``(min, q1, ..., max)`` for each
    binned covariate. ``stratum_arm_counts[s][k]`` is the number of units
    in stratum ``s`` assigned to arm ``k``.
    """

    assignment: np.ndarray
    strata: np.ndarray
    stratum_labels: tuple[str, ...]
    arm_labels: tuple[str, ...]
    stratum_arm_counts: tuple[tuple[int, ...], ...]
    factors: dict[str, np.ndarray]
    bin_edges: dict[str, tuple[float, ...]]
    seed: int | None

    @property
    def n_strata(self) -> int:
        return len(self.stratum_labels)

    @property
    def arm_counts(self) -> tuple[int, ...]:
        if not self.stratum_arm_counts:
            return ()
        totals = [0] * len(self.stratum_arm_counts[0])
        for row in self.stratum_arm_counts:
            for index, count in enumerate(row):
                totals[index] += count
        return tuple(totals)


@dataclass(frozen=True)
class BalanceRow:
    """Balance of one covariate against a reference arm.

    ``smd_by_arm`` is aligned with ``BalanceReport.arm_ids``. The reference
    arm's entry is ``0``. For a categorical covariate the other entries are
    the signed level-indicator SMD with the largest absolute value.
    ``means`` is the per-arm mean of a continuous covariate. ``levels`` is
    ``(label, counts_per_arm)`` for a categorical covariate.
    """

    covariate: str
    kind: str
    smd_by_arm: tuple[float, ...]
    max_abs_smd: float
    chi2: float | None
    p_value: float | None
    dof: int | None
    means: tuple[float, ...] | None
    levels: tuple[tuple[str, tuple[int, ...]], ...] | None


@dataclass(frozen=True)
class BalanceReport:
    """SMD and chi-square balance of an assignment on a set of covariates."""

    rows: tuple[BalanceRow, ...]
    max_abs_smd: float
    reference_arm: int
    arm_ids: tuple[int, ...]
    arm_labels: tuple[str, ...]
    arm_counts: tuple[int, ...]

    def __str__(self) -> str:
        return format_balance_report(self)


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Cut a numeric covariate into quantile bins labeled ``0 .. k-1``.

    Ties in the empirical quantiles drop empty bins, so ``k`` can be
    smaller than ``n_bins``. Labels increase with the covariate. A constant
    covariate returns a single bin of zeros.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be an integer >= 1")
    array = _as_1d("values", values)
    if not _is_numeric_real(array):
        raise ValueError("values must be numeric")
    bins, _edges = _quantile_bins_with_edges(array, n_bins)
    return bins


def stratified_randomization(
    covariates: Mapping[str, np.ndarray],
    *,
    n_arms: int = 2,
    arm_labels: Sequence[str] | None = None,
    ratio: Sequence[float] | None = None,
    n_bins: int | Mapping[str, int] | None = None,
    seed: int | None = None,
) -> StratifiedAssignment:
    """Assign every unit to an arm by stratified randomization.

    Parameters
    ----------
    covariates:
        Mapping of covariate name to a 1-d array, one entry per unit.
        Strings, booleans, and integer codes are categorical levels.
        Integer-valued floats are treated as categorical codes. Any other
        floating-point covariate must be listed in ``n_bins`` (or covered
        by an integer ``n_bins``, which bins every continuous covariate).
    n_arms:
        Number of arms. At least 2, and no greater than the number of units.
    arm_labels:
        Label for each arm, in arm-index order. Defaults to ``control`` and
        ``treatment`` when there are two arms, otherwise ``arm_0``, ...
    ratio:
        Positive allocation weights, one per arm. ``None`` uses equal
        allocation. Within a stratum the counts are the largest-remainder
        rounding of these weights; tied remainders are broken at random.
    n_bins:
        Quantile bins for continuous covariates. An int applies to every
        continuous covariate. A mapping applies only to the named
        covariates (those must be numeric) and leaves other continuous
        covariates untouched, so they still need their own entry.
    seed:
        Seed for the within-stratum draw. ``None`` uses fresh entropy.

    Returns
    -------
    StratifiedAssignment
        Arm index per unit, stratum index per unit, and the arm counts
        inside each stratum.
    """
    if isinstance(n_arms, bool) or not isinstance(n_arms, int) or n_arms < 2:
        raise ValueError("n_arms must be an integer >= 2")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    labels = _arm_labels(n_arms, arm_labels)
    weights = _allocation_weights(n_arms, ratio)
    factors, bin_edges = _prepare_factors(covariates, n_bins)
    n = next(iter(factors.values())).size
    if n < n_arms:
        raise ValueError(f"need at least {n_arms} units to assign {n_arms} arms")

    strata, stratum_labels = _stratum_index(factors)
    rng = np.random.default_rng(seed)
    assignment = np.empty(n, dtype=int)
    counts_matrix = np.zeros((len(stratum_labels), n_arms), dtype=int)

    order = np.argsort(strata, kind="mergesort")
    sorted_strata = strata[order]
    breaks = np.flatnonzero(np.diff(sorted_strata)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    for start, end in zip(starts, ends):
        stratum_id = int(sorted_strata[start])
        index = order[start:end]
        counts = _allocate_counts(int(index.size), weights, rng)
        arm_ids = np.repeat(np.arange(n_arms, dtype=int), counts)
        rng.shuffle(arm_ids)
        assignment[index] = arm_ids
        counts_matrix[stratum_id] = counts

    copied_factors = {name: np.array(values, copy=True) for name, values in factors.items()}
    return StratifiedAssignment(
        assignment=assignment,
        strata=strata,
        stratum_labels=stratum_labels,
        arm_labels=labels,
        stratum_arm_counts=tuple(tuple(int(c) for c in row) for row in counts_matrix),
        factors=copied_factors,
        bin_edges=bin_edges,
        seed=seed,
    )


def standardized_mean_difference(
    values: np.ndarray,
    assignment: np.ndarray,
    *,
    arm: int = 1,
    reference_arm: int = 0,
    binary: bool = False,
) -> float:
    """Standardized mean difference of ``values`` between two arms.

    Returns ``(mean_arm - mean_reference) / pooled_sd``. The pooled
    standard deviation is the square root of the average of the two arm
    variances (Austin 2009), not the sample-size-weighted pooled variance.
    Continuous variances are unbiased (``ddof=1``) when an arm has at least
    two observations, and population variances otherwise. Binary variances
    are ``p (1 - p)``. The result is ``0`` when the means match and the
    pooled variance is ``0``, and signed infinity when the means differ and
    the pooled variance is ``0``.
    """
    if isinstance(arm, bool) or isinstance(reference_arm, bool):
        raise ValueError("arm ids must be integers")
    array = _numeric_1d("values", values)
    arms = _integer_assignment(assignment)
    if array.shape != arms.shape:
        raise ValueError("values and assignment must have the same length")
    group = array[arms == int(arm)]
    reference = array[arms == int(reference_arm)]
    if group.size == 0 or reference.size == 0:
        raise ValueError("both arms must contain at least one unit")
    mean_group = float(np.mean(group))
    mean_reference = float(np.mean(reference))
    difference = mean_group - mean_reference
    if binary:
        if not 0.0 <= mean_group <= 1.0 or not 0.0 <= mean_reference <= 1.0:
            raise ValueError("binary SMD requires arm means in [0, 1]")
        var_group = mean_group * (1.0 - mean_group)
        var_reference = mean_reference * (1.0 - mean_reference)
    else:
        ddof_group = 1 if group.size >= 2 else 0
        ddof_reference = 1 if reference.size >= 2 else 0
        var_group = float(np.var(group, ddof=ddof_group))
        var_reference = float(np.var(reference, ddof=ddof_reference))
    pooled = math.sqrt((var_group + var_reference) / 2.0)
    if pooled == 0.0:
        if difference == 0.0:
            return 0.0
        return math.copysign(math.inf, difference)
    return difference / pooled


def balance_report(
    covariates: Mapping[str, np.ndarray],
    assignment: np.ndarray,
    *,
    arm_labels: Sequence[str] | None = None,
    reference_arm: int = 0,
    categorical: Sequence[str] | None = None,
    continuous: Sequence[str] | None = None,
) -> BalanceReport:
    """Summarize covariate balance of a realized assignment.

    Numeric covariates are compared with a standardized mean difference
    against ``reference_arm``. Non-numeric covariates, and any name listed
    in ``categorical``, are compared with a chi-square test of independence
    plus per-level indicator SMDs. Names in ``continuous`` are forced onto
    the SMD path. Integer codes are numeric, so pass them in ``categorical``
    when they are group labels rather than quantities.
    """
    if not isinstance(covariates, Mapping) or isinstance(covariates, (str, bytes)):
        raise ValueError("covariates must be a mapping of name to 1-d values")
    if len(covariates) == 0:
        raise ValueError("at least one covariate is required")
    categorical_names = _name_set("categorical", categorical)
    continuous_names = _name_set("continuous", continuous)
    overlap = categorical_names & continuous_names
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"covariates cannot be both categorical and continuous: {joined}")
    unknown = (categorical_names | continuous_names) - set(covariates)
    if unknown:
        joined = ", ".join(repr(name) for name in sorted(unknown))
        raise ValueError(f"unknown covariate names: {joined}")

    arms = _integer_assignment(assignment)
    arm_ids = tuple(int(arm) for arm in np.unique(arms))
    if len(arm_ids) < 2:
        raise ValueError("assignment must contain at least two arms")
    if isinstance(reference_arm, bool) or not isinstance(reference_arm, int):
        raise ValueError("reference_arm must be an integer")
    if reference_arm not in arm_ids:
        raise ValueError("reference_arm is not present in assignment")
    labels = _balance_arm_labels(arm_ids, arm_labels)
    arm_counts = tuple(int(np.sum(arms == arm)) for arm in arm_ids)

    rows: list[BalanceRow] = []
    for name, values in covariates.items():
        if not isinstance(name, str) or not name:
            raise ValueError("covariate names must be non-empty strings")
        array = _as_1d(name, values)
        if array.shape != arms.shape:
            raise ValueError(f"{name} and assignment must have the same length")
        kind = _balance_kind(name, array, categorical_names, continuous_names)
        if kind == "continuous":
            rows.append(_continuous_row(name, array, arms, arm_ids, reference_arm))
        else:
            rows.append(_categorical_row(name, array, arms, arm_ids, reference_arm))

    finite = [row.max_abs_smd for row in rows if math.isfinite(row.max_abs_smd)]
    infinite = [row.max_abs_smd for row in rows if not math.isfinite(row.max_abs_smd)]
    if infinite:
        max_abs = math.inf
    elif finite:
        max_abs = max(finite)
    else:
        max_abs = 0.0
    return BalanceReport(
        rows=tuple(rows),
        max_abs_smd=max_abs,
        reference_arm=reference_arm,
        arm_ids=arm_ids,
        arm_labels=labels,
        arm_counts=arm_counts,
    )


def format_balance_report(report: BalanceReport) -> str:
    """Plain-text SMD / chi-square balance table."""
    ref_pos = report.arm_ids.index(report.reference_arm)
    ref_label = report.arm_labels[ref_pos]
    lines: list[str] = []
    for row in report.rows:
        if row.kind == "categorical":
            lines.append(
                f"  {row.covariate} (categorical): chi2={row.chi2:.3f} "
                f"df={row.dof} p={row.p_value:.4f} max|SMD|={_fmt_abs(row.max_abs_smd)}"
            )
            continue
        parts: list[str] = []
        for pos, (arm, smd) in enumerate(zip(report.arm_ids, row.smd_by_arm)):
            if arm == report.reference_arm:
                continue
            parts.append(f"SMD({report.arm_labels[pos]}-{ref_label})={_fmt_signed(smd)}")
        lines.append(
            f"  {row.covariate} (continuous): {', '.join(parts)} "
            f"max|SMD|={_fmt_abs(row.max_abs_smd)}"
        )
    lines.append(f"max |SMD|: {_fmt_abs(report.max_abs_smd)}")
    return "\n".join(lines)


def _fmt_signed(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.3f}"


def _fmt_abs(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{abs(value):.3f}"


def _arm_labels(n_arms: int, arm_labels: Sequence[str] | None) -> tuple[str, ...]:
    if arm_labels is None:
        if n_arms == 2:
            return ("control", "treatment")
        return tuple(f"arm_{i}" for i in range(n_arms))
    if isinstance(arm_labels, str):
        raise ValueError("arm_labels must be a sequence of strings")
    labels = tuple(arm_labels)
    if len(labels) != n_arms:
        raise ValueError("arm_labels must have one entry per arm")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("arm_labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("arm_labels must be unique")
    return labels


def _allocation_weights(n_arms: int, ratio: Sequence[float] | None) -> np.ndarray:
    if ratio is None:
        return np.ones(n_arms, dtype=float)
    if isinstance(ratio, str):
        raise ValueError("ratio must be a sequence of positive weights")
    weights = np.asarray(list(ratio), dtype=float)
    if weights.shape != (n_arms,):
        raise ValueError("ratio must have one positive weight per arm")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("ratio weights must be positive and finite")
    return weights


def _prepare_factors(
    covariates: Mapping[str, np.ndarray],
    n_bins: int | Mapping[str, int] | None,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, ...]]]:
    if not isinstance(covariates, Mapping) or isinstance(covariates, (str, bytes)):
        raise ValueError("covariates must be a mapping of name to 1-d values")
    if len(covariates) == 0:
        raise ValueError("at least one covariate is required")
    bin_spec = _normalize_bins(n_bins, covariates)
    factors: dict[str, np.ndarray] = {}
    edges: dict[str, tuple[float, ...]] = {}
    n: int | None = None
    for name, values in covariates.items():
        if not isinstance(name, str) or not name:
            raise ValueError("covariate names must be non-empty strings")
        array = _as_1d(name, values)
        if n is None:
            n = int(array.size)
        elif int(array.size) != n:
            raise ValueError("all covariates must have the same length")
        if name in bin_spec:
            bins, bin_edges = _quantile_bins_with_edges(array, bin_spec[name])
            if int(np.unique(bins).size) < 2:
                raise ValueError(
                    f"{name} does not have enough distinct values to form "
                    f"{bin_spec[name]} quantile bins"
                )
            factors[name] = bins
            edges[name] = bin_edges
        else:
            if _is_true_continuous(array):
                raise ValueError(
                    f"{name} looks continuous; pass n_bins to stratify on quantile bins"
                )
            factors[name] = np.array(array, copy=True)
    return factors, edges


def _normalize_bins(
    n_bins: int | Mapping[str, int] | None,
    covariates: Mapping[str, np.ndarray],
) -> dict[str, int]:
    if n_bins is None:
        return {}
    if isinstance(n_bins, bool):
        raise ValueError("n_bins must be an integer or a mapping of name to integer")
    if isinstance(n_bins, int):
        _validate_bin_count(n_bins)
        spec: dict[str, int] = {}
        for name, values in covariates.items():
            if _is_true_continuous(np.asarray(values)):
                spec[str(name)] = n_bins
        if not spec:
            raise ValueError("n_bins is an integer but no continuous covariates were found")
        return spec
    if isinstance(n_bins, Mapping):
        spec = {}
        for name, count in n_bins.items():
            if not isinstance(name, str) or name not in covariates:
                raise ValueError(f"n_bins key {name!r} is not a covariate")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(f"n_bins[{name!r}] must be an integer >= 2")
            _validate_bin_count(count)
            if not _is_numeric_real(np.asarray(covariates[name])):
                raise ValueError(f"{name} must be numeric to form quantile bins")
            spec[name] = count
        return spec
    raise ValueError("n_bins must be an integer or a mapping of name to integer")


def _validate_bin_count(n_bins: int) -> None:
    if n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2 for stratified randomization")


def _stratum_index(factors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = list(factors)
    codes: list[np.ndarray] = []
    level_labels: list[tuple[str, ...]] = []
    for name in names:
        factor_codes, labels = _factor_codes(factors[name])
        codes.append(factor_codes)
        level_labels.append(labels)
    n = codes[0].size
    rows = list(zip(*(code.tolist() for code in codes)))
    unique = sorted(set(rows))
    index = {row: i for i, row in enumerate(unique)}
    strata = np.fromiter((index[row] for row in rows), dtype=int, count=n)
    stratum_labels: list[str] = []
    for row in unique:
        parts = [
            f"{names[j]}={level_labels[j][row[j]]}"
            for j in range(len(names))
        ]
        stratum_labels.append(" | ".join(parts))
    return strata, tuple(stratum_labels)


def _factor_codes(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    python_values = [_python_scalar(value) for value in values]
    levels = sorted(set(python_values), key=_level_sort_key)
    index = {level: i for i, level in enumerate(levels)}
    codes = np.fromiter(
        (index[value] for value in python_values),
        dtype=int,
        count=len(python_values),
    )
    labels = tuple(_display_level(level) for level in levels)
    return codes, labels


def _python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _display_level(level: object) -> str:
    return str(level)


def _level_sort_key(value: object) -> tuple:
    if isinstance(value, (bool, np.bool_)):
        return (0, 0, int(value), "")
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return (0, 1, int(value), "")
    if isinstance(value, (float, np.floating)):
        return (0, 2, float(value), "")
    return (1, 0, 0, str(value))


def _allocate_counts(n: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Largest-remainder allocation of ``n`` units. Ties are random."""
    if n < 0:
        raise ValueError("n must be non-negative")
    scaled = np.asarray(weights, dtype=float)
    scaled = scaled / scaled.sum()
    quotas = scaled * n
    counts = np.floor(quotas + 1e-10).astype(int)
    leftover = int(n - counts.sum())
    jitter = rng.random(scaled.size)
    if leftover > 0:
        fractional = quotas - counts
        order = sorted(
            range(scaled.size),
            key=lambda i: (-float(fractional[i]), -float(jitter[i])),
        )
        for arm in order[:leftover]:
            counts[arm] += 1
    elif leftover < 0:
        fractional = quotas - counts
        order = sorted(
            range(scaled.size),
            key=lambda i: (float(fractional[i]), -float(jitter[i])),
        )
        for arm in order[:-leftover]:
            counts[arm] -= 1
    if int(counts.sum()) != n or np.any(counts < 0):
        raise RuntimeError("failed to allocate arm counts within a stratum")
    return counts


def _quantile_bins_with_edges(
    values: np.ndarray, n_bins: int
) -> tuple[np.ndarray, tuple[float, ...]]:
    numeric = np.asarray(values, dtype=float).reshape(-1)
    if numeric.size == 0:
        raise ValueError("values must be non-empty")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("continuous covariates must be finite")
    raw_edges = np.quantile(numeric, np.linspace(0.0, 1.0, n_bins + 1))
    edges = [float(raw_edges[0])]
    for edge in raw_edges[1:]:
        edge_f = float(edge)
        if edge_f > edges[-1]:
            edges.append(edge_f)
    if len(edges) == 1:
        return np.zeros(numeric.size, dtype=int), (edges[0], edges[0])
    interior = np.asarray(edges[1:-1], dtype=float)
    if interior.size == 0:
        return np.zeros(numeric.size, dtype=int), tuple(edges)
    bins = np.digitize(numeric, interior, right=False).astype(int)
    return bins, tuple(edges)


def _as_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real-valued or categorical")
    return array


def _numeric_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = _as_1d(name, values)
    if not _is_numeric_real(array):
        raise ValueError(f"{name} must be numeric")
    numeric = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must be finite")
    return numeric


def _is_numeric_real(array: np.ndarray) -> bool:
    return bool(
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    )


def _is_true_continuous(array: np.ndarray) -> bool:
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.floating):
        return False
    if array.size == 0:
        return False
    if not np.all(np.isfinite(array)):
        return True
    return not bool(np.all(array == np.floor(array)))


def _integer_assignment(assignment: np.ndarray) -> np.ndarray:
    array = np.asarray(assignment)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("assignment must be a non-empty 1-d array")
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("assignment must be integer arm ids")
        return array.astype(int)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("assignment must be integer arm ids")
    return array.astype(int, copy=False)


def _name_set(label: str, names: Sequence[str] | None) -> set[str]:
    if names is None:
        return set()
    if isinstance(names, str):
        raise ValueError(f"{label} must be a sequence of covariate names")
    result = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} entries must be non-empty strings")
        result.add(name)
    return result


def _balance_kind(
    name: str,
    array: np.ndarray,
    categorical_names: set[str],
    continuous_names: set[str],
) -> str:
    if name in categorical_names:
        return "categorical"
    if name in continuous_names:
        if not _is_numeric_real(array):
            raise ValueError(f"{name} must be numeric to report a standardized mean difference")
        return "continuous"
    if _is_numeric_real(array):
        return "continuous"
    return "categorical"


def _balance_arm_labels(
    arm_ids: tuple[int, ...],
    arm_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    if arm_labels is None:
        if arm_ids == (0, 1):
            return ("control", "treatment")
        return tuple(str(arm) for arm in arm_ids)
    if isinstance(arm_labels, str):
        raise ValueError("arm_labels must be a sequence of strings")
    labels = tuple(arm_labels)
    if len(labels) != len(arm_ids):
        raise ValueError("arm_labels must have one entry per observed arm")
    return labels


def _continuous_row(
    name: str,
    values: np.ndarray,
    assignment: np.ndarray,
    arm_ids: tuple[int, ...],
    reference_arm: int,
) -> BalanceRow:
    numeric = _numeric_1d(name, values)
    means = tuple(float(np.mean(numeric[assignment == arm])) for arm in arm_ids)
    smds: list[float] = []
    for arm in arm_ids:
        if arm == reference_arm:
            smds.append(0.0)
        else:
            smds.append(
                standardized_mean_difference(
                    numeric, assignment, arm=arm, reference_arm=reference_arm
                )
            )
    smd_tuple = tuple(smds)
    return BalanceRow(
        covariate=name,
        kind="continuous",
        smd_by_arm=smd_tuple,
        max_abs_smd=_max_abs(smd_tuple),
        chi2=None,
        p_value=None,
        dof=None,
        means=means,
        levels=None,
    )


def _categorical_row(
    name: str,
    values: np.ndarray,
    assignment: np.ndarray,
    arm_ids: tuple[int, ...],
    reference_arm: int,
) -> BalanceRow:
    codes, labels = _factor_codes(values)
    table = np.zeros((len(labels), len(arm_ids)), dtype=int)
    arm_pos = {arm: i for i, arm in enumerate(arm_ids)}
    positions = np.fromiter(
        (arm_pos[int(arm)] for arm in assignment),
        dtype=int,
        count=assignment.size,
    )
    np.add.at(table, (codes, positions), 1)
    chi2, p_value, dof = _chi_square(table)
    level_counts = tuple(
        (label, tuple(int(count) for count in table[i])) for i, label in enumerate(labels)
    )
    smd_by_arm = [0.0] * len(arm_ids)
    all_smds: list[float] = []
    ref_pos = arm_pos[reference_arm]
    for level in range(len(labels)):
        indicator = (codes == level).astype(float)
        for pos, arm in enumerate(arm_ids):
            if arm == reference_arm:
                continue
            smd = standardized_mean_difference(
                indicator,
                assignment,
                arm=arm,
                reference_arm=reference_arm,
                binary=True,
            )
            all_smds.append(smd)
            if abs(smd) >= abs(smd_by_arm[pos]):
                smd_by_arm[pos] = smd
    # Touch ref_pos so a one-arm edge still has an explicit zero at the reference.
    smd_by_arm[ref_pos] = 0.0
    smd_tuple = tuple(smd_by_arm)
    return BalanceRow(
        covariate=name,
        kind="categorical",
        smd_by_arm=smd_tuple,
        max_abs_smd=_max_abs(all_smds if all_smds else smd_tuple),
        chi2=chi2,
        p_value=p_value,
        dof=dof,
        means=None,
        levels=level_counts,
    )


def _chi_square(table: np.ndarray) -> tuple[float, float, int]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0, 0
    if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        return 0.0, 1.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chi2, p_value, dof, _expected = sp.chi2_contingency(table, correction=False)
    return float(chi2), float(p_value), int(dof)


def _max_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = 0.0
    for value in values:
        if math.isnan(value):
            return math.nan
        magnitude = abs(value)
        if math.isinf(magnitude):
            return math.inf
        if magnitude > peak:
            peak = magnitude
    return peak


__all__ = [
    "BalanceReport",
    "BalanceRow",
    "StratifiedAssignment",
    "balance_report",
    "format_balance_report",
    "quantile_bins",
    "standardized_mean_difference",
    "stratified_randomization",
]
