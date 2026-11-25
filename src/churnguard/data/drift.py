"""Distribution (drift) check between the training and scoring data.

*Drift* means new data no longer looks like the data the model learned from,
for example the average monthly bill rising a lot. A model trained on old
patterns then makes worse predictions without any error being raised.

The check uses the Population Stability Index (PSI). For each column, values
are put into bins defined on the reference (training) data, and PSI adds up
how much each bin's share changed:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Common reading: below 0.1 no meaningful change, 0.1 to 0.25 a moderate shift
worth a look (warning), above 0.25 a major shift (failure). The limits come
from ``params.yaml``.

Exported API:
    DriftStatus: ok / warn / fail for one column.
    population_stability_index: PSI for a numeric column.
    categorical_psi: PSI for a text column, one bin per category.
    check_drift: Compare every feature column and build a check result.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from churnguard.data.checks import CheckResult

# Stops log(0) and division by zero when a bin is empty on one side.
_EPSILON: Final = 1e-6


class DriftStatus(StrEnum):
    """Drift verdict for one column."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    """Compute PSI from two arrays of bin shares that each sum to 1.

    Args:
        expected: Bin shares in the reference data.
        actual: Bin shares in the new data, same bin order.

    Returns:
        The PSI value (0 means identical distributions).
    """
    expected = np.clip(expected, _EPSILON, None)
    actual = np.clip(actual, _EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int) -> float:
    """Compute PSI for a numeric column using quantile bins of the reference.

    Args:
        reference: Values from the training data.
        current: Values from the new (scoring) data.
        bins: Number of quantile bins. Fewer are used if the reference has
            many repeated values.

    Returns:
        The PSI value. Missing values are ignored.
    """
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)

    inner_edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1))[1:-1])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))

    expected_counts, _ = np.histogram(reference_values, bins=edges)
    actual_counts, _ = np.histogram(current_values, bins=edges)
    return _psi_from_shares(
        expected_counts / max(expected_counts.sum(), 1),
        actual_counts / max(actual_counts.sum(), 1),
    )


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Compute PSI for a text column, using one bin per category.

    Args:
        reference: Values from the training data.
        current: Values from the new (scoring) data.

    Returns:
        The PSI value. Categories present on only one side still count.
    """
    expected = reference.value_counts(normalize=True)
    actual = current.value_counts(normalize=True)
    categories = expected.index.union(actual.index)
    return _psi_from_shares(
        expected.reindex(categories, fill_value=0.0).to_numpy(dtype=float),
        actual.reindex(categories, fill_value=0.0).to_numpy(dtype=float),
    )


def _status(psi: float, warn_limit: float, fail_limit: float) -> DriftStatus:
    """Classify a PSI value against the warning and failure limits."""
    if psi >= fail_limit:
        return DriftStatus.FAIL
    if psi >= warn_limit:
        return DriftStatus.WARN
    return DriftStatus.OK


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    bins: int,
    warn_limit: float,
    fail_limit: float,
) -> CheckResult:
    """Compare every listed column between reference and current data.

    Args:
        reference: Training data.
        current: New data to compare, e.g. the scoring file.
        numeric_columns: Columns compared with quantile-binned PSI.
        categorical_columns: Columns compared with per-category PSI.
        bins: Number of bins for numeric columns.
        warn_limit: PSI at or above which a column gets a warning.
        fail_limit: PSI at or above which the check fails.

    Returns:
        A result with each column's PSI and status. It fails if any column
        reaches ``fail_limit``; columns in the warning band are listed in
        ``warnings`` but do not fail the check.
    """
    per_column: dict[str, dict[str, float | str]] = {}
    for column in numeric_columns:
        value = population_stability_index(reference[column], current[column], bins)
        per_column[column] = {"psi": round(value, 6)}
    for column in categorical_columns:
        value = categorical_psi(reference[column], current[column])
        per_column[column] = {"psi": round(value, 6)}

    warnings: list[str] = []
    failed: list[str] = []
    for column, stats in per_column.items():
        status = _status(float(stats["psi"]), warn_limit, fail_limit)
        stats["status"] = status.value
        if status is DriftStatus.FAIL:
            failed.append(column)
        elif status is DriftStatus.WARN:
            warnings.append(f"{column}: PSI {stats['psi']} is above the warning limit")

    return CheckResult(
        name="drift",
        passed=not failed,
        details={
            "warn_limit": warn_limit,
            "fail_limit": fail_limit,
            "failed_columns": failed,
            "columns": per_column,
        },
        warnings=warnings,
    )
