"""Tests for the PSI drift check."""

import numpy as np
import pandas as pd
import pytest

from churnguard.data.drift import (
    categorical_psi,
    check_drift,
    population_stability_index,
)


@pytest.fixture
def reference() -> pd.Series:
    """Monthly bills from a stable period."""
    return pd.Series(np.random.default_rng(0).normal(65.0, 20.0, size=5000))


def test_population_stability_index_same_distribution_is_near_zero(
    reference: pd.Series,
) -> None:
    """Two samples from the same distribution give PSI below the warning level."""
    current = pd.Series(np.random.default_rng(1).normal(65.0, 20.0, size=2000))

    assert population_stability_index(reference, current, bins=10) < 0.1


def test_population_stability_index_shifted_distribution_is_large(
    reference: pd.Series,
) -> None:
    """Bills rising by a whole standard deviation give PSI above the failure level."""
    current = pd.Series(np.random.default_rng(1).normal(85.0, 20.0, size=2000))

    assert population_stability_index(reference, current, bins=10) > 0.25


def test_population_stability_index_ignores_missing_values(reference: pd.Series) -> None:
    """Blanks are skipped instead of breaking the calculation."""
    current = pd.concat([reference.iloc[:1000], pd.Series([np.nan] * 10)])

    assert np.isfinite(population_stability_index(reference, current, bins=10))


def test_categorical_psi_identical_shares_is_zero() -> None:
    """The same category shares give PSI of zero."""
    values = pd.Series(["A", "B", "B", "C"] * 50)

    assert categorical_psi(values, values.sample(frac=1, random_state=0)) == pytest.approx(0)


def test_categorical_psi_new_category_increases_psi() -> None:
    """A category that never appeared in training counts as drift."""
    reference = pd.Series(["Month-to-month"] * 50 + ["One year"] * 50)
    current = pd.Series(["Month-to-month"] * 50 + ["Three year"] * 50)

    assert categorical_psi(reference, current) > 0.25


def _frame(monthly: np.ndarray, contracts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"MonthlyCharges": monthly, "Contract": contracts})


def test_check_drift_stable_data_passes_without_warnings() -> None:
    """Stable data passes with every column marked ok."""
    rng = np.random.default_rng(0)
    train = _frame(rng.normal(65, 20, 3000), ["Month-to-month", "Two year"] * 1500)
    scoring = _frame(rng.normal(65, 20, 1000), ["Month-to-month", "Two year"] * 500)

    result = check_drift(train, scoring, ["MonthlyCharges"], ["Contract"], 10, 0.1, 0.25)

    assert result.passed
    assert not result.warnings
    assert result.details["columns"]["MonthlyCharges"]["status"] == "ok"


def test_check_drift_moderate_shift_warns_but_passes() -> None:
    """A PSI in the warning band is reported but does not fail the gate."""
    rng = np.random.default_rng(0)
    train = _frame(rng.normal(65, 20, 3000), ["Month-to-month"] * 3000)
    scoring = _frame(rng.normal(72, 20, 1000), ["Month-to-month"] * 1000)

    result = check_drift(train, scoring, ["MonthlyCharges"], ["Contract"], 10, 0.1, 0.25)

    assert result.details["columns"]["MonthlyCharges"]["status"] == "warn"
    assert result.passed
    assert len(result.warnings) == 1


def test_check_drift_large_shift_fails() -> None:
    """A PSI at or above the failure limit fails the gate and names the column."""
    rng = np.random.default_rng(0)
    train = _frame(rng.normal(65, 20, 3000), ["Month-to-month"] * 3000)
    scoring = _frame(rng.normal(95, 20, 1000), ["Month-to-month"] * 1000)

    result = check_drift(train, scoring, ["MonthlyCharges"], ["Contract"], 10, 0.1, 0.25)

    assert not result.passed
    assert result.details["failed_columns"] == ["MonthlyCharges"]
