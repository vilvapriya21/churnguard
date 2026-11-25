"""Tests for the schema contract and the missing-value gate."""

import numpy as np
import pandas as pd
import pytest

from churnguard.data.columns import ID_COLUMN, MONTHLY_CHARGES, TENURE, TOTAL_CHARGES
from churnguard.data.contracts import check_missing_values, check_schema


def test_check_schema_clean_data_passes(clean_customers: pd.DataFrame) -> None:
    """Data produced by the ingest stage satisfies the contract."""
    result = check_schema(clean_customers, "train")

    assert result.passed, result.details


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("Contract", "Three year"),
        (TENURE, 80),
        (MONTHLY_CHARGES, -5.0),
        ("gender", "Unknown"),
    ],
)
def test_check_schema_value_outside_contract_fails(
    clean_customers: pd.DataFrame, column: str, bad_value: str | float
) -> None:
    """An unknown category or out-of-range number breaks the contract."""
    clean_customers.loc[0, column] = bad_value

    result = check_schema(clean_customers, "train")

    assert not result.passed
    assert any(failure["column"] == column for failure in result.details["failures"])


def test_check_schema_duplicate_customer_id_fails(clean_customers: pd.DataFrame) -> None:
    """Every customer ID must be unique."""
    clean_customers.loc[1, ID_COLUMN] = str(clean_customers[ID_COLUMN].iloc[0])

    result = check_schema(clean_customers, "train")

    assert not result.passed


def test_check_schema_missing_column_fails(clean_customers: pd.DataFrame) -> None:
    """A delivery without an expected column is rejected."""
    result = check_schema(clean_customers.drop(columns=["Contract"]), "train")

    assert not result.passed


def test_check_schema_extra_column_fails(clean_customers: pd.DataFrame) -> None:
    """The contract is strict: unexpected columns are rejected too."""
    clean_customers["surprise"] = 1

    result = check_schema(clean_customers, "train")

    assert not result.passed


def test_check_missing_values_below_limit_passes(clean_customers: pd.DataFrame) -> None:
    """A few blanks under the limit are reported but do not fail."""
    clean_customers.loc[:2, TOTAL_CHARGES] = np.nan  # 3 of 300 rows = 1%

    result = check_missing_values(clean_customers, "train", max_missing_fraction=0.05)

    assert result.passed
    assert result.details["columns_with_blanks"][TOTAL_CHARGES]["count"] == 3


def test_check_missing_values_above_limit_fails(clean_customers: pd.DataFrame) -> None:
    """A column with more than 5% blanks fails the gate."""
    clean_customers.loc[:29, TOTAL_CHARGES] = np.nan  # 30 of 300 rows = 10%

    result = check_missing_values(clean_customers, "train", max_missing_fraction=0.05)

    assert not result.passed
    assert result.details["columns_over_limit"] == [TOTAL_CHARGES]
