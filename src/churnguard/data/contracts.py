"""Schema contract and missing-value gate for the cleaned customer data.

A *schema contract* is a written-down promise about what the data looks like:
which columns exist, their types, which values are allowed, valid ranges and
uniqueness. If a new data delivery breaks the promise, the pipeline stops
before a bad model can be trained on it.

The contract applies to the output of the ingest stage, where text has been
stripped and ``TotalCharges`` is already numeric. ``TotalCharges`` may still be
missing for existing customers; that is the missing-value gate's job, so the
contract allows it and :func:`check_missing_values` decides.

Exported API:
    ALLOWED_VALUES: Allowed categories for every text column.
    build_customer_schema: The pandera contract.
    check_schema: Validate a data frame against the contract.
    check_missing_values: Fail if any column has too many blanks.
"""

from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from churnguard.data.checks import CheckResult
from churnguard.data.columns import (
    ID_COLUMN,
    MAX_TENURE_MONTHS,
    MONTHLY_CHARGES,
    SENIOR_CITIZEN,
    TARGET_COLUMN,
    TENURE,
    TOTAL_CHARGES,
)

MAX_REPORTED_FAILURES: Final = 20

_YES_NO: Final = ("Yes", "No")
_INTERNET_ADDON: Final = ("Yes", "No", "No internet service")

ALLOWED_VALUES: Final[dict[str, tuple[str, ...]]] = {
    "gender": ("Female", "Male"),
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
    TARGET_COLUMN: _YES_NO,
}


def build_customer_schema() -> pa.DataFrameSchema:
    """Build the pandera contract for cleaned customer rows.

    Returns:
        A strict schema: extra or missing columns are errors, and values are
        checked, not converted.
    """
    columns: dict[str, pa.Column] = {
        ID_COLUMN: pa.Column(str, unique=True, nullable=False),
        SENIOR_CITIZEN: pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        TENURE: pa.Column(int, pa.Check.in_range(0, MAX_TENURE_MONTHS), nullable=False),
        MONTHLY_CHARGES: pa.Column(float, pa.Check.gt(0), nullable=False),
        TOTAL_CHARGES: pa.Column(float, pa.Check.ge(0), nullable=True),
    }
    for column, allowed in ALLOWED_VALUES.items():
        columns[column] = pa.Column(str, pa.Check.isin(list(allowed)), nullable=False)

    return pa.DataFrameSchema(columns, strict=True, coerce=False, name="customers")


def check_schema(data: pd.DataFrame, dataset_name: str) -> CheckResult:
    """Validate data against the schema contract, collecting every failure.

    Args:
        data: Cleaned customer rows.
        dataset_name: Label used in the report, e.g. ``"train"``.

    Returns:
        A failed result listing up to :data:`MAX_REPORTED_FAILURES` problems
        if the contract is broken, otherwise a passed result.
    """
    try:
        build_customer_schema().validate(data, lazy=True)
    except SchemaErrors as errors:
        failures = errors.failure_cases.head(MAX_REPORTED_FAILURES)
        failure_records = [
            {
                "column": str(row["column"]),
                "check": str(row["check"]),
                "value": str(row["failure_case"]),
            }
            for _, row in failures.iterrows()
        ]
        return CheckResult(
            name=f"schema_{dataset_name}",
            passed=False,
            details={
                "failure_count": len(errors.failure_cases),
                "failures": failure_records,
            },
        )
    return CheckResult(name=f"schema_{dataset_name}", passed=True, details={"failure_count": 0})


def check_missing_values(
    data: pd.DataFrame, dataset_name: str, max_missing_fraction: float
) -> CheckResult:
    """Count blanks per column and fail if any column exceeds the limit.

    Args:
        data: Cleaned customer rows.
        dataset_name: Label used in the report, e.g. ``"train"``.
        max_missing_fraction: Highest allowed share of missing values per
            column, between 0 and 1.

    Returns:
        A result with the missing count and fraction for every column that has
        any blanks, and the list of columns over the limit.
    """
    missing_counts = data.isna().sum()
    row_count = max(len(data), 1)
    columns_with_blanks = {
        str(column): {"count": int(count), "fraction": round(float(count) / row_count, 6)}
        for column, count in missing_counts.items()
        if count > 0
    }
    over_limit = sorted(
        column
        for column, stats in columns_with_blanks.items()
        if stats["fraction"] > max_missing_fraction
    )
    return CheckResult(
        name=f"missing_{dataset_name}",
        passed=not over_limit,
        details={
            "max_missing_fraction": max_missing_fraction,
            "columns_with_blanks": columns_with_blanks,
            "columns_over_limit": over_limit,
        },
    )
