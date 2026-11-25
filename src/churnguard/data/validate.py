"""Validate stage: run every data quality gate and write the report.

Checks, on the cleaned files from the ingest stage:

* **Schema contract** on the train and scoring files (``contracts.py``).
* **Missing values** per column against ``validate.max_missing_fraction``.
* **Drift** of the scoring file against the train file (``drift.py``).
* **Leakage**: no customer appears in both files, and neither the target nor
  the customer ID is a model feature.

The full result is written to ``reports/data_quality.json`` *before* the stage
decides, so the report is available even when the gate fails. The stage then
exits with an error if any check failed, which stops ``dvc repro`` and fails
the CI ``data-check`` job.

Run as a pipeline stage with ``python -m churnguard.data.validate``.

Exported API:
    check_leakage: Leakage checks between files and features.
    build_quality_report: Run every check and combine the results.
    run_validate: Run the stage between explicit file paths.
    main: Command-line entry point using the project settings.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from churnguard.config import ValidateParams, load_params, load_settings
from churnguard.data.checks import CheckResult
from churnguard.data.columns import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from churnguard.data.contracts import check_missing_values, check_schema
from churnguard.data.drift import check_drift
from churnguard.exceptions import DataValidationError
from churnguard.features.build_features import FEATURE_COLUMNS
from churnguard.io_utils import read_csv
from churnguard.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def check_leakage(train: pd.DataFrame, scoring: pd.DataFrame) -> CheckResult:
    """Check that the answer cannot leak into training.

    Args:
        train: Cleaned training rows.
        scoring: Cleaned scoring rows.

    Returns:
        A result that fails if a customer ID appears in both files, or if the
        target or ID column is listed as a model feature.
    """
    overlapping_ids = set(train[ID_COLUMN]) & set(scoring[ID_COLUMN])
    forbidden_features = sorted({TARGET_COLUMN, ID_COLUMN} & set(FEATURE_COLUMNS))
    return CheckResult(
        name="leakage",
        passed=not overlapping_ids and not forbidden_features,
        details={
            "overlapping_customer_ids": len(overlapping_ids),
            "forbidden_columns_in_features": forbidden_features,
        },
    )


def build_quality_report(
    train: pd.DataFrame, scoring: pd.DataFrame, params: ValidateParams
) -> dict[str, Any]:
    """Run every data quality check and combine the results.

    Args:
        train: Cleaned training rows.
        scoring: Cleaned scoring rows.
        params: Thresholds from ``params.yaml``.

    Returns:
        The report: overall ``passed`` flag, row counts, the names of failed
        checks, all warnings, and each check's details.
    """
    results = [
        check_schema(train, "train"),
        check_schema(scoring, "scoring"),
        check_missing_values(train, "train", params.max_missing_fraction),
        check_missing_values(scoring, "scoring", params.max_missing_fraction),
        check_drift(
            reference=train,
            current=scoring,
            numeric_columns=NUMERIC_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            bins=params.psi_bins,
            warn_limit=params.psi_warn,
            fail_limit=params.psi_fail,
        ),
        check_leakage(train, scoring),
    ]
    return {
        "passed": all(result.passed for result in results),
        "row_counts": {"train": len(train), "scoring": len(scoring)},
        "failed_checks": [result.name for result in results if not result.passed],
        "warnings": [warning for result in results for warning in result.warnings],
        "checks": {result.name: result.to_dict() for result in results},
    }


def run_validate(
    train_path: Path, scoring_path: Path, report_path: Path, params: ValidateParams
) -> dict[str, Any]:
    """Run the validate stage: check both files and write the JSON report.

    Args:
        train_path: Cleaned training file from the ingest stage.
        scoring_path: Cleaned scoring file from the ingest stage.
        report_path: Where to write ``data_quality.json``.
        params: Thresholds from ``params.yaml``.

    Returns:
        The report, if every check passed.

    Raises:
        DataValidationError: If any check failed. The report is written first.
    """
    report = build_quality_report(read_csv(train_path), read_csv(scoring_path), params)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote data quality report to %s", report_path)

    for warning in report["warnings"]:
        logger.warning("Drift warning: %s", warning)
    if not report["passed"]:
        raise DataValidationError(f"Data quality checks failed: {report['failed_checks']}")
    logger.info("All data quality checks passed")
    return report


def main() -> None:
    """Run the validate stage and exit with status 1 if a check fails."""
    settings = load_settings()
    configure_logging(settings.log_level)
    params = load_params()
    try:
        run_validate(
            train_path=settings.resolve(settings.paths.interim_train),
            scoring_path=settings.resolve(settings.paths.interim_scoring),
            report_path=settings.resolve(settings.paths.data_quality_report),
            params=params.validate_,
        )
    except DataValidationError as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
