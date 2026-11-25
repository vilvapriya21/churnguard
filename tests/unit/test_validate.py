"""Tests for the validate stage and the leakage check."""

import json
from pathlib import Path

import pandas as pd
import pytest

from churnguard.config import Params
from churnguard.data.columns import ID_COLUMN
from churnguard.data.ingest import clean_raw_data, split_train_scoring
from churnguard.data.validate import build_quality_report, check_leakage, run_validate
from churnguard.exceptions import DataValidationError
from tests.conftest import make_raw_customers


@pytest.fixture
def train_and_scoring(params: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fake cleaned customers split like the ingest stage does.

    PSI is noisy on small samples, so this uses a realistic number of rows
    (3,000) to keep the drift check meaningful.
    """
    customers = clean_raw_data(make_raw_customers(n_rows=3000))
    return split_train_scoring(customers, params.ingest.scoring_size, random_seed=0)


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_check_leakage_separate_files_pass(
    train_and_scoring: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Disjoint train and scoring customers, and clean feature list, pass."""
    train, scoring = train_and_scoring

    assert check_leakage(train, scoring).passed


def test_check_leakage_customer_in_both_files_fails(
    train_and_scoring: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A customer appearing in both files is flagged as leakage."""
    train, scoring = train_and_scoring
    scoring = scoring.copy()
    scoring.loc[scoring.index[0], ID_COLUMN] = str(train[ID_COLUMN].iloc[0])

    result = check_leakage(train, scoring)

    assert not result.passed
    assert result.details["overlapping_customer_ids"] == 1


def test_build_quality_report_clean_data_passes_every_check(
    train_and_scoring: tuple[pd.DataFrame, pd.DataFrame], params: Params
) -> None:
    """Good data passes all six checks."""
    report = build_quality_report(*train_and_scoring, params.validate_)

    assert report["passed"], report["failed_checks"]
    assert set(report["checks"]) == {
        "schema_train",
        "schema_scoring",
        "missing_train",
        "missing_scoring",
        "drift",
        "leakage",
    }


def test_run_validate_good_data_writes_passing_report(
    tmp_path: Path,
    train_and_scoring: tuple[pd.DataFrame, pd.DataFrame],
    params: Params,
) -> None:
    """The stage writes a JSON report marked as passed."""
    train, scoring = train_and_scoring
    report_path = tmp_path / "reports" / "data_quality.json"

    run_validate(
        _write(train, tmp_path / "train.csv"),
        _write(scoring, tmp_path / "scoring.csv"),
        report_path,
        params.validate_,
    )

    assert json.loads(report_path.read_text())["passed"] is True


def test_run_validate_bad_data_writes_report_then_raises(
    tmp_path: Path,
    train_and_scoring: tuple[pd.DataFrame, pd.DataFrame],
    params: Params,
) -> None:
    """Bad data stops the stage, but the report explaining why is still written."""
    train, scoring = train_and_scoring
    scoring = scoring.copy()
    scoring["Contract"] = "Three year"
    report_path = tmp_path / "data_quality.json"

    with pytest.raises(DataValidationError, match="schema_scoring"):
        run_validate(
            _write(train, tmp_path / "train.csv"),
            _write(scoring, tmp_path / "scoring.csv"),
            report_path,
            params.validate_,
        )

    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert "schema_scoring" in report["failed_checks"]
