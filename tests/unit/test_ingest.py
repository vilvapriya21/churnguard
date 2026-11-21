"""Tests for the ingest stage."""

import re
from pathlib import Path

import pandas as pd
import pytest

from churnguard.config import Params
from churnguard.data.columns import TARGET_COLUMN, TENURE, TOTAL_CHARGES
from churnguard.data.ingest import (
    clean_raw_data,
    load_raw_data,
    run_ingest,
    split_train_scoring,
)


def test_load_raw_data_missing_file_raises_with_download_hint(tmp_path: Path) -> None:
    """A missing raw file explains where to get the data."""
    with pytest.raises(FileNotFoundError, match=re.escape("https://example.com/data")):
        load_raw_data(tmp_path / "missing.csv", "https://example.com/data")


def test_load_raw_data_existing_file_returns_all_rows(raw_csv: Path) -> None:
    """Every row of the raw file is loaded."""
    assert len(load_raw_data(raw_csv, "unused")) == 300


def test_clean_raw_data_blank_total_charges_for_new_customers_become_zero(
    raw_customers: pd.DataFrame,
) -> None:
    """Blank TotalCharges on tenure-0 customers is filled with 0."""
    cleaned = clean_raw_data(raw_customers)

    new_customers = cleaned[cleaned[TENURE] == 0]
    assert not new_customers.empty
    assert (new_customers[TOTAL_CHARGES] == 0.0).all()


def test_clean_raw_data_total_charges_becomes_numeric(raw_customers: pd.DataFrame) -> None:
    """TotalCharges is converted from text to a number column."""
    cleaned = clean_raw_data(raw_customers)

    assert pd.api.types.is_float_dtype(cleaned[TOTAL_CHARGES])


def test_clean_raw_data_blank_total_charges_with_tenure_stays_missing(
    raw_customers: pd.DataFrame,
) -> None:
    """A blank on an existing customer is left for the missing-value gate."""
    raw_customers.loc[0, TENURE] = 10
    raw_customers.loc[0, TOTAL_CHARGES] = " "

    cleaned = clean_raw_data(raw_customers)

    assert cleaned[TOTAL_CHARGES].isna().iloc[0]


def test_clean_raw_data_strips_spaces_from_text(raw_customers: pd.DataFrame) -> None:
    """Leading and trailing spaces are removed from text cells."""
    raw_customers.loc[0, "Contract"] = "  Two year "

    cleaned = clean_raw_data(raw_customers)

    assert cleaned.loc[0, "Contract"] == "Two year"


def test_clean_raw_data_does_not_modify_input(raw_customers: pd.DataFrame) -> None:
    """Cleaning returns a new frame and leaves the input untouched."""
    original = raw_customers.copy()

    clean_raw_data(raw_customers)

    pd.testing.assert_frame_equal(raw_customers, original)


def test_split_train_scoring_sizes_and_churn_rate_match(
    clean_customers: pd.DataFrame,
) -> None:
    """The split has the requested size and keeps the churn rate in both parts."""
    train, scoring = split_train_scoring(clean_customers, scoring_size=0.2, random_seed=1)

    assert len(scoring) == 60
    assert len(train) == 240
    churn_rate_train = (train[TARGET_COLUMN] == "Yes").mean()
    churn_rate_scoring = (scoring[TARGET_COLUMN] == "Yes").mean()
    assert abs(churn_rate_train - churn_rate_scoring) < 0.05


def test_split_train_scoring_same_seed_gives_same_split(
    clean_customers: pd.DataFrame,
) -> None:
    """The split is reproducible for a fixed seed."""
    first, _ = split_train_scoring(clean_customers, 0.2, random_seed=7)
    second, _ = split_train_scoring(clean_customers, 0.2, random_seed=7)

    assert first.index.equals(second.index)


def test_run_ingest_writes_train_and_scoring_files(
    tmp_path: Path, raw_csv: Path, params: Params
) -> None:
    """The stage writes both output files, together holding every row."""
    train_path = tmp_path / "interim" / "train.csv"
    scoring_path = tmp_path / "interim" / "scoring.csv"

    run_ingest(raw_csv, train_path, scoring_path, "unused", params.ingest, 42)

    total_rows = len(pd.read_csv(train_path)) + len(pd.read_csv(scoring_path))
    assert total_rows == 300
