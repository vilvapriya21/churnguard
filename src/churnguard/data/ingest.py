"""Ingest stage: read the raw CSV, clean it, and split train / scoring files.

The raw Kaggle file has two problems that are fixed here, before any check or
model sees the data:

* Text cells may carry stray spaces, and ``TotalCharges`` is stored as text
  because 11 rows contain a blank (``" "``) instead of a number.
* Those blank ``TotalCharges`` rows all have ``tenure == 0``: they are brand-new
  customers who have not paid a bill yet. Their true total is therefore 0, so
  the blank is filled with ``0.0`` rather than a guessed average. Any blank on a
  customer with ``tenure > 0`` is left missing on purpose, so the missing-value
  gate in the validate stage can catch it.

The cleaned data is then split, stratified on the target, into a training file
and a "scoring" file that stands in for new customers arriving later.

Run as a pipeline stage with ``python -m churnguard.data.ingest``.

Exported API:
    load_raw_data: Read the raw CSV.
    clean_raw_data: Fix text and numeric types.
    split_train_scoring: Stratified train / scoring split.
    run_ingest: Run the whole stage between explicit file paths.
    main: Command-line entry point using the project settings.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import IngestParams, load_params, load_settings
from churnguard.data.columns import TARGET_COLUMN, TENURE, TOTAL_CHARGES
from churnguard.io_utils import write_csv
from churnguard.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def load_raw_data(path: Path, source_url: str) -> pd.DataFrame:
    """Read the raw dataset exactly as downloaded.

    Args:
        path: Location of the raw CSV.
        source_url: Where the file can be downloaded, shown if it is missing.

    Returns:
        The raw data with every column as read by pandas.

    Raises:
        FileNotFoundError: If the file is missing, with instructions to get it.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `dvc pull`, or download it from "
            f"{source_url} and place it at that path."
        )
    raw = pd.read_csv(path)
    logger.info("Loaded %d raw rows and %d columns from %s", *raw.shape, path)
    return raw


def clean_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise text cells and fix the type of ``TotalCharges``.

    Args:
        raw: Data as returned by :func:`load_raw_data`.

    Returns:
        A new data frame; the input is not modified. Text cells are stripped
        and empty strings become missing values. ``TotalCharges`` is numeric,
        with blanks for brand-new customers (``tenure == 0``) set to 0.
    """
    cleaned = raw.copy()

    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column]):
            stripped = cleaned[column].str.strip()
            cleaned[column] = stripped.mask(stripped == "")

    cleaned[TOTAL_CHARGES] = pd.to_numeric(cleaned[TOTAL_CHARGES], errors="coerce")
    is_new_customer_blank = cleaned[TOTAL_CHARGES].isna() & (cleaned[TENURE] == 0)
    cleaned.loc[is_new_customer_blank, TOTAL_CHARGES] = 0.0
    logger.info(
        "Filled %d blank %s values with 0 for customers with tenure 0",
        int(is_new_customer_blank.sum()),
        TOTAL_CHARGES,
    )
    return cleaned


def split_train_scoring(
    data: pd.DataFrame, scoring_size: float, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into a training file and a scoring file.

    The split is stratified on the target so both files keep the same churn
    rate, and seeded so it is identical on every run.

    Args:
        data: Cleaned data including the target column.
        scoring_size: Share of rows to put in the scoring file (0 to 1).
        random_seed: Seed that makes the split reproducible.

    Returns:
        A ``(train, scoring)`` pair of data frames.
    """
    train, scoring = train_test_split(
        data,
        test_size=scoring_size,
        random_state=random_seed,
        stratify=data[TARGET_COLUMN],
    )
    return train, scoring


def run_ingest(
    raw_path: Path,
    train_path: Path,
    scoring_path: Path,
    source_url: str,
    params: IngestParams,
    random_seed: int,
) -> None:
    """Run the ingest stage: load, clean, split and save.

    Args:
        raw_path: Raw CSV to read.
        train_path: Where to write the training file.
        scoring_path: Where to write the scoring file.
        source_url: Download location shown if the raw file is missing.
        params: Ingest parameters from ``params.yaml``.
        random_seed: Seed for the split.
    """
    cleaned = clean_raw_data(load_raw_data(raw_path, source_url))
    train, scoring = split_train_scoring(cleaned, params.scoring_size, random_seed)
    write_csv(train, train_path)
    write_csv(scoring, scoring_path)


def main() -> None:
    """Run the ingest stage with the project's settings and parameters."""
    settings = load_settings()
    configure_logging(settings.log_level)
    params = load_params()
    run_ingest(
        raw_path=settings.resolve(settings.paths.raw_data),
        train_path=settings.resolve(settings.paths.interim_train),
        scoring_path=settings.resolve(settings.paths.interim_scoring),
        source_url=settings.dataset_url,
        params=params.ingest,
        random_seed=params.base.random_seed,
    )


if __name__ == "__main__":
    main()
