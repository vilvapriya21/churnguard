"""Feature building shared by training and serving.

This module is the answer to "train/serve consistency": the training pipeline
and the API both call the same :func:`build_features`, and the fitted
preprocessing steps (imputing, scaling, one-hot encoding) are saved *inside*
the model pipeline. A customer row therefore turns into exactly the same
numbers whether it comes from the CSV or from an API request.

It is also where leakage is prevented: :func:`build_features` keeps only known
input columns, so ``customerID`` and ``Churn`` can never reach the model, and
the train/test split happens in :func:`run_featurize` *before* anything is
fitted.

Run as a pipeline stage with ``python -m churnguard.features.build_features``.

Exported API:
    FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES: Model inputs.
    build_features: Raw customer rows to model input columns.
    encode_target: ``Churn`` Yes/No to 1/0.
    build_preprocessor: Unfitted imputing, scaling and encoding steps.
    run_featurize: Split and featurise the training file.
    main: Command-line entry point using the project settings.
"""

import logging
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard.config import SplitParams, load_params, load_settings
from churnguard.data.columns import (
    CATEGORICAL_COLUMNS,
    MONTHLY_CHARGES,
    NUMERIC_COLUMNS,
    POSITIVE_LABEL,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TENURE,
    TOTAL_CHARGES,
)
from churnguard.io_utils import read_csv, write_csv
from churnguard.logging_utils import configure_logging

logger = logging.getLogger(__name__)

AVG_MONTHLY_CHARGE: Final = "avg_monthly_charge"

NUMERIC_FEATURES: Final[tuple[str, ...]] = (*NUMERIC_COLUMNS, AVG_MONTHLY_CHARGE)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = CATEGORICAL_COLUMNS
FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer rows into the model's input columns.

    Works for a full training file and for a single API request alike. Input
    columns that are absent (for example, fields an API caller did not send)
    are added as missing values and filled later by the fitted imputers.

    Args:
        customers: Rows with any subset of the raw dataset columns. Extra
            columns such as ``customerID`` and ``Churn`` are ignored.

    Returns:
        A new data frame with exactly :data:`FEATURE_COLUMNS`, in that order,
        and the same index as the input.
    """
    features = customers.reindex(columns=list(RAW_FEATURE_COLUMNS))

    # Every numeric input is a float, whether it arrives as 2 or 2.0 or is
    # missing, so the model signature has one stable type per column.
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype(object)

    # Average bill per month of tenure. New customers (tenure 0) have no
    # history yet, so their current monthly bill is the best estimate.
    tenure = features[TENURE]
    features[AVG_MONTHLY_CHARGE] = (features[TOTAL_CHARGES] / tenure).where(
        tenure > 0, features[MONTHLY_CHARGES]
    )
    return features[list(FEATURE_COLUMNS)]


def encode_target(labels: pd.Series) -> pd.Series:
    """Convert the ``Churn`` labels to integers.

    Args:
        labels: Values ``"Yes"`` or ``"No"``.

    Returns:
        ``1`` where the customer churned, otherwise ``0``.
    """
    return (labels == POSITIVE_LABEL).astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) preprocessing steps for the model pipeline.

    Numeric columns: fill gaps with the median, then standardise.
    Categorical columns: fill gaps with the most frequent value, then one-hot
    encode; categories never seen in training are ignored instead of failing.

    Returns:
        A column transformer to be fitted on training data only.
    """
    numeric_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_steps, list(NUMERIC_FEATURES)),
            ("categorical", categorical_steps, list(CATEGORICAL_FEATURES)),
        ]
    )


def split_and_featurize(
    data: pd.DataFrame, params: SplitParams, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labelled rows into train and test sets, then build features.

    The split comes first so no information from the test rows can influence
    anything learned from the training rows.

    Args:
        data: Cleaned rows including the ``Churn`` column.
        params: Split parameters from ``params.yaml``.
        random_seed: Seed that makes the split reproducible.

    Returns:
        A ``(train, test)`` pair. Each holds :data:`FEATURE_COLUMNS` plus the
        target encoded as 0/1.
    """
    train_rows, test_rows = train_test_split(
        data,
        test_size=params.test_size,
        random_state=random_seed,
        stratify=data[TARGET_COLUMN],
    )

    def _featurize(rows: pd.DataFrame) -> pd.DataFrame:
        features = build_features(rows)
        features[TARGET_COLUMN] = encode_target(rows[TARGET_COLUMN])
        return features

    return _featurize(train_rows), _featurize(test_rows)


def run_featurize(
    interim_train_path: Path,
    processed_train_path: Path,
    processed_test_path: Path,
    params: SplitParams,
    random_seed: int,
) -> None:
    """Run the featurize stage: read, split, build features and save.

    Args:
        interim_train_path: Cleaned training file from the ingest stage.
        processed_train_path: Where to write the training features.
        processed_test_path: Where to write the held-out test features.
        params: Split parameters from ``params.yaml``.
        random_seed: Seed for the split.
    """
    train, test = split_and_featurize(read_csv(interim_train_path), params, random_seed)
    write_csv(train, processed_train_path)
    write_csv(test, processed_test_path)


def main() -> None:
    """Run the featurize stage with the project's settings and parameters."""
    settings = load_settings()
    configure_logging(settings.log_level)
    params = load_params()
    run_featurize(
        interim_train_path=settings.resolve(settings.paths.interim_train),
        processed_train_path=settings.resolve(settings.paths.processed_train),
        processed_test_path=settings.resolve(settings.paths.processed_test),
        params=params.split,
        random_seed=params.base.random_seed,
    )


if __name__ == "__main__":
    main()
