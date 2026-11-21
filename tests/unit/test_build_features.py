"""Tests for feature building, including leakage and train/serve consistency."""

from pathlib import Path

import pandas as pd

from churnguard.config import Params
from churnguard.data.columns import ID_COLUMN, TARGET_COLUMN
from churnguard.features.build_features import (
    AVG_MONTHLY_CHARGE,
    FEATURE_COLUMNS,
    build_features,
    build_preprocessor,
    encode_target,
    run_featurize,
    split_and_featurize,
)


def test_build_features_returns_exact_feature_columns_in_order(
    clean_customers: pd.DataFrame,
) -> None:
    """The output always has the same columns in the same order."""
    features = build_features(clean_customers)

    assert tuple(features.columns) == FEATURE_COLUMNS


def test_build_features_excludes_target_and_id_columns(
    clean_customers: pd.DataFrame,
) -> None:
    """Leakage guard: the answer and the customer ID never become features."""
    features = build_features(clean_customers)

    assert TARGET_COLUMN not in features.columns
    assert ID_COLUMN not in features.columns
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_build_features_missing_input_columns_become_nan() -> None:
    """A partial row (like a short API request) still gives every feature column."""
    partial = pd.DataFrame([{"tenure": 2, "Contract": "Month-to-month", "MonthlyCharges": 85.5}])

    features = build_features(partial)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features["gender"].isna().all()


def test_build_features_avg_monthly_charge_uses_monthly_bill_for_new_customer() -> None:
    """For tenure 0 the average falls back to the current monthly bill."""
    customers = pd.DataFrame(
        [
            {"tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": 0.0},
            {"tenure": 10, "MonthlyCharges": 50.0, "TotalCharges": 400.0},
        ]
    )

    features = build_features(customers)

    assert features[AVG_MONTHLY_CHARGE].tolist() == [50.0, 40.0]


def test_build_features_same_customer_csv_and_api_rows_match(
    clean_customers: pd.DataFrame,
) -> None:
    """Train/serve consistency: a CSV row and the same row sent as JSON match."""
    from_training_file = build_features(clean_customers.iloc[[0]]).reset_index(drop=True)
    api_payload = clean_customers.iloc[0].to_dict()
    from_api_request = build_features(pd.DataFrame([api_payload]))

    pd.testing.assert_frame_equal(from_training_file, from_api_request, check_dtype=False)


def test_encode_target_maps_yes_to_one_and_no_to_zero() -> None:
    """Churn labels become 1 for Yes and 0 for No."""
    encoded = encode_target(pd.Series(["Yes", "No", "Yes"]))

    assert encoded.tolist() == [1, 0, 1]


def test_build_preprocessor_handles_unseen_category(clean_customers: pd.DataFrame) -> None:
    """A category never seen in training does not break prediction-time transform."""
    preprocessor = build_preprocessor().fit(build_features(clean_customers))
    unseen = clean_customers.iloc[[0]].copy()
    unseen["Contract"] = "Three year"

    transformed = preprocessor.transform(build_features(unseen))

    known = preprocessor.transform(build_features(clean_customers.iloc[[0]]))
    assert transformed.shape == known.shape


def test_split_and_featurize_outputs_have_features_and_encoded_target(
    clean_customers: pd.DataFrame, params: Params
) -> None:
    """Both splits hold the feature columns plus a 0/1 target, with no overlap."""
    train, test = split_and_featurize(clean_customers, params.split, random_seed=0)

    assert tuple(train.columns) == (*FEATURE_COLUMNS, TARGET_COLUMN)
    assert set(train[TARGET_COLUMN].unique()) <= {0, 1}
    assert train.index.intersection(test.index).empty


def test_run_featurize_writes_processed_files(
    tmp_path: Path, clean_customers: pd.DataFrame, params: Params
) -> None:
    """The stage reads the interim file and writes train and test features."""
    interim_path = tmp_path / "interim.csv"
    clean_customers.to_csv(interim_path, index=False)
    train_path = tmp_path / "processed" / "train.csv"
    test_path = tmp_path / "processed" / "test.csv"

    run_featurize(interim_path, train_path, test_path, params.split, random_seed=0)

    assert len(pd.read_csv(train_path)) + len(pd.read_csv(test_path)) == len(clean_customers)
