"""Tests for loading a model and predicting churn."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churnguard.config import Params
from churnguard.data.columns import TARGET_COLUMN
from churnguard.exceptions import ModelNotFoundError
from churnguard.features.build_features import FEATURE_COLUMNS, split_and_featurize
from churnguard.models.predict import (
    DECISION_COLUMN,
    PROBABILITY_COLUMN,
    load_model,
    predict_churn,
    predict_proba,
)
from churnguard.models.train import build_model_pipeline


@pytest.fixture
def fitted_model(clean_customers: pd.DataFrame, params: Params) -> Pipeline:
    """A logistic regression pipeline fitted on the fake customers."""
    train, _ = split_and_featurize(clean_customers, params.split, random_seed=0)
    model = build_model_pipeline(
        "logistic_regression", params.train.models["logistic_regression"], random_seed=0
    )
    return model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])


def test_load_model_missing_file_raises_model_not_found(tmp_path: Path) -> None:
    """A missing model file gives a clear ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError, match="Run the train stage"):
        load_model(tmp_path / "missing.joblib")


def test_load_model_non_pipeline_file_raises_model_not_found(tmp_path: Path) -> None:
    """A file that is not a scikit-learn Pipeline is rejected."""
    path = tmp_path / "not_a_model.joblib"
    joblib.dump({"weights": [1, 2, 3]}, path)

    with pytest.raises(ModelNotFoundError, match="not contain"):
        load_model(path)


def test_predict_proba_returns_one_probability_per_row(
    fitted_model: Pipeline, clean_customers: pd.DataFrame
) -> None:
    """Every raw customer row gets a probability between 0 and 1."""
    probabilities = predict_proba(fitted_model, clean_customers)

    assert probabilities.shape == (len(clean_customers),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_proba_accepts_partial_api_style_input(fitted_model: Pipeline) -> None:
    """The example request from the assignment, with only six fields, is scored."""
    request = pd.DataFrame(
        [
            {
                "tenure": 2,
                "Contract": "Month-to-month",
                "MonthlyCharges": 85.5,
                "TotalCharges": 171.0,
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
            }
        ]
    )

    probabilities = predict_proba(fitted_model, request)

    assert 0 <= probabilities[0] <= 1


def test_predict_churn_decision_follows_threshold(
    fitted_model: Pipeline, clean_customers: pd.DataFrame
) -> None:
    """A customer is flagged exactly when their probability reaches the threshold."""
    predictions = predict_churn(fitted_model, clean_customers, threshold=0.5)

    expected = predictions[PROBABILITY_COLUMN] >= 0.5
    assert predictions[DECISION_COLUMN].equals(expected)
    assert predictions.index.equals(clean_customers.index)


def test_predict_proba_saved_and_reloaded_model_gives_same_output(
    tmp_path: Path, fitted_model: Pipeline, clean_customers: pd.DataFrame
) -> None:
    """Saving and loading the model does not change its predictions."""
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)

    before = predict_proba(fitted_model, clean_customers)
    after = predict_proba(load_model(path), clean_customers)

    assert (before == after).all()
