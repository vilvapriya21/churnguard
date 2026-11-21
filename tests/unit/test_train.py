"""Tests for model training and selection."""

from pathlib import Path

import pandas as pd
import pytest

from churnguard.config import Params
from churnguard.data.columns import TARGET_COLUMN
from churnguard.features.build_features import FEATURE_COLUMNS, split_and_featurize
from churnguard.models.predict import load_model
from churnguard.models.train import (
    ESTIMATORS,
    build_model_pipeline,
    run_train,
    train_best_model,
)


@pytest.fixture
def processed_train(clean_customers: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Featurised training rows from the fake customers."""
    train, _ = split_and_featurize(clean_customers, params.split, random_seed=0)
    return train


def test_build_model_pipeline_unknown_model_raises_value_error() -> None:
    """Only the supported model names are accepted."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model_pipeline("neural_net", {}, random_seed=0)


@pytest.mark.parametrize("model_name", list(ESTIMATORS))
def test_build_model_pipeline_has_preprocess_and_model_steps(
    model_name: str, params: Params
) -> None:
    """Each pipeline bundles the shared preprocessor with its estimator."""
    pipeline = build_model_pipeline(model_name, params.train.models[model_name], random_seed=0)

    assert list(pipeline.named_steps) == ["preprocess", "model"]


def test_train_best_model_scores_every_candidate_and_picks_the_best(
    processed_train: pd.DataFrame, params: Params
) -> None:
    """All configured models are compared and the top scorer is fitted."""
    result = train_best_model(
        processed_train[list(FEATURE_COLUMNS)],
        processed_train[TARGET_COLUMN],
        params.train,
        random_seed=0,
    )

    assert set(result.cv_scores) == set(params.train.models)
    assert result.best_model_name == max(result.cv_scores, key=result.cv_scores.__getitem__)
    probabilities = result.model.predict_proba(processed_train[list(FEATURE_COLUMNS)])[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_train_best_model_learns_signal_above_chance(
    processed_train: pd.DataFrame, params: Params
) -> None:
    """On data with a real pattern the selected model beats random guessing."""
    result = train_best_model(
        processed_train[list(FEATURE_COLUMNS)],
        processed_train[TARGET_COLUMN],
        params.train,
        random_seed=0,
    )

    assert result.cv_scores[result.best_model_name] > 0.7


def test_run_train_saves_a_loadable_model(
    tmp_path: Path, processed_train: pd.DataFrame, params: Params
) -> None:
    """The stage writes a model file that can be loaded back."""
    train_path = tmp_path / "train.csv"
    processed_train.to_csv(train_path, index=False)
    model_path = tmp_path / "models" / "model.joblib"

    run_train(train_path, model_path, params.train, random_seed=0)

    assert list(load_model(model_path).named_steps) == ["preprocess", "model"]
