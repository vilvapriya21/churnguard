"""Tests for model training and MLflow tracking."""

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from churnguard.config import MlflowSettings, Params, TrainParams
from churnguard.data.columns import TARGET_COLUMN
from churnguard.features.build_features import FEATURE_COLUMNS
from churnguard.models.predict import load_model
from churnguard.models.train import (
    ESTIMATORS,
    TrainingResult,
    build_model_pipeline,
    cross_validate_candidate,
    run_train,
)

LINEAGE = {"git_commit": "abc123", "dvc_data_version": "d41d8cd9"}


@pytest.fixture
def trained(
    tmp_path: Path,
    processed_split: tuple[pd.DataFrame, pd.DataFrame],
    fast_train_params: TrainParams,
    mlflow_settings: MlflowSettings,
) -> tuple[TrainingResult, Path, Path]:
    """Run the train stage once against a throw-away MLflow store."""
    train_path = tmp_path / "train.csv"
    processed_split[0].to_csv(train_path, index=False)
    model_path = tmp_path / "models" / "model.joblib"
    summary_path = tmp_path / "reports" / "train_summary.json"
    result = run_train(train_path, model_path, summary_path, fast_train_params, 0, LINEAGE)
    return result, model_path, summary_path


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


def test_cross_validate_candidate_returns_all_metrics_and_one_score_per_row(
    processed_split: tuple[pd.DataFrame, pd.DataFrame], params: Params
) -> None:
    """Out-of-fold scoring predicts every training row once."""
    train, _ = processed_split
    pipeline = build_model_pipeline(
        "logistic_regression", params.train.models["logistic_regression"], 0
    )

    metrics, out_of_fold = cross_validate_candidate(
        pipeline, train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN], 3, 0, 0.5
    )

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert len(out_of_fold) == len(train)
    assert metrics["roc_auc"] > 0.7  # the fake data has a real signal


def test_run_train_logs_one_run_per_candidate_plus_parent(
    trained: tuple[TrainingResult, Path, Path], fast_train_params: TrainParams
) -> None:
    """Every candidate gets its own run, grouped under one parent run."""
    result, _, _ = trained

    runs = mlflow.search_runs(output_format="list")

    assert len(runs) == len(fast_train_params.models) + 1
    assert {c.model_name for c in result.candidates} == set(fast_train_params.models)


def test_run_train_candidate_run_has_params_metrics_tags_and_dataset(
    trained: tuple[TrainingResult, Path, Path],
) -> None:
    """A candidate run records params, CV metrics, lineage tags and its input data."""
    result, _, _ = trained

    run = mlflow.get_run(result.best.run_id)

    assert run.data.params["model_name"] == result.best.model_name
    assert "cv_roc_auc" in run.data.metrics
    assert run.data.tags["git_commit"] == "abc123"
    assert run.data.tags["dvc_data_version"] == "d41d8cd9"
    assert run.data.tags["selected"] == "true"
    assert run.inputs is not None
    assert run.inputs.dataset_inputs[0].dataset.name == "processed_train"


def test_run_train_logged_model_has_signature_and_threshold(
    trained: tuple[TrainingResult, Path, Path], fast_train_params: TrainParams
) -> None:
    """The logged model carries its input signature and decision threshold."""
    result, _, _ = trained

    model = mlflow.pyfunc.load_model(result.best.model_uri)

    input_names = model.metadata.get_input_schema().input_names()
    assert input_names == list(FEATURE_COLUMNS)
    threshold = model.metadata.metadata["decision_threshold"]
    assert threshold == fast_train_params.decision_threshold


def test_run_train_picks_highest_cv_roc_auc(trained: tuple[TrainingResult, Path, Path]) -> None:
    """The selected model is the candidate with the best cross-validated ROC-AUC."""
    result, _, _ = trained

    best_score = max(c.cv_metrics["roc_auc"] for c in result.candidates)

    assert result.best.cv_metrics["roc_auc"] == best_score


def test_run_train_writes_model_file_and_summary(
    trained: tuple[TrainingResult, Path, Path],
) -> None:
    """The stage saves the best model and a JSON summary with its run ID."""
    result, model_path, summary_path = trained

    summary = json.loads(summary_path.read_text())

    assert list(load_model(model_path).named_steps) == ["preprocess", "model"]
    assert summary["best_run_id"] == result.best.run_id
    assert summary["best_model_name"] == result.best.model_name
