"""Tests for model metrics and the evaluate stage."""

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churnguard.config import MlflowSettings, Params
from churnguard.data.columns import TARGET_COLUMN
from churnguard.features.build_features import FEATURE_COLUMNS
from churnguard.models.evaluate import (
    CONFUSION_MATRIX_FILE,
    FEATURE_IMPORTANCE_FILE,
    METRIC_NAMES,
    compute_metrics,
    feature_importance,
    run_evaluate,
)
from churnguard.models.train import build_model_pipeline


def test_compute_metrics_perfect_predictions_score_one() -> None:
    """Probabilities that separate the classes perfectly give every metric 1."""
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)

    assert set(metrics) == set(METRIC_NAMES)
    assert all(value == 1.0 for value in metrics.values())


def test_compute_metrics_lower_threshold_raises_recall() -> None:
    """Lowering the threshold catches more churners."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.3, 0.9]

    strict = compute_metrics(y_true, y_proba, threshold=0.5)
    lenient = compute_metrics(y_true, y_proba, threshold=0.25)

    assert strict["recall"] == pytest.approx(0.5)
    assert lenient["recall"] == pytest.approx(1.0)


def test_compute_metrics_no_positive_predictions_gives_zero_precision() -> None:
    """When nobody is flagged, precision is 0 instead of an error."""
    metrics = compute_metrics([0, 1], [0.1, 0.2], threshold=0.9)

    assert metrics["precision"] == 0.0


def test_feature_importance_works_for_linear_and_tree_models(
    processed_split: tuple[pd.DataFrame, pd.DataFrame], params: Params
) -> None:
    """Coefficients and tree importances are both turned into a ranking."""
    train, _ = processed_split
    for model_name in ("logistic_regression", "random_forest"):
        model = build_model_pipeline(model_name, params.train.models[model_name], 0)
        model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])

        importance = feature_importance(model)

        assert importance.is_monotonic_decreasing
        assert (importance >= 0).all()


def test_feature_importance_unsupported_estimator_raises_type_error() -> None:
    """An estimator without importances or coefficients is rejected."""
    model = Pipeline([("preprocess", FunctionTransformer()), ("model", FunctionTransformer())])

    with pytest.raises(TypeError, match="no feature importance"):
        feature_importance(model)


def test_run_evaluate_writes_metrics_charts_and_logs_to_mlflow(
    tmp_path: Path,
    processed_split: tuple[pd.DataFrame, pd.DataFrame],
    params: Params,
    mlflow_settings: MlflowSettings,
) -> None:
    """The stage writes metrics.json, two charts, and test metrics in MLflow."""
    train, test = processed_split
    model = build_model_pipeline(
        "logistic_regression", params.train.models["logistic_regression"], 0
    )
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    test_path = tmp_path / "test.csv"
    test.to_csv(test_path, index=False)
    with mlflow.start_run() as run:
        pass

    metrics = run_evaluate(
        model_path,
        test_path,
        tmp_path / "metrics.json",
        tmp_path / "figures",
        threshold=0.5,
        run_id=run.info.run_id,
    )

    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert (tmp_path / "figures" / CONFUSION_MATRIX_FILE).is_file()
    assert (tmp_path / "figures" / FEATURE_IMPORTANCE_FILE).is_file()
    logged = mlflow.get_run(run.info.run_id).data.metrics
    assert logged["test_roc_auc"] == pytest.approx(metrics["roc_auc"])
