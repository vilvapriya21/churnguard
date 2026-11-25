"""Evaluate stage: score the trained model on the held-out test set.

Probabilities are turned into yes/no decisions with a configurable threshold
(``train.decision_threshold`` in ``params.yaml``). Lowering the threshold flags
more customers, which raises recall (fewer churners missed) at the cost of
precision (more unnecessary calls).

The stage writes ``reports/metrics.json`` (read by ``dvc metrics show`` and
``dvc metrics diff``) and two charts: a confusion matrix and the most
important features.

Run as a pipeline stage with ``python -m churnguard.models.evaluate``.

Exported API:
    METRIC_NAMES: Names of the metrics returned by :func:`compute_metrics`.
    compute_metrics: Accuracy, precision, recall, F1 and ROC-AUC.
    feature_importance: Importance of each model input after encoding.
    plot_confusion_matrix, plot_feature_importance: Chart builders.
    run_evaluate: Run the stage between explicit file paths.
    main: Command-line entry point using the project settings.
"""

import json
import logging
from pathlib import Path
from typing import Final

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churnguard.config import load_params, load_settings
from churnguard.data.columns import TARGET_COLUMN
from churnguard.features.build_features import FEATURE_COLUMNS
from churnguard.io_utils import read_csv
from churnguard.logging_utils import configure_logging
from churnguard.models.predict import load_model
from churnguard.models.tracking import configure_mlflow

logger = logging.getLogger(__name__)

METRIC_NAMES: Final = ("accuracy", "precision", "recall", "f1", "roc_auc")
CONFUSION_MATRIX_FILE: Final = "confusion_matrix.png"
FEATURE_IMPORTANCE_FILE: Final = "feature_importance.png"
TOP_FEATURES: Final = 15


def compute_metrics(
    y_true: npt.ArrayLike, y_proba: npt.ArrayLike, threshold: float
) -> dict[str, float]:
    """Score churn probabilities against the true labels.

    Args:
        y_true: True labels, 1 for churn and 0 otherwise.
        y_proba: Predicted churn probabilities between 0 and 1.
        threshold: Probability at or above which a customer counts as churn.

    Returns:
        A mapping from each name in :data:`METRIC_NAMES` to its value.
    """
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_proba)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def feature_importance(model: Pipeline) -> pd.Series:
    """Get how much each encoded input contributes to the model.

    Tree models report ``feature_importances_``; linear models report
    coefficients, whose absolute size is used instead.

    Args:
        model: Fitted pipeline with ``preprocess`` and ``model`` steps.

    Returns:
        Importance per encoded feature name, largest first.

    Raises:
        TypeError: If the estimator exposes neither kind of importance.
    """
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).ravel()
    else:
        raise TypeError(f"{type(estimator).__name__} has no feature importance")
    names = model.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_confusion_matrix(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> Figure:
    """Draw a labelled confusion matrix.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        The figure, ready to save.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure = Figure(figsize=(4.5, 4))
    axes = figure.subplots()
    axes.imshow(matrix, cmap="Blues")
    labels = ["Stayed", "Churned"]
    axes.set_xticks([0, 1], labels=labels)
    axes.set_yticks([0, 1], labels=labels)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title("Confusion matrix (test set)")
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            colour = "white" if value > matrix.max() / 2 else "black"
            axes.text(column, row, str(value), ha="center", va="center", color=colour)
    figure.tight_layout()
    return figure


def plot_feature_importance(importance: pd.Series, top: int = TOP_FEATURES) -> Figure:
    """Draw a horizontal bar chart of the most important features.

    Args:
        importance: Output of :func:`feature_importance`.
        top: How many features to show.

    Returns:
        The figure, ready to save.
    """
    shown = importance.head(top).iloc[::-1]
    figure = Figure(figsize=(7, 5))
    axes = figure.subplots()
    axes.barh([str(name) for name in shown.index], shown.to_numpy())
    axes.set_xlabel("Importance")
    axes.set_title(f"Top {len(shown)} features")
    figure.tight_layout()
    return figure


def run_evaluate(
    model_path: Path,
    processed_test_path: Path,
    metrics_path: Path,
    figures_dir: Path,
    threshold: float,
    run_id: str | None = None,
) -> dict[str, float]:
    """Run the evaluate stage: score the test set, save metrics and charts.

    Args:
        model_path: Fitted pipeline from the train stage.
        processed_test_path: Held-out test features from the featurize stage.
        metrics_path: Where to write ``metrics.json``.
        figures_dir: Folder for the chart images.
        threshold: Decision threshold from ``params.yaml``.
        run_id: MLflow run of the selected model. If given, the test metrics
            (prefixed ``test_``) and charts are added to that run, which is
            what the register stage reads.

    Returns:
        The test metrics.
    """
    model = load_model(model_path)
    test = read_csv(processed_test_path)
    labels = test[TARGET_COLUMN]
    probabilities = model.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]

    metrics = compute_metrics(labels, probabilities, threshold)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Test metrics: %s", {name: round(value, 4) for name, value in metrics.items()})

    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    plot_confusion_matrix(labels, predictions).savefig(figures_dir / CONFUSION_MATRIX_FILE)
    plot_feature_importance(feature_importance(model)).savefig(
        figures_dir / FEATURE_IMPORTANCE_FILE
    )
    logger.info("Saved charts to %s", figures_dir)

    if run_id is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
            mlflow.log_artifacts(str(figures_dir), artifact_path="charts/test")
        logger.info("Logged test metrics to MLflow run %s", run_id)
    return metrics


def main() -> None:
    """Run the evaluate stage with the project's settings and parameters."""
    settings = load_settings()
    configure_logging(settings.log_level)
    configure_mlflow(settings.mlflow)
    params = load_params()
    summary = json.loads(settings.resolve(settings.paths.train_summary).read_text("utf-8"))
    run_evaluate(
        model_path=settings.resolve(settings.paths.model_file),
        processed_test_path=settings.resolve(settings.paths.processed_test),
        metrics_path=settings.resolve(settings.paths.metrics_report),
        figures_dir=settings.resolve(settings.paths.figures_dir),
        threshold=params.train.decision_threshold,
        run_id=summary["best_run_id"],
    )


if __name__ == "__main__":
    main()
