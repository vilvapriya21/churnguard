"""Train stage: fit candidate models, track every run in MLflow, keep the best.

Every candidate is a full scikit-learn ``Pipeline`` of the shared preprocessor
plus one estimator, so the saved model contains everything needed to go from
feature columns to a probability.

Candidates are compared with stratified cross-validation on the *training*
data only. The held-out test file is not looked at here; it is reserved for
the evaluate stage, so its scores stay an honest estimate.

MLflow layout: one parent run named ``train`` with one nested run per
candidate. Each candidate run records:

* **params**: model settings from ``params.yaml``, seed, folds, threshold;
* **metrics**: cross-validated accuracy, precision, recall, F1 and ROC-AUC;
* **artifacts**: confusion matrix and feature importance charts;
* **tags**: Git commit and DVC data version (see ``tracking.py``);
* **dataset**: the training data, via ``mlflow.log_input``;
* **model**: the fitted pipeline with a signature and an input example.

Run as a pipeline stage with ``python -m churnguard.models.train``.

Exported API:
    ESTIMATORS: Supported model names and their scikit-learn classes.
    CandidateResult, TrainingResult: What training produced.
    build_model_pipeline: Preprocessor plus estimator for one model name.
    cross_validate_candidate: Out-of-fold metrics for a pipeline.
    train_and_track: Train every candidate inside MLflow runs.
    run_train: Run the whole stage between explicit file paths.
    main: Command-line entry point using the project settings.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset, from_pandas
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from churnguard.config import PROJECT_ROOT, TrainParams, load_params, load_settings
from churnguard.data.columns import TARGET_COLUMN
from churnguard.features.build_features import FEATURE_COLUMNS, build_preprocessor
from churnguard.io_utils import read_csv
from churnguard.logging_utils import configure_logging
from churnguard.models.evaluate import (
    compute_metrics,
    feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
)
from churnguard.models.tracking import configure_mlflow, lineage_tags

logger = logging.getLogger(__name__)

ESTIMATORS: Final[Mapping[str, type[BaseEstimator]]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}
MODEL_ARTIFACT_NAME: Final = "model"
# MLflow saves scikit-learn models with skops, which (unlike pickle) refuses to
# load any type not known to be safe. Only these reviewed types are trusted:
# - numpy.dtype: stored by the one-hot encoder; a plain data type, no code.
# - sklearn.tree._tree.Tree: node storage of random forest and gradient
#   boosting. A hand-crafted malicious file could crash the process, so it is
#   trusted only because every model here is produced by this pipeline and
#   loaded from this project's own registry, never from an outside source.
SKOPS_TRUSTED_TYPES: Final = ("numpy.dtype", "sklearn.tree._tree.Tree")
SELECTION_METRIC: Final = "roc_auc"
INPUT_EXAMPLE_ROWS: Final = 5


@dataclass(frozen=True)
class CandidateResult:
    """One trained candidate and where it is recorded in MLflow.

    Attributes:
        model_name: Key of :data:`ESTIMATORS`.
        run_id: MLflow run that holds its params, metrics and model.
        model_uri: URI of the logged model, used for registration.
        cv_metrics: Cross-validated metrics on the training data.
        model: The pipeline fitted on all training rows.
    """

    model_name: str
    run_id: str
    model_uri: str
    cv_metrics: dict[str, float]
    model: Pipeline


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of training every candidate.

    Attributes:
        parent_run_id: MLflow run that groups the candidates.
        best: The candidate with the highest cross-validated ROC-AUC.
        candidates: Every candidate, in the order they were trained.
    """

    parent_run_id: str
    best: CandidateResult
    candidates: list[CandidateResult]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for ``train_summary.json``.

        Returns:
            Best model name, run IDs, model URI and each candidate's CV scores.
        """
        return {
            "best_model_name": self.best.model_name,
            "best_run_id": self.best.run_id,
            "best_model_uri": self.best.model_uri,
            "parent_run_id": self.parent_run_id,
            "cv_metrics": {c.model_name: c.cv_metrics for c in self.candidates},
        }


def build_model_pipeline(
    model_name: str, model_params: Mapping[str, Any], random_seed: int
) -> Pipeline:
    """Create an unfitted preprocessor-plus-estimator pipeline.

    Args:
        model_name: A key of :data:`ESTIMATORS`.
        model_params: Keyword arguments for the estimator, from ``params.yaml``.
        random_seed: Seed passed to the estimator for reproducible results.

    Returns:
        The unfitted pipeline.

    Raises:
        ValueError: If ``model_name`` is not supported.
    """
    if model_name not in ESTIMATORS:
        supported = ", ".join(ESTIMATORS)
        raise ValueError(f"Unknown model {model_name!r}; supported: {supported}")
    estimator = ESTIMATORS[model_name](random_state=random_seed, **model_params)
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def cross_validate_candidate(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    folds: int,
    random_seed: int,
    threshold: float,
) -> tuple[dict[str, float], npt.NDArray[np.float64]]:
    """Score a pipeline with stratified k-fold cross-validation.

    Every row is predicted by a model that did not see it during fitting
    ("out-of-fold"), and all five metrics are computed on those predictions.

    Args:
        pipeline: Unfitted pipeline to evaluate.
        features: Training feature columns.
        target: Training labels (0/1).
        folds: Number of folds.
        random_seed: Seed for shuffling the folds.
        threshold: Decision threshold for the yes/no metrics.

    Returns:
        The metrics and the out-of-fold churn probabilities.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    out_of_fold: npt.NDArray[np.float64] = cross_val_predict(
        pipeline, features, target, cv=splitter, method="predict_proba"
    )[:, 1]
    return compute_metrics(target, out_of_fold, threshold), out_of_fold


def _train_candidate(
    model_name: str,
    model_params: Mapping[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    params: TrainParams,
    random_seed: int,
    dataset: PandasDataset,
    tags: Mapping[str, str],
) -> CandidateResult:
    """Cross-validate, fit and log one candidate in a nested MLflow run."""
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        mlflow.set_tags({**tags, "model_name": model_name, "run_type": "candidate"})
        mlflow.log_params(
            {
                "model_name": model_name,
                "random_seed": random_seed,
                "cv_folds": params.cv_folds,
                "decision_threshold": params.decision_threshold,
                **{f"model__{key}": value for key, value in model_params.items()},
            }
        )
        mlflow.log_input(dataset, context="training")

        pipeline = build_model_pipeline(model_name, model_params, random_seed)
        cv_metrics, out_of_fold = cross_validate_candidate(
            pipeline, features, target, params.cv_folds, random_seed, params.decision_threshold
        )
        mlflow.log_metrics({f"cv_{name}": value for name, value in cv_metrics.items()})

        pipeline.fit(features, target)
        cv_predictions = (out_of_fold >= params.decision_threshold).astype(int)
        mlflow.log_figure(
            plot_confusion_matrix(target, cv_predictions), "charts/cv_confusion_matrix.png"
        )
        mlflow.log_figure(
            plot_feature_importance(feature_importance(pipeline)), "charts/feature_importance.png"
        )

        input_example = features.head(INPUT_EXAMPLE_ROWS)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=infer_signature(input_example, pipeline.predict_proba(input_example)),
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
            metadata={"decision_threshold": params.decision_threshold},
            skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
        )
        logger.info(
            "%s: CV ROC-AUC %.4f, recall %.4f",
            model_name,
            cv_metrics["roc_auc"],
            cv_metrics["recall"],
        )
        return CandidateResult(
            model_name=model_name,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            cv_metrics=cv_metrics,
            model=pipeline,
        )


def train_and_track(
    features: pd.DataFrame,
    target: pd.Series,
    params: TrainParams,
    random_seed: int,
    dataset: PandasDataset,
    tags: Mapping[str, str],
) -> TrainingResult:
    """Train every configured candidate inside one parent MLflow run.

    MLflow must already point at a tracking store and experiment.

    Args:
        features: Training feature columns.
        target: Training labels (0/1).
        params: Training parameters from ``params.yaml``.
        random_seed: Seed for the models and the folds.
        dataset: The training data, recorded with ``mlflow.log_input``.
        tags: Lineage tags added to every run.

    Returns:
        All candidates and the best one by cross-validated ROC-AUC.
    """
    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags({**tags, "run_type": "selection"})
        candidates = [
            _train_candidate(
                name, model_params, features, target, params, random_seed, dataset, tags
            )
            for name, model_params in params.models.items()
        ]
        best = max(candidates, key=lambda candidate: candidate.cv_metrics[SELECTION_METRIC])
        mlflow.set_tag("best_model_name", best.model_name)
        mlflow.log_metric(f"best_cv_{SELECTION_METRIC}", best.cv_metrics[SELECTION_METRIC])
    mlflow.MlflowClient().set_tag(best.run_id, "selected", "true")
    logger.info("Selected %s (run %s)", best.model_name, best.run_id)
    return TrainingResult(parent_run_id=parent.info.run_id, best=best, candidates=candidates)


def run_train(
    processed_train_path: Path,
    model_path: Path,
    summary_path: Path,
    params: TrainParams,
    random_seed: int,
    tags: Mapping[str, str],
) -> TrainingResult:
    """Run the train stage: read features, train and track, save outputs.

    Args:
        processed_train_path: Training features from the featurize stage.
        model_path: Where to save the best fitted pipeline.
        summary_path: Where to write ``train_summary.json``.
        params: Training parameters from ``params.yaml``.
        random_seed: Seed for the models and the folds.
        tags: Lineage tags added to every MLflow run.

    Returns:
        The training result.
    """
    train = read_csv(processed_train_path)
    features = train[list(FEATURE_COLUMNS)]
    dataset = from_pandas(
        train, source=str(processed_train_path), targets=TARGET_COLUMN, name="processed_train"
    )
    result = train_and_track(features, train[TARGET_COLUMN], params, random_seed, dataset, tags)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.best.model, model_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s to %s", result.best.model_name, model_path)
    return result


def main() -> None:
    """Run the train stage with the project's settings and parameters."""
    settings = load_settings()
    configure_logging(settings.log_level)
    configure_mlflow(settings.mlflow)
    params = load_params()
    train_path = settings.resolve(settings.paths.processed_train)
    run_train(
        processed_train_path=train_path,
        model_path=settings.resolve(settings.paths.model_file),
        summary_path=settings.resolve(settings.paths.train_summary),
        params=params.train,
        random_seed=params.base.random_seed,
        tags=lineage_tags(PROJECT_ROOT, settings.resolve(settings.paths.raw_data), train_path),
    )


if __name__ == "__main__":
    main()
