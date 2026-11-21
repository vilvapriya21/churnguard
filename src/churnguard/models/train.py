"""Train stage: fit candidate models and keep the best one.

Every candidate is a full scikit-learn ``Pipeline`` of the shared preprocessor
plus one estimator, so the saved file contains everything needed to go from
feature columns to a probability.

Candidates are compared with stratified cross-validation on the *training*
data only. The held-out test file is not looked at here; it is reserved for
the evaluate stage, so its scores stay an honest estimate.

MLflow tracking and model registration are added in Task 6.

Run as a pipeline stage with ``python -m churnguard.models.train``.

Exported API:
    ESTIMATORS: Supported model names and their scikit-learn classes.
    build_model_pipeline: Preprocessor plus estimator for one model name.
    cross_validate_auc: Mean cross-validated ROC-AUC of a pipeline.
    train_best_model: Compare all candidates and fit the best one.
    run_train: Run the whole stage between explicit file paths.
    main: Command-line entry point using the project settings.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churnguard.config import TrainParams, load_params, load_settings
from churnguard.data.columns import TARGET_COLUMN
from churnguard.features.build_features import FEATURE_COLUMNS, build_preprocessor
from churnguard.io_utils import read_csv
from churnguard.logging_utils import configure_logging

logger = logging.getLogger(__name__)

ESTIMATORS: Final[Mapping[str, type[BaseEstimator]]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of comparing and fitting the candidate models.

    Attributes:
        best_model_name: Name of the selected candidate.
        model: The selected pipeline, fitted on all training rows.
        cv_scores: Mean cross-validated ROC-AUC for every candidate.
    """

    best_model_name: str
    model: Pipeline
    cv_scores: dict[str, float]


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


def cross_validate_auc(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series, folds: int, random_seed: int
) -> float:
    """Estimate a pipeline's ROC-AUC with stratified k-fold cross-validation.

    Args:
        pipeline: Unfitted pipeline to evaluate.
        features: Training feature columns.
        target: Training labels (0/1).
        folds: Number of folds.
        random_seed: Seed for shuffling the folds.

    Returns:
        The mean ROC-AUC over the folds.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    scores = cross_val_score(pipeline, features, target, cv=splitter, scoring="roc_auc")
    return float(scores.mean())


def train_best_model(
    features: pd.DataFrame, target: pd.Series, params: TrainParams, random_seed: int
) -> TrainingResult:
    """Compare every configured candidate and fit the best on all rows.

    Args:
        features: Training feature columns.
        target: Training labels (0/1).
        params: Training parameters from ``params.yaml``.
        random_seed: Seed for the models and the folds.

    Returns:
        The selected, fitted model and every candidate's score.
    """
    cv_scores: dict[str, float] = {}
    for model_name, model_params in params.models.items():
        pipeline = build_model_pipeline(model_name, model_params, random_seed)
        cv_scores[model_name] = cross_validate_auc(
            pipeline, features, target, params.cv_folds, random_seed
        )
        logger.info("%s: cross-validated ROC-AUC %.4f", model_name, cv_scores[model_name])

    best_model_name = max(cv_scores, key=cv_scores.__getitem__)
    logger.info("Selected %s", best_model_name)
    best_model = build_model_pipeline(best_model_name, params.models[best_model_name], random_seed)
    best_model.fit(features, target)
    return TrainingResult(best_model_name, best_model, cv_scores)


def run_train(
    processed_train_path: Path, model_path: Path, params: TrainParams, random_seed: int
) -> TrainingResult:
    """Run the train stage: read features, select and fit, save the model.

    Args:
        processed_train_path: Training features from the featurize stage.
        model_path: Where to save the fitted pipeline.
        params: Training parameters from ``params.yaml``.
        random_seed: Seed for the models and the folds.

    Returns:
        The training result, for callers that want to log it.
    """
    train = read_csv(processed_train_path)
    result = train_best_model(
        train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN], params, random_seed
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_path)
    logger.info("Saved %s to %s", result.best_model_name, model_path)
    return result


def main() -> None:
    """Run the train stage with the project's settings and parameters."""
    settings = load_settings()
    configure_logging(settings.log_level)
    params = load_params()
    run_train(
        processed_train_path=settings.resolve(settings.paths.processed_train),
        model_path=settings.resolve(settings.paths.model_file),
        params=params.train,
        random_seed=params.base.random_seed,
    )


if __name__ == "__main__":
    main()
