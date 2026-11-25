"""Turn customer details into churn probabilities.

Raw customer rows always go through the same
:func:`~churnguard.features.build_features.build_features` used in training,
then through the trained pipeline, which holds the fitted preprocessing.

Two ways to get a model:

* :func:`load_model` reads the file written by the train stage (used by the
  evaluate stage and tests).
* :func:`load_champion` loads ``models:/<name>@champion`` from the MLflow
  registry (used by the API). That model carries its *signature*, so MLflow
  rejects input with missing columns or wrong types before predicting, and
  its decision threshold travels with it as model metadata.

Exported API:
    PROBABILITY_COLUMN, DECISION_COLUMN: Names of the prediction outputs.
    ServingModel: A registry model plus its version and threshold.
    load_model: Load a fitted pipeline from a file.
    load_champion: Load the champion model from the MLflow registry.
    predict_proba: Churn probability per row with a fitted pipeline.
    predict_proba_serving: Churn probability per row with a registry model.
    predict_churn: Probability plus yes/no decision per row.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from sklearn.pipeline import Pipeline

from churnguard.config import MlflowSettings
from churnguard.exceptions import ModelNotFoundError
from churnguard.features.build_features import build_features

logger = logging.getLogger(__name__)

PROBABILITY_COLUMN: Final = "churn_probability"
DECISION_COLUMN: Final = "will_churn"
THRESHOLD_METADATA_KEY: Final = "decision_threshold"


@dataclass(frozen=True)
class ServingModel:
    """A model loaded from the registry, ready to serve.

    Attributes:
        model: MLflow model that returns class probabilities.
        name: Registered model name.
        version: Registry version behind the alias.
        threshold: Decision threshold saved with the model.
    """

    model: PyFuncModel
    name: str
    version: str
    threshold: float


def load_model(path: Path) -> Pipeline:
    """Load a fitted pipeline saved by the train stage.

    Only load model files produced by this project's own pipeline: the format
    can run code when loaded, so files from unknown sources are not safe.

    Args:
        path: Location of the saved model.

    Returns:
        The fitted pipeline.

    Raises:
        ModelNotFoundError: If the file is missing or is not a pipeline.
    """
    if not path.is_file():
        raise ModelNotFoundError(f"No model at {path}. Run the train stage first.")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise ModelNotFoundError(f"{path} does not contain a scikit-learn Pipeline")
    logger.info("Loaded model from %s", path)
    return model


def load_champion(settings: MlflowSettings) -> ServingModel:
    """Load the model behind the champion alias from the MLflow registry.

    Args:
        settings: Tracking address, registered model name and alias.

    Returns:
        The model with its version and decision threshold.

    Raises:
        ModelNotFoundError: If the registry has no such model or alias, or
            the model has no decision threshold in its metadata.
    """
    name, alias = settings.registered_model_name, settings.champion_alias
    uri = f"models:/{name}@{alias}"
    client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    mlflow.set_tracking_uri(settings.tracking_uri)
    try:
        version = client.get_model_version_by_alias(name, alias)
        model = mlflow.pyfunc.load_model(uri)
    except MlflowException as error:
        raise ModelNotFoundError(f"Cannot load {uri}: {error.message}") from error

    metadata = model.metadata.metadata or {}
    if THRESHOLD_METADATA_KEY not in metadata:
        raise ModelNotFoundError(f"{uri} has no {THRESHOLD_METADATA_KEY!r} in its metadata")
    logger.info("Loaded %s (version %s)", uri, version.version)
    return ServingModel(
        model=model,
        name=name,
        version=str(version.version),
        threshold=float(metadata[THRESHOLD_METADATA_KEY]),
    )


def predict_proba(model: Pipeline, customers: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Compute the churn probability for each customer row.

    Args:
        model: Fitted pipeline from :func:`load_model`.
        customers: Raw customer rows; missing input columns are allowed.

    Returns:
        One probability between 0 and 1 per row.
    """
    probabilities: npt.NDArray[np.float64] = model.predict_proba(build_features(customers))[:, 1]
    return probabilities


def predict_proba_serving(
    serving: ServingModel, customers: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """Compute churn probabilities with a registry model.

    The input is checked against the model signature before predicting.

    Args:
        serving: Model from :func:`load_champion`.
        customers: Raw customer rows; missing input columns are allowed.

    Returns:
        One probability between 0 and 1 per row.
    """
    class_probabilities = np.asarray(serving.model.predict(build_features(customers)))
    probabilities: npt.NDArray[np.float64] = class_probabilities[:, 1].astype(float)
    return probabilities


def predict_churn(model: Pipeline, customers: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compute probabilities and yes/no churn decisions.

    Args:
        model: Fitted pipeline from :func:`load_model`.
        customers: Raw customer rows; missing input columns are allowed.
        threshold: Probability at or above which a customer is flagged.

    Returns:
        A data frame with the input's index and the columns
        :data:`PROBABILITY_COLUMN` and :data:`DECISION_COLUMN`.
    """
    probabilities = predict_proba(model, customers)
    return pd.DataFrame(
        {
            PROBABILITY_COLUMN: probabilities,
            DECISION_COLUMN: probabilities >= threshold,
        },
        index=customers.index,
    )
