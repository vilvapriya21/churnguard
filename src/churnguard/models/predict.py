"""Turn customer details into churn probabilities.

Used by the API (Task 7) and by tests. Raw customer rows go through the same
:func:`~churnguard.features.build_features.build_features` used in training,
then through the saved pipeline, which holds the fitted preprocessing steps.

Until the MLflow registry is added (Task 6) the model is loaded from the file
written by the train stage.

Exported API:
    PROBABILITY_COLUMN, DECISION_COLUMN: Names of the prediction outputs.
    load_model: Load a saved pipeline.
    predict_proba: Churn probability for each customer row.
    predict_churn: Probability plus yes/no decision for each row.
"""

import logging
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from churnguard.exceptions import ModelNotFoundError
from churnguard.features.build_features import build_features

logger = logging.getLogger(__name__)

PROBABILITY_COLUMN: Final = "churn_probability"
DECISION_COLUMN: Final = "will_churn"


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
