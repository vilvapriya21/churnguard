"""Model quality metrics.

Probabilities are turned into yes/no decisions with a configurable threshold
(``train.decision_threshold`` in ``params.yaml``). Lowering the threshold flags
more customers, which raises recall (fewer churners missed) at the cost of
precision (more unnecessary calls).

Exported API:
    METRIC_NAMES: Names of the metrics returned by :func:`compute_metrics`.
    compute_metrics: Accuracy, precision, recall, F1 and ROC-AUC.
"""

from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES: Final = ("accuracy", "precision", "recall", "f1", "roc_auc")


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
