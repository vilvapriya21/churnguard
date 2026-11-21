"""Tests for model metrics."""

import pytest

from churnguard.models.evaluate import METRIC_NAMES, compute_metrics


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
