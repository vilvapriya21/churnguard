"""Tests for the quality bar and registry aliases."""

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.data.pandas_dataset import from_pandas

from churnguard.config import MlflowSettings, Params, RegisterParams, TrainParams
from churnguard.data.columns import TARGET_COLUMN
from churnguard.exceptions import QualityBarError
from churnguard.features.build_features import FEATURE_COLUMNS
from churnguard.models.register import (
    check_quality_bar,
    get_alias_metrics,
    register_model,
    run_register,
)
from churnguard.models.train import CandidateResult, train_and_track

GOOD = {"roc_auc": 0.85, "recall": 0.75, "precision": 0.5, "f1": 0.6, "accuracy": 0.8}
BETTER = {**GOOD, "roc_auc": 0.90}
WORSE = {**GOOD, "roc_auc": 0.82}


@pytest.fixture
def candidate(
    processed_split: tuple[pd.DataFrame, pd.DataFrame],
    fast_train_params: TrainParams,
    mlflow_settings: MlflowSettings,
) -> CandidateResult:
    """One logged candidate model in the throw-away MLflow store."""
    train, _ = processed_split
    one_model = fast_train_params.model_copy(
        update={"models": {"logistic_regression": fast_train_params.models["logistic_regression"]}}
    )
    dataset = from_pandas(train, targets=TARGET_COLUMN, name="processed_train")
    result = train_and_track(
        train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN], one_model, 0, dataset, {}
    )
    return result.best


def _log_test_metrics(run_id: str, metrics: dict[str, float]) -> None:
    """Do what the evaluate stage does: attach test metrics to the run."""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})


def test_check_quality_bar_good_metrics_have_no_shortfalls(params: Params) -> None:
    """Metrics above both minimums pass."""
    assert check_quality_bar(GOOD, params.register_) == []


def test_check_quality_bar_low_recall_is_reported(params: Params) -> None:
    """Recall below 0.70 is named in the shortfalls."""
    shortfalls = check_quality_bar({**GOOD, "recall": 0.5}, params.register_)

    assert len(shortfalls) == 1
    assert "recall" in shortfalls[0]


def test_get_alias_metrics_unknown_model_returns_none(mlflow_settings: MlflowSettings) -> None:
    """Before anything is registered there is no champion."""
    client = mlflow.MlflowClient()

    assert get_alias_metrics(client, "does-not-exist", "champion") is None


def test_register_model_first_good_model_becomes_champion(
    candidate: CandidateResult, mlflow_settings: MlflowSettings, params: Params
) -> None:
    """With no champion yet, a model that passes the bar is promoted."""
    _log_test_metrics(candidate.run_id, GOOD)

    decision = register_model(
        candidate.run_id, candidate.model_uri, GOOD, mlflow_settings, params.register_
    )

    assert decision.promoted_to_champion
    version = mlflow.MlflowClient().get_model_version_by_alias(
        mlflow_settings.registered_model_name, "champion"
    )
    assert str(version.version) == decision.registered_version


def test_register_model_better_model_replaces_champion(
    candidate: CandidateResult, mlflow_settings: MlflowSettings, params: Params
) -> None:
    """A higher test ROC-AUC moves the champion alias to the new version."""
    _log_test_metrics(candidate.run_id, GOOD)
    register_model(candidate.run_id, candidate.model_uri, GOOD, mlflow_settings, params.register_)

    decision = register_model(
        candidate.run_id, candidate.model_uri, BETTER, mlflow_settings, params.register_
    )

    assert decision.promoted_to_champion
    assert decision.champion_version == "2"


def test_register_model_worse_model_stays_challenger(
    candidate: CandidateResult, mlflow_settings: MlflowSettings, params: Params
) -> None:
    """A model that passes the bar but does not beat the champion stays challenger."""
    _log_test_metrics(candidate.run_id, GOOD)
    register_model(candidate.run_id, candidate.model_uri, GOOD, mlflow_settings, params.register_)

    decision = register_model(
        candidate.run_id, candidate.model_uri, WORSE, mlflow_settings, params.register_
    )

    client = mlflow.MlflowClient()
    name = mlflow_settings.registered_model_name
    assert not decision.promoted_to_champion
    assert decision.champion_version == "1"
    assert str(client.get_model_version_by_alias(name, "challenger").version) == "2"


def test_register_model_below_bar_is_not_registered(
    candidate: CandidateResult, mlflow_settings: MlflowSettings, params: Params
) -> None:
    """A model below the quality bar never enters the registry."""
    decision = register_model(
        candidate.run_id,
        candidate.model_uri,
        {**GOOD, "roc_auc": 0.6},
        mlflow_settings,
        params.register_,
    )

    assert not decision.passed_quality_bar
    assert decision.registered_version is None
    assert mlflow.search_registered_models() == []


def test_run_register_below_bar_writes_report_then_raises(
    tmp_path: Path, candidate: CandidateResult, mlflow_settings: MlflowSettings
) -> None:
    """The stage fails below the bar, after writing the report explaining why."""
    summary_path = tmp_path / "train_summary.json"
    summary_path.write_text(
        json.dumps({"best_run_id": candidate.run_id, "best_model_uri": candidate.model_uri})
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({**GOOD, "recall": 0.4}))
    report_path = tmp_path / "registration.json"
    strict = RegisterParams(min_roc_auc=0.8, min_recall=0.7)

    with pytest.raises(QualityBarError, match="recall"):
        run_register(summary_path, metrics_path, report_path, mlflow_settings, strict)

    assert json.loads(report_path.read_text())["passed_quality_bar"] is False
