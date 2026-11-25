"""Register stage: quality bar and model registry aliases.

The MLflow *model registry* keeps numbered versions of a named model
(``churnguard`` v1, v2, ...). *Aliases* are movable labels on those versions:

* ``challenger``: a new version that passed the quality bar and is waiting
  to prove it is better than the current best.
* ``champion``: the official version. The API always loads
  ``models:/churnguard@champion``, so moving this alias is how a new model goes
  live, and moving it back is an instant rollback, with no API redeploy.

Decision flow for the model selected by the train stage:

1. **Quality bar**: test ROC-AUC and recall must reach the minimums in
   ``params.yaml``. If not, nothing is registered and the stage fails.
2. **Register** the model as a new version with the ``challenger`` alias.
3. **Compare**: if there is no champion yet, or the challenger's test ROC-AUC
   is higher, the ``champion`` alias moves to the new version. Otherwise the
   old champion stays and the new version keeps ``challenger`` for review.

Run as a pipeline stage with ``python -m churnguard.models.register``.

Exported API:
    RegistrationDecision: What the stage decided and why.
    check_quality_bar: Compare metrics with the minimum scores.
    get_alias_metrics: Test metrics of the version behind an alias.
    register_model: Apply the full decision flow.
    run_register: Run the stage between explicit file paths.
    main: Command-line entry point using the project settings.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import mlflow
from mlflow.exceptions import MlflowException

from churnguard.config import MlflowSettings, RegisterParams, load_params, load_settings
from churnguard.exceptions import QualityBarError
from churnguard.logging_utils import configure_logging
from churnguard.models.tracking import configure_mlflow

logger = logging.getLogger(__name__)

COMPARISON_METRIC: Final = "roc_auc"


@dataclass(frozen=True)
class RegistrationDecision:
    """Outcome of the register stage, written to ``registration.json``.

    Attributes:
        passed_quality_bar: Whether the minimum scores were met.
        registered_version: New registry version, or ``None`` if not registered.
        promoted_to_champion: Whether the champion alias moved to it.
        champion_version: Version holding the champion alias afterwards.
        reasons: Human-readable explanation of each decision.
        metrics: Test metrics of the new model.
        previous_champion_metrics: Test metrics of the champion before this run.
    """

    passed_quality_bar: bool
    registered_version: str | None
    promoted_to_champion: bool
    champion_version: str | None
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    previous_champion_metrics: dict[str, float] | None = None


def check_quality_bar(metrics: dict[str, float], params: RegisterParams) -> list[str]:
    """Compare test metrics with the minimum scores.

    Args:
        metrics: Test metrics with ``roc_auc`` and ``recall``.
        params: Minimum scores from ``params.yaml``.

    Returns:
        One message per metric below its minimum; empty if the bar is met.
    """
    shortfalls = []
    if metrics["roc_auc"] < params.min_roc_auc:
        shortfalls.append(f"ROC-AUC {metrics['roc_auc']:.4f} < {params.min_roc_auc}")
    if metrics["recall"] < params.min_recall:
        shortfalls.append(f"recall {metrics['recall']:.4f} < {params.min_recall}")
    return shortfalls


def get_alias_metrics(
    client: mlflow.MlflowClient, model_name: str, alias: str
) -> tuple[str, dict[str, float]] | None:
    """Look up the version behind an alias and its test metrics.

    Args:
        client: MLflow client connected to the tracking server.
        model_name: Registered model name.
        alias: Alias to resolve, e.g. ``"champion"``.

    Returns:
        ``(version, metrics)`` with the ``test_`` prefix removed from metric
        names, or ``None`` if the model or alias does not exist yet.
    """
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None
    if version.run_id is None:
        return str(version.version), {}
    run_metrics = client.get_run(version.run_id).data.metrics
    metrics = {
        name.removeprefix("test_"): value
        for name, value in run_metrics.items()
        if name.startswith("test_")
    }
    return str(version.version), metrics


def register_model(
    run_id: str,
    model_uri: str,
    metrics: dict[str, float],
    mlflow_settings: MlflowSettings,
    params: RegisterParams,
) -> RegistrationDecision:
    """Apply the quality bar, register, and move aliases.

    Args:
        run_id: MLflow run of the selected model.
        model_uri: URI of the logged model to register.
        metrics: The model's test metrics.
        mlflow_settings: Registry name and alias names.
        params: Minimum scores from ``params.yaml``.

    Returns:
        The decision, including the reasons.
    """
    client = mlflow.MlflowClient()
    name = mlflow_settings.registered_model_name
    current_champion = get_alias_metrics(client, name, mlflow_settings.champion_alias)
    previous_version, previous_metrics = current_champion or (None, None)

    shortfalls = check_quality_bar(metrics, params)
    if shortfalls:
        return RegistrationDecision(
            passed_quality_bar=False,
            registered_version=None,
            promoted_to_champion=False,
            champion_version=previous_version,
            reasons=[f"Below quality bar: {message}" for message in shortfalls],
            metrics=metrics,
            previous_champion_metrics=previous_metrics,
        )

    version = str(mlflow.register_model(model_uri, name).version)
    client.set_registered_model_alias(name, mlflow_settings.challenger_alias, version)
    client.set_model_version_tag(name, version, "source_run_id", run_id)
    reasons = [
        f"Passed quality bar; registered as version {version} ({mlflow_settings.challenger_alias})"
    ]

    new_score = metrics[COMPARISON_METRIC]
    if previous_metrics is None or new_score > previous_metrics.get(COMPARISON_METRIC, 0.0):
        client.set_registered_model_alias(name, mlflow_settings.champion_alias, version)
        client.delete_registered_model_alias(name, mlflow_settings.challenger_alias)
        reasons.append(
            "No champion yet; promoted to champion"
            if previous_metrics is None
            else f"Beats champion v{previous_version} on {COMPARISON_METRIC} "
            f"({new_score:.4f} > {previous_metrics[COMPARISON_METRIC]:.4f}); promoted"
        )
        return RegistrationDecision(
            True, version, True, version, reasons, metrics, previous_metrics
        )

    reasons.append(
        f"Does not beat champion v{previous_version} on {COMPARISON_METRIC} "
        f"({new_score:.4f} <= {previous_metrics.get(COMPARISON_METRIC, 0.0):.4f}); "
        "kept as challenger"
    )
    return RegistrationDecision(
        True, version, False, previous_version, reasons, metrics, previous_metrics
    )


def run_register(
    train_summary_path: Path,
    metrics_path: Path,
    report_path: Path,
    mlflow_settings: MlflowSettings,
    params: RegisterParams,
) -> RegistrationDecision:
    """Run the register stage and write ``registration.json``.

    Args:
        train_summary_path: ``train_summary.json`` from the train stage.
        metrics_path: ``metrics.json`` from the evaluate stage.
        report_path: Where to write the decision.
        mlflow_settings: Registry name and alias names.
        params: Minimum scores from ``params.yaml``.

    Returns:
        The decision.

    Raises:
        QualityBarError: If the model missed the quality bar. The report is
            written first.
    """
    summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    decision = register_model(
        summary["best_run_id"], summary["best_model_uri"], metrics, mlflow_settings, params
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(decision), indent=2) + "\n", encoding="utf-8")
    for reason in decision.reasons:
        logger.info("%s", reason)
    if not decision.passed_quality_bar:
        raise QualityBarError("; ".join(decision.reasons))
    return decision


def main() -> None:
    """Run the register stage and exit with status 1 below the quality bar."""
    settings = load_settings()
    configure_logging(settings.log_level)
    configure_mlflow(settings.mlflow)
    params = load_params()
    try:
        run_register(
            train_summary_path=settings.resolve(settings.paths.train_summary),
            metrics_path=settings.resolve(settings.paths.metrics_report),
            report_path=settings.resolve(settings.paths.registration_report),
            mlflow_settings=settings.mlflow,
            params=params.register_,
        )
    except QualityBarError as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
