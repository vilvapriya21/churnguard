"""Typed access to environment settings and experiment parameters.

Two kinds of configuration are kept apart on purpose:

* **Settings** (``configs/<env>.yaml``): things that change between
  environments, such as file paths, the log level and the MLflow address.
  The environment is chosen with the ``APP_ENV`` variable (default ``dev``).
  Any value can be overridden by an environment variable prefixed with
  ``CHURNGUARD_`` (nested keys use ``__``), which is how Docker and CI pass
  values in without editing files. Secrets are only ever read from the
  environment or a git-ignored ``.env`` file.
* **Parameters** (``params.yaml``): the experiment knobs that change model
  results, such as the random seed and model hyperparameters. DVC tracks this
  file so a change re-runs only the affected pipeline stages.

Both are validated with pydantic, so a typo or a wrong type fails immediately
with a clear message instead of deep inside the pipeline.

Exported API:
    Environment, Settings, Params: Typed configuration models.
    load_settings: Build ``Settings`` for an environment.
    load_params: Read and validate ``params.yaml``.
"""

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from churnguard.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
APP_ENV_VARIABLE = "APP_ENV"
DEFAULT_ENV = "dev"


class Environment(StrEnum):
    """Deployment environments the project can run in."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


# --------------------------------------------------------------------------- #
# Settings (configs/<env>.yaml)
# --------------------------------------------------------------------------- #
class PathsSettings(BaseModel):
    """File-system locations, relative to the project root."""

    model_config = ConfigDict(frozen=True)

    raw_data: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


class MlflowSettings(BaseModel):
    """Where MLflow stores runs and how the registered model is named."""

    model_config = ConfigDict(frozen=True)

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    champion_alias: str
    challenger_alias: str


class Settings(BaseSettings):
    """All environment settings for one run of the application.

    Values are read from, highest priority first: environment variables,
    the ``.env`` file, then ``configs/<env>.yaml``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CHURNGUARD_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    env: Environment
    log_level: str = Field(pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    paths: PathsSettings
    mlflow: MlflowSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let environment variables override the values from the YAML file.

        The YAML content is passed in as init arguments by ``load_settings``.
        Putting ``init_settings`` last gives it the lowest priority, so a value
        set in the environment always wins over the file.

        Args:
            settings_cls: The settings class being built.
            init_settings: Values passed to the constructor (the YAML file).
            env_settings: Values from environment variables.
            dotenv_settings: Values from the ``.env`` file.
            file_secret_settings: Values from a secrets directory.

        Returns:
            The sources in priority order, highest first.
        """
        return env_settings, dotenv_settings, file_secret_settings, init_settings

    def resolve(self, path: Path) -> Path:
        """Turn a project-relative path into an absolute one.

        Args:
            path: A path from ``Settings.paths``.

        Returns:
            The path unchanged if it is absolute, otherwise anchored at the
            project root so commands work from any working directory.
        """
        return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------- #
# Parameters (params.yaml)
# --------------------------------------------------------------------------- #
class _StrictModel(BaseModel):
    """Base model that rejects unknown keys, catching typos in params.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseParams(_StrictModel):
    """Parameters shared by every stage."""

    random_seed: int
    target_column: str
    id_column: str


class IngestParams(_StrictModel):
    """Parameters for the ingest stage."""

    scoring_size: float = Field(gt=0, lt=1)


class SplitParams(_StrictModel):
    """Parameters for the train/test split."""

    test_size: float = Field(gt=0, lt=1)


class ValidateParams(_StrictModel):
    """Thresholds for the data quality gates."""

    max_missing_fraction: float = Field(ge=0, le=1)
    psi_bins: int = Field(ge=2)
    psi_warn: float = Field(gt=0)
    psi_fail: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_psi_order(self) -> "ValidateParams":
        """Ensure the warning limit is below the failure limit."""
        if self.psi_warn >= self.psi_fail:
            raise ValueError("psi_warn must be lower than psi_fail")
        return self


class TrainParams(_StrictModel):
    """Parameters for model training.

    ``models`` maps a model name to the keyword arguments for its estimator.
    The values are passed straight to scikit-learn, so their types vary.
    """

    decision_threshold: float = Field(gt=0, lt=1)
    models: dict[str, dict[str, Any]] = Field(min_length=1)


class RegisterParams(_StrictModel):
    """Quality bar a model must pass to be promoted to champion."""

    min_roc_auc: float = Field(ge=0, le=1)
    min_recall: float = Field(ge=0, le=1)


class Params(_StrictModel):
    """All experiment parameters from ``params.yaml``.

    ``validate`` and ``register`` are names of built-in pydantic methods, so
    those sections are stored as ``validate_`` and ``register_`` and mapped
    from the YAML keys with aliases.
    """

    base: BaseParams
    ingest: IngestParams
    split: SplitParams
    validate_: ValidateParams = Field(alias="validate")
    train: TrainParams
    register_: RegisterParams = Field(alias="register")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping at the top level.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        ConfigError: If the file is missing, unreadable YAML, or not a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(content, dict):
        raise ConfigError(f"Expected a mapping at the top of {path}")
    return content


def load_settings(env: str | None = None, config_dir: Path = CONFIG_DIR) -> Settings:
    """Load and validate the settings for one environment.

    Args:
        env: Environment name. If ``None``, the ``APP_ENV`` variable is used,
            falling back to ``dev``.
        config_dir: Folder that holds the ``<env>.yaml`` files.

    Returns:
        Validated, read-only settings.

    Raises:
        ConfigError: If the environment name is unknown or the file is
            missing or invalid.
    """
    env_name = env or os.environ.get(APP_ENV_VARIABLE, DEFAULT_ENV)
    try:
        environment = Environment(env_name)
    except ValueError as error:
        allowed = ", ".join(item.value for item in Environment)
        raise ConfigError(f"Unknown environment {env_name!r}; use one of: {allowed}") from error

    file_values = _read_yaml(config_dir / f"{environment.value}.yaml")
    try:
        return Settings(**file_values)
    except ValidationError as error:
        raise ConfigError(f"Invalid settings for {environment.value!r}:\n{error}") from error


def load_params(path: Path = PARAMS_FILE) -> Params:
    """Load and validate the experiment parameters.

    Args:
        path: Location of ``params.yaml``.

    Returns:
        Validated, read-only parameters.

    Raises:
        ConfigError: If the file is missing or any value is invalid.
    """
    try:
        return Params.model_validate(_read_yaml(path))
    except ValidationError as error:
        raise ConfigError(f"Invalid parameters in {path}:\n{error}") from error
