"""Tests for loading environment settings and experiment parameters."""

from pathlib import Path

import pytest
import yaml

from churnguard.config import (
    PROJECT_ROOT,
    Environment,
    load_params,
    load_settings,
)
from churnguard.exceptions import ConfigError


@pytest.mark.parametrize("env", ["dev", "staging", "prod"])
def test_load_settings_each_environment_file_is_valid(env: str) -> None:
    """Every committed configs/<env>.yaml loads and reports its own env name."""
    settings = load_settings(env)

    assert settings.env == Environment(env)


def test_load_settings_without_argument_uses_app_env_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no argument, the environment is taken from APP_ENV."""
    monkeypatch.setenv("APP_ENV", "staging")

    assert load_settings().env is Environment.STAGING


def test_load_settings_environment_variable_overrides_yaml_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CHURNGUARD_ variable wins over the value in the YAML file."""
    monkeypatch.setenv("CHURNGUARD_MLFLOW__TRACKING_URI", "http://mlflow:5000")

    settings = load_settings("prod")

    assert settings.mlflow.tracking_uri == "http://mlflow:5000"


def test_load_settings_unknown_environment_raises_config_error() -> None:
    """An environment name outside dev/staging/prod is rejected."""
    with pytest.raises(ConfigError, match="Unknown environment"):
        load_settings("qa")


def test_load_settings_missing_file_raises_config_error(tmp_path: Path) -> None:
    """A missing settings file gives a clear ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_settings("dev", config_dir=tmp_path)


def test_load_settings_invalid_log_level_raises_config_error(tmp_path: Path) -> None:
    """A log level that is not a standard name fails validation."""
    content = yaml.safe_load((PROJECT_ROOT / "configs" / "dev.yaml").read_text())
    content["log_level"] = "LOUD"
    (tmp_path / "dev.yaml").write_text(yaml.safe_dump(content))

    with pytest.raises(ConfigError, match="Invalid settings"):
        load_settings("dev", config_dir=tmp_path)


def test_settings_resolve_relative_path_is_anchored_at_project_root() -> None:
    """Relative paths from the config are turned into absolute project paths."""
    settings = load_settings("dev")

    resolved = settings.resolve(settings.paths.raw_data)

    assert resolved.is_absolute()
    assert resolved.is_relative_to(PROJECT_ROOT)


def test_load_params_project_file_is_valid() -> None:
    """The committed params.yaml passes validation."""
    params = load_params()

    assert params.base.target_column == "Churn"
    assert params.validate_.psi_warn < params.validate_.psi_fail


def test_load_params_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """A misspelled key in params.yaml is caught instead of silently ignored."""
    content = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())
    content["split"]["tset_size"] = 0.3  # deliberate typo
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(content))

    with pytest.raises(ConfigError, match="tset_size"):
        load_params(params_file)


def test_load_params_psi_warn_above_fail_raises_config_error(tmp_path: Path) -> None:
    """The drift warning limit must be lower than the failure limit."""
    content = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())
    content["validate"]["psi_warn"] = 0.5
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(content))

    with pytest.raises(ConfigError, match="psi_warn must be lower"):
        load_params(params_file)


def test_load_params_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    """A YAML file whose top level is not a mapping is rejected."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("- just\n- a list\n")

    with pytest.raises(ConfigError, match="Expected a mapping"):
        load_params(params_file)
