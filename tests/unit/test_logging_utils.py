"""Tests for the shared logging setup."""

import logging

import pytest

from churnguard.logging_utils import PROJECT_LOGGERS, configure_logging


def test_configure_logging_valid_level_sets_project_logger_levels() -> None:
    """A lowercase level name is accepted and applied to project loggers."""
    configure_logging("debug")

    for name in PROJECT_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG


def test_configure_logging_debug_keeps_third_party_loggers_quiet() -> None:
    """DEBUG for our code does not switch on DEBUG output of other libraries."""
    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.WARNING
    assert not logging.getLogger("matplotlib").isEnabledFor(logging.DEBUG)


def test_configure_logging_called_twice_keeps_single_handler() -> None:
    """Re-configuring replaces the handler instead of duplicating log lines."""
    configure_logging("INFO")
    configure_logging("DEBUG")

    assert len(logging.getLogger().handlers) == 1


def test_configure_logging_unknown_level_raises_value_error() -> None:
    """An unknown level name is rejected with a clear message."""
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("LOUD")
