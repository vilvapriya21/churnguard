"""Tests for the shared logging setup."""

import logging

import pytest

from churnguard.logging_utils import configure_logging


def test_configure_logging_valid_level_sets_root_level() -> None:
    """A lowercase level name is accepted and applied to the root logger."""
    configure_logging("warning")

    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_called_twice_keeps_single_handler() -> None:
    """Re-configuring replaces the handler instead of duplicating log lines."""
    configure_logging("INFO")
    configure_logging("DEBUG")

    assert len(logging.getLogger().handlers) == 1


def test_configure_logging_unknown_level_raises_value_error() -> None:
    """An unknown level name is rejected with a clear message."""
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("LOUD")
