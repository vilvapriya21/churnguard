"""Logging setup shared by every module.

Each module gets its own logger with ``logging.getLogger(__name__)``; this
module only configures where the messages go and in what format. The level
comes from the environment settings (DEBUG in dev, WARNING in prod), so the
code never hard-codes it.

Exported API:
    configure_logging: Configure the root logger once per process.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str) -> None:
    """Configure the root logger to write formatted messages to stdout.

    Calling this more than once replaces the previous configuration instead of
    adding duplicate handlers.

    Args:
        level: Standard logging level name, e.g. ``"DEBUG"`` or ``"INFO"``.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
