"""Logging setup shared by every module.

Each module gets its own logger with ``logging.getLogger(__name__)``; this
module only configures where the messages go and in what format. The level
comes from the environment settings (DEBUG in dev, WARNING in prod), so the
code never hard-codes it.

The configured level applies to this project's own loggers. Third-party
libraries (matplotlib, urllib3, MLflow, ...) stay at WARNING, so turning on
DEBUG in dev shows our details without flooding the output with theirs.

Exported API:
    PROJECT_LOGGERS: Logger names that follow the configured level.
    configure_logging: Configure logging once per process.
"""

import logging
import sys
from typing import Final

LOG_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
THIRD_PARTY_LEVEL: Final = logging.WARNING
# "__main__" is the name a stage module gets when run with `python -m`.
PROJECT_LOGGERS: Final = ("churnguard", "__main__")


def configure_logging(level: str) -> None:
    """Send formatted log messages to stdout.

    Calling this more than once replaces the previous configuration instead of
    adding duplicate handlers.

    Args:
        level: Standard logging level name for project loggers, e.g.
            ``"DEBUG"`` or ``"INFO"``. Case does not matter.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    logging.basicConfig(
        level=THIRD_PARTY_LEVEL,
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    for name in PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(numeric_level)
