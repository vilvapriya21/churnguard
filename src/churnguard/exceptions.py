"""Project-specific exceptions.

Using named exceptions instead of generic ``ValueError`` or ``Exception`` lets
callers (pipeline stages, CI jobs, the API) react to each failure type on
purpose, and makes logs easier to read.

Exported API:
    ChurnGuardError: Base class for every error raised by this package.
    ConfigError: Settings or parameter files are missing or invalid.
    DataValidationError: Data broke the schema contract or quality gates.
    ModelNotFoundError: No model is available to serve predictions.
"""


class ChurnGuardError(Exception):
    """Base class for all ChurnGuard errors."""


class ConfigError(ChurnGuardError):
    """Raised when a settings or parameter file is missing or invalid."""


class DataValidationError(ChurnGuardError):
    """Raised when data fails the schema contract or a quality gate."""


class ModelNotFoundError(ChurnGuardError):
    """Raised when no trained or registered model can be loaded."""
