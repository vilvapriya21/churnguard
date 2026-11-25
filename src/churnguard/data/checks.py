"""Common result type for data quality checks.

Every check (schema, missing values, drift, leakage) returns a
:class:`CheckResult`, so the validate stage can combine them into one report
and decide pass or fail in a single place.

Exported API:
    CheckResult: Outcome of one check, convertible to plain JSON.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one data quality check.

    Attributes:
        name: Short identifier of the check, e.g. ``"schema"``.
        passed: ``False`` if the check found a blocking problem.
        details: JSON-serialisable facts that explain the outcome.
        warnings: Non-blocking findings worth a human look.
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary for the JSON report.

        Returns:
            A dictionary with the keys ``passed``, ``warnings`` and ``details``.
        """
        return {"passed": self.passed, "warnings": self.warnings, "details": self.details}
