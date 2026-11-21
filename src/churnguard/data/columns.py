"""Column names of the Telco Customer Churn dataset.

This is the single place that knows the dataset's column names. Ingest,
feature building, data contracts and the API all import from here, so a
renamed column is changed once instead of in several files.

Exported API:
    ID_COLUMN, TARGET_COLUMN: Identifier and label columns (never features).
    POSITIVE_LABEL: Value of ``TARGET_COLUMN`` that means "customer churned".
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS: Raw input columns by kind.
    RAW_FEATURE_COLUMNS: All raw columns the model may use as input.
"""

from typing import Final

ID_COLUMN: Final = "customerID"
TARGET_COLUMN: Final = "Churn"
POSITIVE_LABEL: Final = "Yes"

TENURE: Final = "tenure"
MONTHLY_CHARGES: Final = "MonthlyCharges"
TOTAL_CHARGES: Final = "TotalCharges"
SENIOR_CITIZEN: Final = "SeniorCitizen"

NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    SENIOR_CITIZEN,
    TENURE,
    MONTHLY_CHARGES,
    TOTAL_CHARGES,
)

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)

RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
