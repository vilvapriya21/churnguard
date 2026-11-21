"""Shared pytest fixtures and synthetic data.

Tests never read the real Kaggle file. They use :func:`make_raw_customers`,
which builds small, deterministic data with the same columns and quirks as the
raw dataset (including blank ``TotalCharges`` for new customers). This keeps
tests fast, independent of DVC, and runnable in CI without downloads.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnguard.config import Params, load_params
from churnguard.data.ingest import clean_raw_data

YES_NO = ("Yes", "No")
INTERNET_SERVICE = ("DSL", "Fiber optic", "No")
CONTRACT = ("Month-to-month", "One year", "Two year")
PAYMENT_METHOD = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)


def make_raw_customers(n_rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """Build fake customers shaped like the raw Kaggle CSV.

    Churn is made more likely for month-to-month contracts and short tenure,
    so models have a real signal to learn. Customers with ``tenure == 0`` get a
    blank ``" "`` ``TotalCharges``, exactly like the real file.

    Args:
        n_rows: Number of customers.
        seed: Random seed, so the same arguments give the same data.

    Returns:
        A data frame with the 21 raw columns, all values as the CSV holds them.
    """
    rng = np.random.default_rng(seed)

    def pick(options: tuple[str, ...]) -> np.ndarray:
        return rng.choice(options, size=n_rows)

    tenure = rng.integers(0, 73, size=n_rows)
    contract = pick(CONTRACT)
    monthly_charges = rng.uniform(18.0, 120.0, size=n_rows).round(2)
    total_charges = (monthly_charges * tenure).round(2).astype(str).astype(object)
    total_charges[tenure == 0] = " "

    risk = (
        1.5 * (contract == "Month-to-month")
        + 1.0 * (tenure < 12)
        + rng.normal(0.0, 0.6, size=n_rows)
    )
    churn = np.where(risk > 1.5, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"{index:04d}-TEST" for index in range(n_rows)],
            "gender": pick(("Female", "Male")),
            "SeniorCitizen": rng.integers(0, 2, size=n_rows),
            "Partner": pick(YES_NO),
            "Dependents": pick(YES_NO),
            "tenure": tenure,
            "PhoneService": pick(YES_NO),
            "MultipleLines": pick(("Yes", "No", "No phone service")),
            "InternetService": pick(INTERNET_SERVICE),
            "OnlineSecurity": pick(("Yes", "No", "No internet service")),
            "OnlineBackup": pick(("Yes", "No", "No internet service")),
            "DeviceProtection": pick(("Yes", "No", "No internet service")),
            "TechSupport": pick(("Yes", "No", "No internet service")),
            "StreamingTV": pick(("Yes", "No", "No internet service")),
            "StreamingMovies": pick(("Yes", "No", "No internet service")),
            "Contract": contract,
            "PaperlessBilling": pick(YES_NO),
            "PaymentMethod": pick(PAYMENT_METHOD),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )


@pytest.fixture
def raw_customers() -> pd.DataFrame:
    """Fake raw customers, as read from the CSV."""
    return make_raw_customers()


@pytest.fixture
def clean_customers(raw_customers: pd.DataFrame) -> pd.DataFrame:
    """Fake customers after the ingest stage's cleaning."""
    return clean_raw_data(raw_customers)


@pytest.fixture
def params() -> Params:
    """The project's real parameters from params.yaml."""
    return load_params()


@pytest.fixture
def raw_csv(tmp_path: Path, raw_customers: pd.DataFrame) -> Path:
    """Fake raw customers written to a temporary CSV file."""
    path = tmp_path / "raw" / "customers.csv"
    path.parent.mkdir()
    raw_customers.to_csv(path, index=False)
    return path
