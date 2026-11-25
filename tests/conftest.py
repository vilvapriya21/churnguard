"""Shared pytest fixtures and synthetic data.

Tests never read the real Kaggle file. They use :func:`make_raw_customers`,
which builds small, deterministic data with the same columns and quirks as the
raw dataset (including blank ``TotalCharges`` for new customers). This keeps
tests fast, independent of DVC, and runnable in CI without downloads.
"""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from churnguard.config import MlflowSettings, Params, TrainParams, load_params
from churnguard.data.ingest import clean_raw_data
from churnguard.features.build_features import split_and_featurize

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


@pytest.fixture
def processed_split(
    clean_customers: pd.DataFrame, params: Params
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Featurised train and test rows from the fake customers."""
    return split_and_featurize(clean_customers, params.split, random_seed=0)


@pytest.fixture
def fast_train_params(params: Params) -> TrainParams:
    """Training parameters shrunk so MLflow tests run in seconds."""
    return params.train.model_copy(
        update={
            "cv_folds": 2,
            "models": {
                "logistic_regression": params.train.models["logistic_regression"],
                "random_forest": {"n_estimators": 20, "max_depth": 4, "class_weight": "balanced"},
            },
        }
    )


@pytest.fixture
def mlflow_settings(tmp_path: Path) -> Iterator[MlflowSettings]:
    """A private, throw-away MLflow tracking store and registry for one test.

    Uses a SQLite file and an artifact folder inside the test's temporary
    directory, so tests never touch the real MLflow server.
    """
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    experiment_name = "churnguard-test"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(experiment_name, artifact_location=(tmp_path / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)
    yield MlflowSettings(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registered_model_name="churnguard-test",
        champion_alias="champion",
        challenger_alias="challenger",
    )
    mlflow.end_run()
