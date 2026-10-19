"""Shared fixtures. Uses synthetic data shaped like the UCI file so tests run offline."""
import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    score = (df.cp == 4) * 1.5 + df.exang + df.oldpeak / 3 + (df.ca > 0) - df.thalach / 150
    df["num"] = (score + rng.normal(0, 0.5, n) > 1.2).astype(int) * rng.integers(1, 3, n)
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df[config.COLUMNS]


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from src.data import clean
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory, clean_df):
    from src.train import train
    d = tmp_path_factory.mktemp("model")
    path = d / "model.joblib"
    train(df=clean_df, model_path=path, tracking_uri=f"sqlite:///{d}/mlflow.db",
          outer_folds=3, inner_folds=2)
    return path
