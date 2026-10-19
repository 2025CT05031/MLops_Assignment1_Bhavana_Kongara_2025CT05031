"""Data acquisition and cleaning."""
import logging
import urllib.request

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def download_data(dest=config.RAW_PATH, url=config.UCI_URL):
    """Download the Cleveland heart disease file from UCI (idempotent)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Raw data already present at %s", dest)
        return dest
    log.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_raw(path=config.RAW_PATH):
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values="?")


def clean(df):
    """Coerce types, binarise the target (num>0 -> 1), drop duplicates.

    Missing values (ca, thal) are intentionally kept as NaN here and imputed
    inside the sklearn pipeline so that imputation is fitted on training folds only
    (prevents leakage).
    """
    df = df.copy()
    df = df.replace("?", np.nan)
    for col in config.COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)
    return df


def prepare_and_save(raw_path=config.RAW_PATH, out=config.CLEAN_PATH):
    df = clean(load_raw(raw_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Saved cleaned data: %s rows -> %s", len(df), out)
    return df


def load_clean(path=config.CLEAN_PATH):
    return pd.read_csv(path)
