import numpy as np

from src import config
from src.data import clean
from src.features import build_preprocessor


def test_clean_binarises_target(clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {5}
    assert "num" not in clean_df.columns


def test_clean_handles_question_marks(raw_df):
    dirty = raw_df.astype(object).copy()
    dirty.loc[0, "ca"] = "?"
    out = clean(dirty)
    assert np.issubdtype(out["ca"].dtype, np.floating)
    assert out["ca"].isna().sum() >= 1


def test_clean_has_expected_columns(clean_df):
    assert list(clean_df.columns) == config.FEATURES + [config.TARGET] or \
        set(clean_df.columns) == set(config.FEATURES + [config.TARGET])


def test_preprocessor_outputs_no_nans(clean_df):
    X = build_preprocessor().fit_transform(clean_df[config.FEATURES])
    assert not np.isnan(X.toarray() if hasattr(X, "toarray") else X).any()


def test_preprocessor_handles_unseen_category(clean_df):
    prep = build_preprocessor().fit(clean_df[config.FEATURES])
    row = clean_df[config.FEATURES].head(1).copy()
    row["cp"] = 99
    assert prep.transform(row).shape[0] == 1
