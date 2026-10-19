import joblib
import numpy as np

from src import config


def test_model_artifact_saved(trained_model_path):
    assert trained_model_path.exists()


def test_model_predicts_probabilities(trained_model_path, clean_df):
    model = joblib.load(trained_model_path)
    proba = model.predict_proba(clean_df[config.FEATURES].head(10))
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_beats_chance(trained_model_path, clean_df):
    from sklearn.metrics import roc_auc_score
    model = joblib.load(trained_model_path)
    p = model.predict_proba(clean_df[config.FEATURES])[:, 1]
    assert roc_auc_score(clean_df[config.TARGET], p) > 0.6
