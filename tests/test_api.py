import pytest
from fastapi.testclient import TestClient

SAMPLE = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


@pytest.fixture()
def client(trained_model_path, monkeypatch):
    from src import app as app_module
    monkeypatch.setattr(app_module.config, "MODEL_PATH", trained_model_path)
    app_module.get_model.cache_clear()
    return TestClient(app_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_invalid_input(client):
    bad = dict(SAMPLE, age=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "model_predictions_total" in text
