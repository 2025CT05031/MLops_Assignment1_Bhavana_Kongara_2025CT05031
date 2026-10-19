"""FastAPI model-serving app with request logging and Prometheus metrics."""
import logging
import time
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["prediction"])


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[54])
    sex: int = Field(..., ge=0, le=1, description="1=male, 0=female", examples=[1])
    cp: int = Field(..., ge=1, le=4, description="chest pain type", examples=[4])
    trestbps: float = Field(..., ge=50, le=300, description="resting BP", examples=[140])
    chol: float = Field(..., ge=50, le=700, description="serum cholesterol", examples=[239])
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar >120", examples=[0])
    restecg: int = Field(..., ge=0, le=2, examples=[0])
    thalach: float = Field(..., ge=40, le=250, description="max heart rate", examples=[160])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[1.2])
    slope: int = Field(..., ge=1, le=3, examples=[2])
    ca: float = Field(..., ge=0, le=4, examples=[0])
    thal: float = Field(..., ge=3, le=7, description="3,6 or 7", examples=[3])


class Prediction(BaseModel):
    prediction: int
    confidence: float
    probability_disease: float
    risk_label: str


@lru_cache(maxsize=1)
def get_model():
    log.info("Loading model from %s", config.MODEL_PATH)
    return joblib.load(config.MODEL_PATH)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        log.info("method=%s path=%s status=%s latency_ms=%.1f",
                 request.method, path, status, elapsed * 1000)


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not loaded: {exc}")


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientData):
    model = get_model()
    X = pd.DataFrame([patient.model_dump()])[config.FEATURES]
    p = float(model.predict_proba(X)[0, 1])
    pred = int(p >= 0.5)
    PREDICTIONS.labels(str(pred)).inc()
    log.info("prediction=%s probability=%.3f", pred, p)
    return Prediction(prediction=pred,
                      confidence=round(p if pred else 1 - p, 4),
                      probability_disease=round(p, 4),
                      risk_label="high risk" if pred else "low risk")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
