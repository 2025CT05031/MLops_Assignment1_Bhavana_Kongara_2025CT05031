# Heart Disease Risk Prediction - MLOps Assignment 01
Submitted by: Bhavana Kongara (2025CT05031), BITS Pilani, AIMLCZG523

Classifier predicting heart-disease risk from the UCI Cleveland dataset, with MLflow tracking,
pytest + GitHub Actions CI, Docker, Kubernetes, and Prometheus/Grafana monitoring.

## Repository layout
```
src/            config, data, features (preprocessing), eda, train (MLflow), app (FastAPI)
scripts/        download_data.py
notebooks/      01_eda, 02_training, 03_inference
tests/          pytest suite (data, model, API) - runs offline on synthetic data
k8s/            deployment.yaml, service.yaml, monitoring.yaml (Prometheus + Grafana)
monitoring/     prometheus / grafana config for docker-compose
.github/workflows/ci.yml   lint -> test -> train -> docker build + smoke test
docs/REPORT.md  report template (export to PDF for submission)
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py     # downloads UCI data + writes data/processed/heart_disease_clean.csv
```
If the download fails, manually save `processed.cleveland.data` from
https://archive.ics.uci.edu/dataset/45/heart+disease into `data/raw/` and rerun the script.

## 2. EDA, training, tracking
```bash
python -m src.eda          # plots -> reports/eda/
python -m src.train        # nested CV + tuning for LogReg & RandomForest; logs to MLflow; saves models/model.joblib
make mlflow-ui             # http://localhost:5000
```
MLflow logs per run: parameters, nested-CV and held-out metrics (accuracy, precision, recall, F1, ROC-AUC),
ROC/confusion-matrix plots, and the full sklearn Pipeline (preprocessing + model) with signature.

## 3. Tests & lint
```bash
make lint && make test
```

## 4. Serve locally
```bash
uvicorn src.app:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Endpoints: `POST /predict`, `GET /health`, `GET /metrics`, `GET /docs` (Swagger).

## 5. Docker (train first so `models/model.joblib` exists)
```bash
docker build -t heart-api:latest .
docker run --rm -p 8000:8000 heart-api:latest
# full monitoring stack locally:
docker compose up --build        # API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
```

## 6. Kubernetes (Minikube or Docker Desktop)
```bash
# Minikube: build the image into minikube's docker daemon
eval $(minikube docker-env) && docker build -t heart-api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/monitoring.yaml
kubectl get pods,svc
minikube tunnel                  # separate terminal -> LoadBalancer gets an external IP
# or: minikube service heart-api --url
curl http://<EXTERNAL-IP>/health
```
Docker Desktop: enable Kubernetes, skip the minikube step; service is on `localhost:80`.

## 7. Grafana dashboard
Add a Prometheus data source (provisioned automatically) and create panels with:
- Request rate: `sum(rate(api_requests_total[1m])) by (endpoint)`
- p95 latency: `histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[5m])) by (le))`
- Error rate: `sum(rate(api_requests_total{status=~"5.."}[1m]))`
- Predictions by class: `sum(model_predictions_total) by (prediction)`

Generate traffic: `for i in $(seq 100); do curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{...}'; done`
