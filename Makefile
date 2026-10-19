.PHONY: setup data eda train test lint serve docker-build docker-run mlflow-ui k8s-deploy

setup:
	pip install -r requirements.txt
data:
	python scripts/download_data.py
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest -v --cov=src
lint:
	flake8 src tests scripts
serve:
	uvicorn src.app:app --reload --port 8000
docker-build:
	docker build -t heart-api:latest .
docker-run:
	docker run --rm -p 8000:8000 heart-api:latest
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
k8s-deploy:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/monitoring.yaml
