# Breast Cancer Triage Service — MLOps Assignment 2

A containerized machine learning API that predicts breast cancer classification using a trained Logistic Regression / Random Forest model. The service uses FastAPI for prediction serving, DVC for dataset versioning, Docker for containerization, and Redis for prediction request counting.

## Project Architecture

- **Data versioning:** DVC
- **Model training:** Scikit-learn
- **API:** FastAPI
- **Containerization:** Docker
- **Orchestration:** Docker Compose
- **Request counter:** Redis
- **Image registry:** Docker Hub

## Repository Structure

```text
app/                  FastAPI application
training/             Dataset preparation and model training
data/raw/             DVC-tracked dataset pointer
dvc-remote/           Local DVC remote storage
artifacts/            Generated model and metrics
.dvc/                 DVC configuration
params.yaml           Training parameters
dvc.yaml              DVC training pipeline
Dockerfile            Multi-stage production image
Dockerfile.naive      Baseline Docker image
docker-compose.yml    Local API and Redis stack
docker-compose.hub.yml Registry-based stack
```

## 1. Clone the Repository

```bash
git clone https://github.com/qamar-prog/MLOPS-Assignment-2.git
cd MLOPS-Assignment-2
```

Create and activate a virtual environment:

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements-train.txt
pip install -r requirements-serve.txt
pip install -r requirements-dev.txt
```

## 2. Restore the Dataset Using DVC

Install DVC if required:

```powershell
pip install dvc
```

Restore the dataset:

```powershell
dvc pull data/raw/breast_cancer.csv.dvc
```

The dataset is stored using DVC rather than directly in Git.

## 3. Run the Training Pipeline

Run the complete DVC pipeline:

```powershell
dvc repro
```

Alternatively, run training directly:

```powershell
python training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
```

Generated artifacts include:

- `artifacts/model.joblib`
- `artifacts/model_card.json`
- `artifacts/cv_results.json`
- `artifacts/metrics.json`

## 4. Run the FastAPI Service Locally

```powershell
uvicorn app.main:app --reload
```

Open the interactive API documentation:

http://localhost:8000/docs

Available endpoints:

- `GET /health`
- `POST /predict`
- `GET /model/info`
- `GET /stats`

## 5. Build the Production Docker Image

Restore the dataset before building:

```powershell
dvc pull data/raw/breast_cancer.csv.dvc
```

Build the production image:

```powershell
$GIT_SHA = git rev-parse --short HEAD

docker build `
  -f Dockerfile `
  -t breast-cancer-triage:1.0.0 `
  --build-arg MODEL_VERSION=1.0.0 `
  --build-arg GIT_SHA=$GIT_SHA `
  .
```

## 6. Start the Local Docker Compose Stack

```powershell
docker compose up --build -d
```

Check running containers:

```powershell
docker compose ps
```

Open:

http://localhost:8000/docs

Stop the stack:

```powershell
docker compose down
```

## 7. Publish the Image to Docker Hub

Docker Hub repository:

https://hub.docker.com/repository/docker/qamartech/ml-assignment-2/

Login:

```powershell
docker login
```

Tag the image:

```powershell
docker tag breast-cancer-triage:1.0.0 qamartech/ml-assignment-2:1.0.0
docker tag breast-cancer-triage:1.0.0 qamartech/ml-assignment-2:latest
```

Push both tags:

```powershell
docker push qamartech/ml-assignment-2:1.0.0
docker push qamartech/ml-assignment-2:latest
```

Pull the published image:

```powershell
docker pull qamartech/ml-assignment-2:1.0.0
```

## 8. Run the Registry-Based Compose Stack

The registry-based Compose configuration uses the published image and does not build from source.

```powershell
docker compose -f docker-compose.hub.yml pull
docker compose -f docker-compose.hub.yml up -d
```

Check services:

```powershell
docker compose -f docker-compose.hub.yml ps
```

Open:

http://localhost:8000/docs

Stop the stack:

```powershell
docker compose -f docker-compose.hub.yml down
```

## Environment Variables

| Variable | Default / Example | Purpose |
|---|---|---|
| MODEL_PATH | artifacts/model.joblib | Model location |
| MODEL_CARD_PATH | artifacts/model_card.json | Model metadata |
| DECISION_THRESHOLD | 0.5 | Classification threshold |
| LOG_LEVEL | INFO | Logging level |
| REDIS_HOST | redis | Redis hostname |
| REDIS_PORT | 6379 | Redis port |
| API_PORT | 8000 | Host API port |

## Dataset

The project uses the Scikit-learn Breast Cancer Wisconsin dataset, with a corrected version containing 539 rows and 30 input features plus the target column.

## Docker Hub

Published image:

`qamartech/ml-assignment-2:1.0.0`

`qamartech/ml-assignment-2:latest`

## License

Academic project developed for the MLOps assignment.