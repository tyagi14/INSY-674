# INSY-674 End-to-End ML Project

[![CI](https://github.com/tyagi14/INSY-674/actions/workflows/ci.yml/badge.svg)](https://github.com/tyagi14/INSY-674/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository contains an end-to-end machine learning workflow for predicting whether a candidate is likely to look for a new job (`target`), using the HR analytics dataset included in `src/data`.

## Live deployment

| Endpoint | URL |
|----------|-----|
| API Docs (Swagger) | [https://insy674-ml-api.onrender.com/docs](https://insy674-ml-api.onrender.com/docs) |
| Health Check | [https://insy674-ml-api.onrender.com/api/v1/health](https://insy674-ml-api.onrender.com/api/v1/health) |
| Predict | `POST` [https://insy674-ml-api.onrender.com/api/v1/predict](https://insy674-ml-api.onrender.com/api/v1/predict) |

> **Note:** The free-tier Render instance spins down after inactivity. The first request after idle may take ~30-60 seconds to cold-start.

## Workflow implemented (Research to Production)
### Phase 1: Research
- Data analysis in Jupyter notebooks
- Model experimentation
- Pipeline creation

### Phase 2: Production Code
- Central config management via `src/config.yml`
- Modularized processing in `src/processing/`
- Reusable training and inference modules (`src/train_pipeline.py`, `src/predict.py`)

### Phase 3: Package Development
- Python package structure under `src/`
- Versioning in `src/VERSION`
- Dependency management with split requirements and packaging metadata (`setup.py`, `pyproject.toml`)

### Phase 4: API Development
- FastAPI REST endpoints (`/api/v1/health`, `/api/v1/predict`)
- Pydantic validation schemas
- Application logging with Loguru (with safe fallback)

### Phase 5: Deployment
- Docker containerization (`Dockerfile`)
- Render blueprint deployment config (`render.yaml`)
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`)

## Stack used
- ML & Data Processing: Scikit-learn, Feature-engine, Pandas, NumPy
- Production & Deployment: FastAPI, Docker, Render blueprint config, PyPI-ready packaging
- Development Tools: Jupyter, Tox, Loguru, Git/GitHub

## Project structure
- `src/`: ML package (config, preprocessing, training, prediction, model artifacts)
- `app-fastapi/`: REST API for model serving
- `requirements/`: split dependency files for dev/research/production
- `notebooks/`: experimentation notebooks
- `MODEL_CARD.md`: model purpose, performance snapshot, risks, and retraining policy

## Quick start
1) Create and activate virtual environment
2) Install dependencies
3) Train model
4) Run API

```bash
make bootstrap
make train
make run-api
```

Training now also writes a versioned evaluation report JSON to `src/trained_models/`.

## API endpoints
- `GET /` : basic welcome page
- `GET /api/v1/health` : API + model health/version (+ optional model metadata if available)
- `POST /api/v1/predict` : batch predictions
- `POST /api/v1/monitor/drift` : batch drift report against training baseline
- `GET /api/v1/monitor/drift/history` : recent drift checks (supports `limit` and optional `model_version`)
- `GET /api/v1/monitor/drift/summary` : aggregate drift stats for a time window (supports `window_days` and optional `model_version`)

Example health response with metadata:

```json
{
  "name": "INSY-674 Employability Prediction API",
  "api_version": "0.1.0",
  "model_version": "0.1.0",
  "model_metadata": {
    "model_version": "0.1.0",
    "trained_at": "2026-02-25T19:00:00+00:00",
    "git_sha": "abc1234",
    "metrics": {
      "accuracy": 0.81,
      "roc_auc": 0.88
    },
    "n_rows": 19158,
    "n_features": 12,
    "feature_names": ["city", "city_development_index", "gender"]
  }
}
```

Example payload:

```json
{
  "inputs": [
    {
      "city": "city_41",
      "city_development_index": 0.827,
      "gender": "Male",
      "relevent_experience": "Has relevent experience",
      "enrolled_university": "Full time course",
      "education_level": "Graduate",
      "major_discipline": "STEM",
      "experience": "9",
      "company_size": "<10",
      "company_type": "Pvt Ltd",
      "last_new_job": "1",
      "training_hours": 21
    }
  ]
}
```

Example drift request payload:

```json
{
  "inputs": [
    {
      "city": "city_41",
      "city_development_index": 0.827,
      "gender": "Male",
      "relevent_experience": "Has relevent experience",
      "enrolled_university": "Full time course",
      "education_level": "Graduate",
      "major_discipline": "STEM",
      "experience": "9",
      "company_size": "<10",
      "company_type": "Pvt Ltd",
      "last_new_job": "1",
      "training_hours": 21.0
    }
  ]
}
```

Example drift response:

```json
{
  "model_version": "0.1.0",
  "baseline_version": "0.1.0",
  "n_records": 60,
  "overall_status": "ok",
  "thresholds": {
    "warn": 0.1,
    "drifted": 0.25
  },
  "top_drifting_features": [],
  "feature_reports": [
    {
      "feature_name": "city",
      "feature_type": "categorical",
      "psi": 0.02,
      "status": "ok",
      "expected_missing_rate": 0.0,
      "observed_missing_rate": 0.0,
      "missing_rate_delta": 0.0,
      "expected_distribution": {"city_41": 0.7, "city_11": 0.3, "__OTHER__": 0.0, "__MISSING__": 0.0},
      "observed_distribution": {"city_41": 0.68, "city_11": 0.32, "__OTHER__": 0.0, "__MISSING__": 0.0}
    }
  ],
  "message": null
}
```

Example drift history request:

```bash
curl -s "http://127.0.0.1:8000/api/v1/monitor/drift/history?limit=5&model_version=0.1.0"
```

Example drift history response:

```json
{
  "model_version": "0.1.0",
  "limit": 5,
  "total_records": 12,
  "records": [
    {
      "checked_at": "2026-02-27T20:05:11.257217+00:00",
      "model_version": "0.1.0",
      "baseline_version": "0.1.0",
      "n_records": 60,
      "overall_status": "warn",
      "top_drifting_features": ["city", "training_hours"],
      "max_psi": 0.21,
      "mean_psi": 0.07,
      "n_warn_features": 2,
      "n_drifted_features": 0
    }
  ]
}
```

Example drift summary request:

```bash
curl -s "http://127.0.0.1:8000/api/v1/monitor/drift/summary?window_days=7&model_version=0.1.0"
```

Example drift summary response:

```json
{
  "model_version": "0.1.0",
  "window_days": 7,
  "total_checks": 24,
  "status_counts": {
    "ok": 16,
    "warn": 5,
    "drifted": 2,
    "insufficient_data": 1
  },
  "drift_rate": 0.0833,
  "warn_rate": 0.2083,
  "top_recurrent_features": [
    {"feature_name": "city", "count": 5},
    {"feature_name": "training_hours", "count": 3}
  ],
  "latest_check_at": "2026-02-27T20:05:11.257217+00:00"
}
```

Interpretation note:
- `overall_status = insufficient_data` means the request had fewer rows than configured `drift_min_samples` (default: `50`).
- PSI thresholds used by the API are configurable in `src/config.yml`:
  - `drift_warn_threshold` (default `0.1`)
  - `drift_alert_threshold` (default `0.25`)
- Drift history records are append-only JSONL entries generated on successful `/api/v1/monitor/drift` checks.
- `GET /api/v1/monitor/drift/history` returns `200` with an empty list when history does not exist yet.
- Query validation uses FastAPI defaults (`422`), and unexpected internal issues return `500`.

## Run tests
```bash
make test
```

## Model Governance Artifacts
- Model card: `MODEL_CARD.md`
- Training metadata artifact: `src/trained_models/*_metadata.json`
- Training evaluation artifact: `src/trained_models/*_evaluation.json`
  - contains metrics, classification report, and confusion matrix

Example command to inspect the latest evaluation artifact:

```bash
cat src/trained_models/employability_model_v0.1.0_evaluation.json
```

## Production run (Docker)
```bash
docker build -t insy674-ml-api .
docker run --rm -p 8000:8000 insy674-ml-api
```
## Deploy on Render (Free tier)
1) Push this repository to GitHub.
2) In Render dashboard, choose **New +** → **Blueprint**.
3) Select this repository. Render will read `render.yaml` and create the web service.
4) Confirm deploy; once live, open:
- `https://insy674-ml-api.onrender.com/api/v1/health`
- `https://insy674-ml-api.onrender.com/docs`

The service uses the Docker build defined in `Dockerfile`, including model training during image build.

## API smoke test
```bash
curl -s http://127.0.0.1:8000/api/v1/health
```

## Notes
- The trained model is versioned and saved under `src/trained_models/`.
- Drift baseline artifacts are versioned and saved under `src/trained_models/` as `*_drift_baseline.json`.
- Evaluation artifacts are versioned and saved under `src/trained_models/` as `*_evaluation.json`.
- Drift baseline artifacts are generated at train time and ignored by Git.
- Retraining overwrites old model artifacts to keep one active version per package version.
