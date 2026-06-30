# UPI Fraud Detection — Production MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.11-blue?logo=mlflow)
![Evidently](https://img.shields.io/badge/Evidently-AI-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

End-to-end MLOps pipeline for real-time UPI transaction fraud detection — featuring drift monitoring, SHAP explainability, and MLflow experiment tracking.

---

## Architecture

```mermaid
flowchart LR
    User([User / Client]) -->|form input| SD[Streamlit Dashboard\nport 8501]
    SD -->|POST /score| API[FastAPI\nport 8000]
    SD -->|GET /drift/drift-report| API
    SD -->|search_runs| ML[(MLflow\nmlflow.db)]

    API --> LGB[LightGBM\nmodel.pkl]
    API --> SH[SHAP Explainer]
    API --> EV[Evidently AI\nDataDriftPreset]
    API --> DB[(predictions.db\nSQLite)]

    LGB --> SH
    EV -->|reference_data.parquet| EV
    DB -->|last 500 rows| EV

    ML -->|upi-fraud-lgbm v1| REG[Model Registry]
```

---

## Quick Start

```bash
git clone https://github.com/RohitRathod0/Upi-Fraud-detection.git
cd Upi-Fraud-detection/upi-fraud-mlops
pip install -r requirements.txt
docker-compose up --build
```

Services start at:
| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5001 |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/score` | Score a transaction → fraud probability + SHAP top-3 |
| `GET` | `/drift/drift-report` | JSON drift summary (cached 1 hr) |
| `GET` | `/drift/drift-report/html` | Full Evidently HTML report download |
| `GET` | `/drift/drift-report/status` | Lightweight status: `ok / warning / critical` |
| `GET` | `/health` | Model version + API liveness |

---

## AWS EC2 Deployment

An automated deployment script is provided to spin up the entire stack on an AWS EC2 instance (Ubuntu).

1. Launch an Ubuntu EC2 instance.
2. In your AWS Security Group, open inbound ports: **8000** (FastAPI), **8501** (Streamlit), **5001** (MLflow), and **22** (SSH).
3. SSH into your instance, clone this repository, and run the deployment script:

```bash
git clone https://github.com/RohitRathod0/Upi-Fraud-detection.git
cd Upi-Fraud-detection/upi-fraud-mlops
chmod +x deploy_ec2.sh
./deploy_ec2.sh
```

---

## Project Structure

```
upi-fraud-mlops/
├── api/               # FastAPI app + drift router
├── src/               # Feature pipeline, training, evaluation, MLflow utils
├── monitoring/        # Standalone drift CLI + alerts log
├── dashboard/         # Streamlit multi-tab UI
├── data/              # Raw CSV, reference parquet, predictions DB, drift reports
├── models/            # model.pkl, pipeline.pkl
├── docker/            # Dockerfiles for API and dashboard
└── docker-compose.yml
```

---



