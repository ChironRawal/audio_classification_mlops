# Audio Classification MLOps

**End-to-end MLOps system for 12-class speech keyword classification** — from reproducible pipeline and experiment tracking to a quantized ONNX model served via a Dockerized FastAPI service, deployed on Render with Prometheus/Grafana monitoring and GitHub Actions CI.

[![CI](https://github.com/MeetSakhiya234/audio-classification-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/MeetSakhiya234/audio-classification-mlops/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8-informational)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

**Live API:** https://audio-classification-mlops.onrender.com/docs
*(Render free tier — the first request after inactivity can take ~50 s to wake the service.)*

---

## Highlights

| Area | What was built |
|---|---|
| **Model** | `superb/wav2vec2-base-superb-ks` (pretrained) on Google Speech Commands v0.02 — 10 keywords + `_unknown_` + `_silence_` |
| **Optimization** | Exported to ONNX and INT8-quantized (~117 MB); served with ONNX Runtime on CPU, **no PyTorch in the production image** |
| **Orchestration** | 5-step ZenML pipeline: load data → prepare eval set → inference → evaluate → track |
| **Experiment tracking** | MLflow runs for baseline, Optuna trials, pipeline runs; model versioned in MLflow Model Registry |
| **Serving** | FastAPI (`/predict`, `/health`, `/metrics`, Swagger docs), input validation, 503 handling when model isn't loaded |
| **Deployment** | Docker image → Render web service |
| **Monitoring** | Prometheus metrics (request count, predictions per class, errors, latency histogram) → Grafana dashboard (provisioned as code) |
| **CI** | GitHub Actions: compile check → Docker build → start container → smoke-test `/docs` and `/metrics` |

## Results

Evaluated on a class-balanced subset of the official test split (**120 samples, 10 per class**):

| Metric | Score |
|---|---:|
| Accuracy | 97.50% |
| Macro F1 | 97.53% |
| Macro Precision | 97.85% |
| Macro Recall | 97.50% |

Average inference latency observed on the Grafana dashboard: **~145 ms** per request (CPU, local Docker).

> **Scope note:** the model is a pretrained SUPERB checkpoint, not trained from scratch. The focus of this project is the production lifecycle around the model. Metrics come from a small balanced subset, not the full test set.

## Architecture

```text
Speech Commands ─► Preprocessing ─► Wav2Vec2 ─► Evaluation
                                        │
                     ZenML pipeline ────┴──── MLflow tracking + Model Registry
                                        │
                              ONNX INT8 model
                                        │
                        FastAPI ─► Docker ─► Render
                           │
                      /metrics ─► Prometheus ─► Grafana
        GitHub Actions: build + smoke tests on every push / PR
```

## Screenshots

| MLflow runs | Model Registry run | Grafana dashboard |
|---|---|---|
| ![](docs/images/mlflow-runs.png) | ![](docs/images/mlflow-registry.png) | ![](docs/images/grafana.png) |

| Prometheus target (UP) | Render deployment |
|---|---|
| ![](docs/images/prometheus-targets.png) | ![](docs/images/render-live.png) |

## API example

```bash
curl -X POST "https://audio-classification-mlops.onrender.com/predict" \
  -F "file=@sample.wav"
```

Supports `.wav`, `.mp3`, `.flac`. Interactive docs at `/docs`.

## Run locally

```bash
git clone https://github.com/MeetSakhiya234/audio-classification-mlops.git
cd audio-classification-mlops
git lfs pull                      # fetches the ~117 MB ONNX model

# API + Prometheus + Grafana
docker compose up --build
```

| Service | URL |
|---|---|
| API / Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Run the ML pipeline and MLflow UI:

```bash
pip install -r requirements.txt
python run_pipeline.py
mlflow ui --port 5001
```

## Project structure

```text
app/            FastAPI service + Prometheus instrumentation
src/            Data, preprocessing, ONNX inference, evaluation, tracking
steps/          ZenML pipeline steps
pipelines/      ZenML pipeline definition
scripts/        Evaluation, Optuna tuning, threshold evaluation, model registration
monitoring/     Prometheus config + Grafana dashboard/provisioning
.github/        CI workflow
```

## Challenges solved

- **Heavy production image:** API failed on startup because PyTorch was imported unnecessarily → moved preprocessing/inference to NumPy + ONNX Runtime and a slim `requirements-api.txt`.
- **CPU-friendly inference:** quantized the Wav2Vec2 model to INT8 ONNX.
- **MLflow registration failure:** local SQLite tracking URI conflicted with the HTTP artifact server → aligned tracking/artifact configuration and re-registered as v2.

## Roadmap

- Unit/integration tests with `pytest` and accuracy regression gate in CI
- Evaluate on the full test split and report per-class confusion matrix
- Compare INT8 vs FP32 accuracy/latency side by side
- Data/prediction drift monitoring; Prometheus alert rules
- Auto-deploy from CI and move off the free tier to remove cold starts

## Author

**Meet Sakhiya** — [LinkedIn](https://www.linkedin.com/in/YOUR-HANDLE) · [GitHub](https://github.com/MeetSakhiya234)
