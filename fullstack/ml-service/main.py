"""
ml-service/main.py — FastAPI microservice for CI/CD Pipeline Failure Prediction
Loads pre-trained models from ../../models/ and serves predictions via REST API.
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Repo root is on the path so joblib can resolve model_wrappers.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import model_wrappers  # noqa: F401

warnings.filterwarnings("ignore")

# ── Path resolution (service lives in fullstack/ml-service/) ──────────────────
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Feature config (mirrors train_v3.py / predict.py) ────────────────────────
CAT_COLS = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]
NUM_COLS = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
            "cpu_usage_pct", "memory_usage_mb", "retry_count"]
ENG_NUM = [
    "sev_score",
    "total_duration_sec", "build_ratio", "test_ratio", "deploy_ratio",
    "log_build_duration_sec", "log_test_duration_sec",
    "log_deploy_duration_sec", "log_memory_usage_mb", "log_total_duration_sec",
    "cpu_x_retry", "mem_x_retry", "cpu_x_flaky", "sev_x_retry", "sev_x_flaky",
]
ENG_BOOL = [
    "is_flaky_test",
    "is_main_branch", "is_release_branch",
    "is_deploy_stage", "is_test_stage", "is_build_stage",
]
FEATURE_COLS = CAT_COLS + NUM_COLS + ENG_NUM + ENG_BOOL

TASKS = ["risk_level", "rollback_triggered", "incident_created"]

SEV_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CI/CD Failure Prediction ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache models in memory ────────────────────────────────────────────────────
_models = {}

def load_task_artifacts(task: str):
    if task in _models:
        return _models[task]
    prep  = joblib.load(os.path.join(MODELS_DIR, f"preprocessor_{task}.pkl"))
    le    = joblib.load(os.path.join(MODELS_DIR, f"label_encoder_{task}.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, f"best_model_{task}.pkl"))
    _models[task] = (prep, le, model)
    return _models[task]

# ── Request / Response models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ci_tool:            str   = Field("Jenkins",       description="CI tool used")
    language:           str   = Field("Python",        description="Programming language")
    os:                 str   = Field("ubuntu-latest", description="Operating system")
    cloud_provider:     str   = Field("AWS",           description="Cloud provider")
    branch:             str   = Field("main",          description="Git branch")
    failure_stage:      str   = Field("build",         description="Stage that failed")
    build_duration_sec: int   = Field(1200,            ge=0)
    test_duration_sec:  int   = Field(600,             ge=0)
    deploy_duration_sec:int   = Field(300,             ge=0)
    cpu_usage_pct:      float = Field(60.0,            ge=0, le=100)
    memory_usage_mb:    int   = Field(4096,            ge=0)
    retry_count:        int   = Field(1,               ge=0)
    is_flaky_test:      int   = Field(0,               ge=0, le=1)
    rollback_triggered: int   = Field(0,               ge=0, le=1)
    incident_created:   int   = Field(0,               ge=0, le=1)
    severity:           str   = Field("MEDIUM")

class TaskResult(BaseModel):
    prediction:   str
    confidence:   Optional[float]
    probabilities: dict
    model_used:   str

class PredictResponse(BaseModel):
    risk_level:         TaskResult
    rollback_triggered: TaskResult
    incident_created:   TaskResult

# ── Feature engineering (mirrors predict.py) ──────────────────────────────────
def engineer_features(req: PredictRequest) -> pd.DataFrame:
    sev_score = SEV_MAP.get(req.severity, 2)
    total     = req.build_duration_sec + req.test_duration_sec + req.deploy_duration_sec
    total_safe = total if total > 0 else 1

    row = {
        "ci_tool":             req.ci_tool,
        "language":            req.language,
        "os":                  req.os,
        "cloud_provider":      req.cloud_provider,
        "branch":              req.branch,
        "failure_stage":       req.failure_stage,
        "build_duration_sec":  req.build_duration_sec,
        "test_duration_sec":   req.test_duration_sec,
        "deploy_duration_sec": req.deploy_duration_sec,
        "cpu_usage_pct":       req.cpu_usage_pct,
        "memory_usage_mb":     req.memory_usage_mb,
        "retry_count":         req.retry_count,
        "is_flaky_test":       req.is_flaky_test,
        "rollback_triggered":  req.rollback_triggered,
        "incident_created":    req.incident_created,
        "sev_score":           sev_score,
        "total_duration_sec":  total,
        "build_ratio":         req.build_duration_sec / total_safe,
        "test_ratio":          req.test_duration_sec  / total_safe,
        "deploy_ratio":        req.deploy_duration_sec / total_safe,
        "log_build_duration_sec":  np.log1p(req.build_duration_sec),
        "log_test_duration_sec":   np.log1p(req.test_duration_sec),
        "log_deploy_duration_sec": np.log1p(req.deploy_duration_sec),
        "log_memory_usage_mb":     np.log1p(req.memory_usage_mb),
        "log_total_duration_sec":  np.log1p(total),
        "cpu_x_retry":         req.cpu_usage_pct * req.retry_count,
        "mem_x_retry":         req.memory_usage_mb * req.retry_count,
        "cpu_x_flaky":         req.cpu_usage_pct * req.is_flaky_test,
        "sev_x_retry":         sev_score * req.retry_count,
        "sev_x_flaky":         sev_score * req.is_flaky_test,
        "is_main_branch":    int(req.branch == "main"),
        "is_release_branch": int(req.branch == "release"),
        "is_deploy_stage":   int(req.failure_stage == "deploy"),
        "is_test_stage":     int(req.failure_stage == "test"),
        "is_build_stage":    int(req.failure_stage == "build"),
    }
    return pd.DataFrame([row])

def run_task(task: str, X_full: pd.DataFrame, metrics: dict) -> TaskResult:
    prep, le, model = load_task_artifacts(task)
    X     = X_full[FEATURE_COLS]
    X_enc = prep.transform(X)

    pred_enc = model.predict(X_enc)[0]
    pred_lbl = str(le.inverse_transform([pred_enc])[0])

    probas     = {}
    confidence = None
    if hasattr(model, "predict_proba"):
        raw    = model.predict_proba(X_enc)[0]
        labels = le.inverse_transform(np.arange(len(le.classes_)))
        probas = {str(lbl): round(float(p), 4) for lbl, p in zip(labels, raw)}
        confidence = round(float(max(raw)), 4)

    best_model = metrics.get(task, {}).get("__best_model__", "Unknown")
    return TaskResult(
        prediction=pred_lbl,
        confidence=confidence,
        probabilities=probas,
        model_used=best_model,
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_dir": MODELS_DIR}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        metrics_path = os.path.join(RESULTS_DIR, "metrics_v3.json")
        if not os.path.exists(metrics_path):
            metrics_path = os.path.join(RESULTS_DIR, "metrics_v2.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}

    X = engineer_features(req)
    return PredictResponse(
        risk_level         = run_task("risk_level",         X, metrics),
        rollback_triggered = run_task("rollback_triggered", X, metrics),
        incident_created   = run_task("incident_created",   X, metrics),
    )

@app.get("/metrics")
def get_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "metrics_v3.json")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(RESULTS_DIR, "metrics_v2.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="metrics.json not found")
    with open(metrics_path) as f:
        return json.load(f)

@app.get("/feature-importances/{task}")
def get_feature_importances(task: str):
    valid = {"risk_level", "rollback_triggered", "incident_created",
             "failure_type", "severity"}
    if task not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {valid}")
    fi_path = os.path.join(RESULTS_DIR, f"feature_importances_{task}.json")
    if not os.path.exists(fi_path):
        raise HTTPException(status_code=404, detail=f"Feature importances for {task} not found")
    with open(fi_path) as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
