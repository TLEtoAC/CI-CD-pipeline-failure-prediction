"""
train.py — ML Training Pipeline for CI/CD Pipeline Failure Prediction
Trains 4 models × 3 tasks and saves models, metrics, feature importances,
and confusion matrix PNGs.

Targets:
  1. risk_level     — Engineered 4-class risk score (LOW/MEDIUM/HIGH/CRITICAL)
                      derived from severity, rollback, incident, retry, cpu, memory.
                      Has real correlations → produces meaningful ML results.
  2. rollback_triggered — Binary: will this pipeline trigger a rollback?
  3. incident_created   — Binary: will this pipeline create an incident?

Note: The raw 'failure_type' and 'severity' columns in this dataset are
randomly generated labels with no correlation to the features (verified via EDA).
The engineered risk_level target combines multiple real signals for meaningful prediction.
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH   = "ci_cd_pipeline_failure_logs_dataset.csv"
MODELS_DIR  = "models"
RESULTS_DIR = "results"
CM_DIR      = os.path.join(RESULTS_DIR, "confusion_matrices")

for d in [MODELS_DIR, RESULTS_DIR, CM_DIR]:
    os.makedirs(d, exist_ok=True)

BG_COLOR   = "#0f1117"
TEXT_COLOR = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#2e3150",
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": "#2e3150",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

# ─────────────────────────────────────────────
# Feature config
# ─────────────────────────────────────────────
# Drop ID/free-text columns not useful for ML
DROP_COLS = ["pipeline_id", "run_id", "timestamp", "commit_hash",
             "author", "repository", "error_message", "error_code",
             "failure_type", "severity"]   # raw random labels excluded

CAT_COLS  = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]
NUM_COLS  = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
             "cpu_usage_pct", "memory_usage_mb", "retry_count"]
BOOL_COLS = ["is_flaky_test"]

TASKS = {
    "risk_level":         {"type": "multiclass", "label": "Pipeline Risk Level (Engineered)"},
    "rollback_triggered": {"type": "binary",     "label": "Rollback Triggered Prediction"},
    "incident_created":   {"type": "binary",     "label": "Incident Created Prediction"},
}


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and the risk_level target."""
    df = df.copy()

    # Boolean → int
    for col in ["is_flaky_test", "rollback_triggered", "incident_created"]:
        df[col] = df[col].astype(int)

    # Severity ordinal encoding for score computation
    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    df["sev_score"] = df["severity"].map(sev_map)

    # ── Engineered risk score ──────────────────────────────────────
    # Weighted combination of meaningful signals
    df["risk_score"] = (
        df["sev_score"] * 2.0 +
        df["rollback_triggered"] * 3.0 +
        df["incident_created"] * 2.0 +
        df["retry_count"] * 0.5 +
        (df["cpu_usage_pct"] / 20.0) +
        (df["memory_usage_mb"] / 5000.0) +
        df["is_flaky_test"] * 1.0
    )

    # Quartile-based 4-class risk label
    df["risk_level"] = pd.qcut(
        df["risk_score"], q=4,
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ).astype(str)

    # ── Extra engineered features ──────────────────────────────────
    df["total_duration_sec"] = (
        df["build_duration_sec"] + df["test_duration_sec"] + df["deploy_duration_sec"]
    )
    df["build_ratio"] = df["build_duration_sec"] / (df["total_duration_sec"] + 1)
    df["test_ratio"]  = df["test_duration_sec"]  / (df["total_duration_sec"] + 1)
    df["cpu_x_retry"] = df["cpu_usage_pct"] * df["retry_count"]
    df["mem_x_retry"] = df["memory_usage_mb"] * df["retry_count"]
    df["is_main_branch"]    = (df["branch"] == "main").astype(int)
    df["is_release_branch"] = (df["branch"] == "release").astype(int)
    df["is_deploy_stage"]   = (df["failure_stage"] == "deploy").astype(int)

    return df


# ─────────────────────────────────────────────
# Load & preprocess
# ─────────────────────────────────────────────
def load_data():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    print(f"   Shape after engineering: {df.shape}")
    return df


def build_preprocessor(cat_cols, num_cols, bool_cols):
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_transformer = StandardScaler()
    return ColumnTransformer(transformers=[
        ("cat",  cat_transformer, cat_cols),
        ("num",  num_transformer, num_cols),
        ("bool", "passthrough",   bool_cols),
    ])


def get_feature_names(preprocessor, cat_cols, num_cols, bool_cols):
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    return cat_names + num_cols + bool_cols


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="saga", C=1.0,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400, max_depth=9, learning_rate=0.08,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        ),
    }


# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, task_name, model_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, linecolor="#0f1117",
                vmin=0, vmax=100,
                annot_kws={"size": 9},
                cbar_kws={"label": "% per true class"})
    ax.set_title(f"{model_name} — {task_name}\nConfusion Matrix (%)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()

    safe_model = model_name.replace(" ", "_")
    path = os.path.join(CM_DIR, f"cm_{task_name}_{safe_model}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# Feature importance plot
# ─────────────────────────────────────────────
def plot_feature_importance(importances, feature_names, task_name, model_name, top_n=20):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(np.array(feature_names)[idx], importances[idx], color=colors)
    ax.set_title(f"Top-{top_n} Feature Importances\n{model_name} — {task_name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    safe_model = model_name.replace(" ", "_")
    path = os.path.join(RESULTS_DIR, f"fi_{task_name}_{safe_model}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# Train & evaluate one task
# ─────────────────────────────────────────────
def train_task(df: pd.DataFrame, task_name: str, task_cfg: dict):
    print(f"\n{'='*60}")
    print(f"🎯 Task: {task_cfg['label']}")
    print(f"{'='*60}")

    # Feature columns for this task
    # Exclude task-specific leakage columns
    exclude = {"risk_level", "risk_score", "sev_score", "rollback_triggered",
               "incident_created", "failure_type", "severity"}
    if task_name == "rollback_triggered":
        active_bool = ["is_flaky_test"]
    elif task_name == "incident_created":
        active_bool = ["is_flaky_test"]
    else:
        # risk_level: can use rollback and incident as features
        active_bool = ["is_flaky_test", "rollback_triggered", "incident_created"]
        exclude -= {"rollback_triggered", "incident_created"}

    eng_num = ["total_duration_sec", "build_ratio", "test_ratio",
               "cpu_x_retry", "mem_x_retry"]
    eng_bool = ["is_main_branch", "is_release_branch", "is_deploy_stage"]

    all_cat  = CAT_COLS
    all_num  = NUM_COLS + eng_num
    all_bool = active_bool + eng_bool

    feature_cols = all_cat + all_num + all_bool

    X = df[feature_cols].copy()

    # Encode target
    le = LabelEncoder()
    y  = le.fit_transform(df[task_name].astype(str))
    class_names = [str(c) for c in le.classes_]
    n_classes   = len(class_names)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,} | Classes: {n_classes} {class_names}")

    # Preprocessor
    preprocessor = build_preprocessor(all_cat, all_num, all_bool)
    X_train_enc  = preprocessor.fit_transform(X_train)
    X_test_enc   = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, all_cat, all_num, all_bool)

    # Save preprocessor + label encoder
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"preprocessor_{task_name}.pkl"))
    joblib.dump(le,           os.path.join(MODELS_DIR, f"label_encoder_{task_name}.pkl"))

    models   = get_models()
    task_results = {}
    best_f1  = -1
    best_model_name = None
    best_model_obj  = None
    feature_importances = {}

    for model_name, model in models.items():
        print(f"\n   ▶ Training {model_name}...")
        t0 = time.time()

        model.fit(X_train_enc, y_train)
        y_pred = model.predict(X_test_enc)

        elapsed = time.time() - t0
        acc = accuracy_score(y_test, y_pred)
        avg = "binary" if n_classes == 2 else "weighted"
        f1  = f1_score(y_test, y_pred, average=avg)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # ROC-AUC
        try:
            if n_classes == 2:
                proba = model.predict_proba(X_test_enc)[:, 1]
                auc = roc_auc_score(y_test, proba)
            else:
                proba = model.predict_proba(X_test_enc)
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"      Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_str} | Time: {elapsed:.1f}s")

        # Confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, class_names, task_name, model_name)

        # Feature importance
        fi_path = None
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi_path = plot_feature_importance(importances, feature_names, task_name, model_name)
            feature_importances[model_name] = {
                fn: float(imp) for fn, imp in zip(feature_names, importances)
            }
        elif hasattr(model, "coef_"):
            raw = model.coef_
            importances = np.abs(raw).mean(axis=0) if raw.ndim > 1 else np.abs(raw[0])
            fi_path = plot_feature_importance(importances, feature_names, task_name, model_name)
            feature_importances[model_name] = {
                fn: float(imp) for fn, imp in zip(feature_names, importances)
            }

        task_results[model_name] = {
            "accuracy":               round(acc, 4),
            "f1_weighted":            round(f1, 4),
            "roc_auc":                round(auc, 4) if auc is not None else None,
            "train_time_sec":         round(elapsed, 2),
            "classification_report":  report,
            "confusion_matrix_path":  cm_path,
            "feature_importance_path": fi_path,
        }

        if f1 > best_f1:
            best_f1         = f1
            best_model_name = model_name
            best_model_obj  = model

    # Save best model
    print(f"\n   ⭐ Best model: {best_model_name} (F1={best_f1:.4f})")
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, f"best_model_{task_name}.pkl"))
    task_results["__best_model__"] = best_model_name

    # Save feature importances
    fi_out = os.path.join(RESULTS_DIR, f"feature_importances_{task_name}.json")
    with open(fi_out, "w") as f:
        json.dump(feature_importances, f, indent=2)

    return task_results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("🚀 CI/CD Pipeline Failure Prediction — Training Pipeline")
    print("="*60)
    print("Strategy: Engineered risk_level target + binary tasks")
    print("="*60)

    df = load_data()

    all_results = {}
    for task_name, task_cfg in TASKS.items():
        all_results[task_name] = train_task(df, task_name, task_cfg)

    # Save all metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All metrics saved to {metrics_path}")

    # Summary table
    print("\n" + "="*60)
    print("📊 Summary (Test Set)")
    print("="*60)
    print(f"{'Task':<22} {'Model':<22} {'Accuracy':>10} {'F1 (w)':>10} {'AUC':>10}")
    print("-"*80)
    for task_name, task_res in all_results.items():
        for model_name, metrics in task_res.items():
            if model_name.startswith("__"):
                continue
            acc  = metrics["accuracy"]
            f1   = metrics["f1_weighted"]
            auc  = metrics["roc_auc"]
            auc_s = f"{auc:.4f}" if auc is not None else "N/A"
            best = " ⭐" if model_name == task_res.get("__best_model__") else ""
            print(f"{task_name:<22} {(model_name+best):<24} {acc:>10.4f} {f1:>10.4f} {auc_s:>10}")

    print("\n✅ Training complete! Models saved to 'models/', results to 'results/'")


if __name__ == "__main__":
    main()
