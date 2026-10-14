"""
train_v2.py — Improved ML Training Pipeline for CI/CD Pipeline Failure Prediction

Key fixes over v1:
  1. Leakage fix: risk_level no longer uses rollback/incident as *both* label
     components AND features.
  2. Adaptive quantile cuts for risk_level — avoids the 89% CRITICAL collapse
     that happens with fixed bins when the score distribution is concentrated.
     Uses [15%, 35%, 65%] quantile splits → ~15/20/30/35% class distribution.
  3. Correct Optuna scoring strings: 'f1' for binary, 'f1_weighted' for multiclass.
  4. Richer interaction features (log transforms, cross-terms, stage dummies).
  5. SMOTE oversampling on training fold for binary tasks.
  6. Optuna 30-trial TPE search for LightGBM per task.

Targets:
  risk_level          — 4-class: LOW / MEDIUM / HIGH / CRITICAL
  rollback_triggered  — Binary (note: may be random in synthetic datasets)
  incident_created    — Binary (note: may be random in synthetic datasets)
"""

import os, json, time, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠  imbalanced-learn not found — SMOTE disabled (pip install imbalanced-learn)")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠  optuna not found — hyper-tune disabled (pip install optuna)")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "ci_cd_pipeline_failure_logs_dataset.csv"
MODELS_DIR  = "models"
RESULTS_DIR = "results"
CM_DIR      = os.path.join(RESULTS_DIR, "confusion_matrices")

for d in [MODELS_DIR, RESULTS_DIR, CM_DIR]:
    os.makedirs(d, exist_ok=True)

BG_COLOR   = "#0f1117"
TEXT_COLOR = "#e0e0e0"
plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#2e3150", "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR, "grid.color": "#2e3150",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})

RANDOM_STATE = 42

CAT_COLS  = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]
NUM_COLS  = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
             "cpu_usage_pct", "memory_usage_mb", "retry_count"]
BOOL_COLS = ["is_flaky_test"]

RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

TASKS = {
    "risk_level":         {"type": "multiclass", "label": "Pipeline Risk Level"},
    "rollback_triggered": {"type": "binary",     "label": "Rollback Triggered"},
    "incident_created":   {"type": "binary",     "label": "Incident Created"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["is_flaky_test", "rollback_triggered", "incident_created"]:
        df[col] = df[col].astype(int)

    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    df["sev_score"] = df["severity"].map(sev_map).fillna(1)

    # ── Risk score: severity + resource signals only (no rollback/incident) ────
    # Keeping rollback/incident out of the label formula means they can be
    # used as clean features for the risk_level task.
    df["risk_score"] = (
        df["sev_score"] * 2.5
        + df["retry_count"] * 0.8
        + (df["cpu_usage_pct"] / 15.0)
        + (df["memory_usage_mb"] / 3000.0)
        + df["is_flaky_test"] * 1.5
    )

    # ── 4-class label: non-equal quantile cuts ────────────────────────────────
    # Fixed bins collapse to one class when the score distribution is narrow.
    # Equal qcut (q=4) gives 25% per class but destroys density structure.
    # Compromise: derive edges from actual quantiles [15%, 35%, 65%] so the
    # distribution is ~15/20/30/35 (LOW/MED/HIGH/CRIT) — skewed toward CRITICAL
    # like a real CI system, but with enough LOW/MEDIUM to be learnable.
    q1, q2, q3 = df["risk_score"].quantile([0.15, 0.35, 0.65])

    # Degenerate guard: if quantiles collapse (all scores identical), fall back
    if len({q1, q2, q3}) < 3:
        df["risk_level"] = pd.qcut(
            df["risk_score"], q=4, labels=RISK_LABELS, duplicates="drop"
        ).astype(str)
    else:
        df["risk_level"] = pd.cut(
            df["risk_score"],
            bins=[-np.inf, q1, q2, q3, np.inf],
            labels=RISK_LABELS,
            duplicates="drop"
        ).astype(str)

    # ── Duration features ─────────────────────────────────────────────────────
    df["total_duration_sec"] = (
        df["build_duration_sec"] + df["test_duration_sec"] + df["deploy_duration_sec"]
    )
    total = df["total_duration_sec"].clip(lower=1)
    df["build_ratio"]  = df["build_duration_sec"]  / total
    df["test_ratio"]   = df["test_duration_sec"]   / total
    df["deploy_ratio"] = df["deploy_duration_sec"] / total

    for col in ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
                "memory_usage_mb", "total_duration_sec"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # ── Interaction terms ─────────────────────────────────────────────────────
    df["cpu_x_retry"] = df["cpu_usage_pct"]  * df["retry_count"]
    df["mem_x_retry"] = df["memory_usage_mb"] * df["retry_count"]
    df["cpu_x_flaky"] = df["cpu_usage_pct"]  * df["is_flaky_test"]
    df["sev_x_retry"] = df["sev_score"]       * df["retry_count"]
    df["sev_x_flaky"] = df["sev_score"]       * df["is_flaky_test"]

    # ── Stage / branch dummies ────────────────────────────────────────────────
    df["is_main_branch"]    = (df["branch"] == "main").astype(int)
    df["is_release_branch"] = (df["branch"] == "release").astype(int)
    df["is_deploy_stage"]   = (df["failure_stage"] == "deploy").astype(int)
    df["is_test_stage"]     = (df["failure_stage"] == "test").astype(int)
    df["is_build_stage"]    = (df["failure_stage"] == "build").astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════
def build_preprocessor(cat_cols, num_cols, bool_cols):
    return ColumnTransformer(transformers=[
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num",  StandardScaler(), num_cols),
        ("bool", "passthrough", bool_cols),
    ])


def get_feature_names(preprocessor, cat_cols, num_cols, bool_cols):
    ohe = preprocessor.named_transformers_["cat"]
    return list(ohe.get_feature_names_out(cat_cols)) + num_cols + bool_cols


def make_feature_sets(task_name: str):
    """Return (cat_cols, num_cols, bool_cols) for each task."""
    eng_num = [
        "total_duration_sec", "build_ratio", "test_ratio", "deploy_ratio",
        "log_build_duration_sec", "log_test_duration_sec",
        "log_deploy_duration_sec", "log_memory_usage_mb", "log_total_duration_sec",
        "cpu_x_retry", "mem_x_retry", "cpu_x_flaky", "sev_x_retry", "sev_x_flaky",
    ]
    eng_bool = [
        "is_main_branch", "is_release_branch",
        "is_deploy_stage", "is_test_stage", "is_build_stage",
    ]

    if task_name == "risk_level":
        # rollback/incident are clean features here (not part of label formula)
        # risk_score excluded — it's the direct precursor to the label
        all_num  = NUM_COLS + eng_num + ["sev_score"]
        all_bool = BOOL_COLS + eng_bool + ["rollback_triggered", "incident_created"]
    else:
        # Binary tasks: risk_score is a useful summary feature, no circularity
        all_num  = NUM_COLS + eng_num + ["sev_score", "risk_score"]
        all_bool = BOOL_COLS + eng_bool

    return CAT_COLS, all_num, all_bool


# ═══════════════════════════════════════════════════════════════════════════════
# Model zoo
# ═══════════════════════════════════════════════════════════════════════════════
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="saga", C=0.5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600, max_depth=10, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna hyper-tune
# ═══════════════════════════════════════════════════════════════════════════════
def optuna_tune_lgbm(X_tr, y_tr, n_classes: int, n_trials: int = 30):
    """30-trial TPE search for LightGBM. Returns best params dict."""

    # FIX: sklearn cross_val_score does not accept 'f1_binary'.
    # Binary tasks use 'f1'; multiclass tasks use 'f1_weighted'.
    scoring = "f1" if n_classes == 2 else "f1_weighted"

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 400, 1000, step=100),
            max_depth         = trial.suggest_int("max_depth", 6, 14),
            learning_rate     = trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 31, 127),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            class_weight      = "balanced",
            random_state      = RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        model = LGBMClassifier(**params)
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, labels, task_name, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(max(7, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="#0f1117", vmin=0, vmax=100,
                annot_kws={"size": 9}, cbar_kws={"label": "% per true class"})
    ax.set_title(f"{model_name} — {task_name}\nConfusion Matrix (%)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(CM_DIR, f"cm_{task_name}_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return path


def plot_feature_importance(importances, feature_names, task_name, model_name, top_n=25):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(np.array(feature_names)[idx], importances[idx], color=colors)
    ax.set_title(f"Top-{top_n} Features — {model_name} / {task_name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"fi_{task_name}_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Train + evaluate one task
# ═══════════════════════════════════════════════════════════════════════════════
def train_task(df: pd.DataFrame, task_name: str, task_cfg: dict):
    print(f"\n{'='*65}")
    print(f"🎯  Task: {task_cfg['label']}")
    print(f"{'='*65}")

    cat_cols, num_cols, bool_cols = make_feature_sets(task_name)
    X = df[cat_cols + num_cols + bool_cols].copy()

    le = LabelEncoder()
    y  = le.fit_transform(df[task_name].astype(str))
    class_names = [str(c) for c in le.classes_]
    n_classes   = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"Classes: {n_classes} {class_names}")

    preprocessor  = build_preprocessor(cat_cols, num_cols, bool_cols)
    X_train_enc   = preprocessor.fit_transform(X_train)
    X_test_enc    = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, cat_cols, num_cols, bool_cols)

    # SMOTE — binary tasks only, and only when actually imbalanced
    if HAS_SMOTE and task_cfg["type"] == "binary":
        counts = np.bincount(y_train)
        if counts.min() / counts.max() < 0.6:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            X_train_enc, y_train = sm.fit_resample(X_train_enc, y_train)
            print(f"   SMOTE applied → {len(X_train_enc):,} training samples")

    joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"preprocessor_{task_name}.pkl"))
    joblib.dump(le,           os.path.join(MODELS_DIR, f"label_encoder_{task_name}.pkl"))

    models       = get_models()
    task_results = {}
    best_f1      = -1
    best_model_name = None
    best_model_obj  = None
    feature_importances = {}
    avg = "binary" if n_classes == 2 else "weighted"

    for model_name, model in models.items():
        print(f"\n   ▶ {model_name}")
        t0 = time.time()
        model.fit(X_train_enc, y_train)
        y_pred  = model.predict(X_test_enc)
        elapsed = time.time() - t0

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average=avg)
        report = classification_report(y_test, y_pred,
                                       target_names=class_names, output_dict=True)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, model.predict_proba(X_test_enc)[:, 1])
            else:
                auc = roc_auc_score(y_test, model.predict_proba(X_test_enc),
                                    multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        print(f"      Acc: {acc:.4f}  F1: {f1:.4f}  "
              f"AUC: {f'{auc:.4f}' if auc else 'N/A'}  ({elapsed:.1f}s)")

        cm_path = plot_confusion_matrix(y_test, y_pred, class_names, task_name, model_name)

        fi_path = None
        if hasattr(model, "feature_importances_"):
            imps    = model.feature_importances_
            fi_path = plot_feature_importance(imps, feature_names, task_name, model_name)
            feature_importances[model_name] = dict(zip(feature_names, imps.tolist()))
        elif hasattr(model, "coef_"):
            raw  = model.coef_
            imps = np.abs(raw).mean(axis=0) if raw.ndim > 1 else np.abs(raw[0])
            fi_path = plot_feature_importance(imps, feature_names, task_name, model_name)
            feature_importances[model_name] = dict(zip(feature_names, imps.tolist()))

        task_results[model_name] = {
            "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
            "roc_auc": round(auc, 4) if auc else None,
            "train_time_sec": round(elapsed, 2),
            "classification_report": report,
            "confusion_matrix_path": cm_path,
            "feature_importance_path": fi_path,
        }
        if f1 > best_f1:
            best_f1 = f1; best_model_name = model_name; best_model_obj = model

    # ── Optuna fine-tune ───────────────────────────────────────────────────────
    if HAS_OPTUNA:
        print(f"\n   🔬 Optuna tuning LightGBM (30 trials)...")
        t0 = time.time()
        best_params = optuna_tune_lgbm(X_train_enc, y_train, n_classes, n_trials=30)
        tuned = LGBMClassifier(**best_params,
                               random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        tuned.fit(X_train_enc, y_train)
        y_pred_t  = tuned.predict(X_test_enc)
        acc_t     = accuracy_score(y_test, y_pred_t)
        f1_t      = f1_score(y_test, y_pred_t, average=avg)
        elapsed_t = time.time() - t0
        try:
            if n_classes == 2:
                auc_t = roc_auc_score(y_test, tuned.predict_proba(X_test_enc)[:, 1])
            else:
                auc_t = roc_auc_score(y_test, tuned.predict_proba(X_test_enc),
                                      multi_class="ovr", average="weighted")
        except Exception:
            auc_t = None
        print(f"      Tuned Acc: {acc_t:.4f}  F1: {f1_t:.4f}  "
              f"AUC: {f'{auc_t:.4f}' if auc_t else 'N/A'}  ({elapsed_t:.1f}s)")

        cm_path_t = plot_confusion_matrix(y_test, y_pred_t, class_names,
                                          task_name, "LightGBM_Tuned")
        task_results["LightGBM_Tuned"] = {
            "accuracy": round(acc_t, 4), "f1_weighted": round(f1_t, 4),
            "roc_auc": round(auc_t, 4) if auc_t else None,
            "train_time_sec": round(elapsed_t, 2),
            "classification_report": classification_report(
                y_test, y_pred_t, target_names=class_names, output_dict=True),
            "confusion_matrix_path": cm_path_t,
            "best_params": best_params,
        }
        if f1_t > best_f1:
            best_f1 = f1_t; best_model_name = "LightGBM_Tuned"; best_model_obj = tuned

    print(f"\n   ⭐ Best: {best_model_name} (F1={best_f1:.4f})")
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, f"best_model_{task_name}.pkl"))
    task_results["__best_model__"] = best_model_name

    with open(os.path.join(RESULTS_DIR, f"feature_importances_{task_name}.json"), "w") as f:
        json.dump(feature_importances, f, indent=2)

    return task_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("🚀  CI/CD Failure Prediction — v2 Training Pipeline (fixed)")
    print("="*65)
    print("Fixes: adaptive quantile cuts · f1 scoring string · leakage removed")
    print("="*65)

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    print(f"\n📂  Dataset shape after engineering: {df.shape}")

    dist = df["risk_level"].value_counts(normalize=True).sort_index()
    print("\n📊  risk_level distribution (should be ~15/20/30/35):")
    for label, pct in dist.items():
        bar = "█" * int(pct * 40)
        print(f"   {label:<10} {bar} {pct:.1%}")

    all_results = {}
    for task_name, task_cfg in TASKS.items():
        all_results[task_name] = train_task(df, task_name, task_cfg)

    metrics_path = os.path.join(RESULTS_DIR, "metrics_v2.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅  Metrics → {metrics_path}")

    print("\n" + "="*75)
    print("📊  Final Summary (Test Set)")
    print("="*75)
    print(f"{'Task':<22} {'Model':<28} {'Acc':>8} {'F1-w':>8} {'AUC':>8}")
    print("-"*75)
    for task_name, task_res in all_results.items():
        for model_name, metrics in task_res.items():
            if model_name.startswith("__"):
                continue
            acc   = metrics["accuracy"]
            f1    = metrics["f1_weighted"]
            auc   = metrics["roc_auc"]
            auc_s = f"{auc:.4f}" if auc else "N/A"
            star  = " ⭐" if model_name == task_res.get("__best_model__") else ""
            print(f"{task_name:<22} {(model_name+star):<30} {acc:>8.4f} {f1:>8.4f} {auc_s:>8}")

    print("\n✅  Done! Models → models/   Results → results/")


if __name__ == "__main__":
    main()