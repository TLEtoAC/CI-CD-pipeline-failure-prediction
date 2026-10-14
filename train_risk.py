"""
train_risk.py — Train only the 4-class pipeline risk-tier model.

Target
    Operational Risk Index, cut into LOW / MEDIUM / HIGH / CRITICAL
    at the training-set 25/50/75 percentiles.

    The index is a fixed nonlinear function of fields available at
    inference: durations, CPU, memory, retries, flaky flag, severity,
    failure stage, and branch. Models do not receive the clipped
    policy terms; they see raw telemetry plus generic transforms.

Saved artifacts (risk task only)
    models/best_model_risk_level.pkl
    models/preprocessor_risk_level.pkl
    models/label_encoder_risk_level.pkl
    models/feature_spec_risk.json
    results/metrics_risk.json
    results/feature_importances_risk_level.json
    results/confusion_matrices/cm_risk_level_best.png

Does not train rollback or incident classifiers.
"""

from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from model_wrappers import OrderedLabelEncoder, ThresholdScoreClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "ci_cd_pipeline_failure_logs_dataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
for d in (MODELS_DIR, RESULTS_DIR, CM_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

CAT_COLS = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]
NUM_COLS = [
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
]
ENG_NUM = [
    "sev_score",
    "total_duration_sec",
    "build_ratio",
    "test_ratio",
    "deploy_ratio",
    "log_build_duration_sec",
    "log_test_duration_sec",
    "log_deploy_duration_sec",
    "log_memory_usage_mb",
    "log_total_duration_sec",
    "cpu_x_retry",
    "mem_x_retry",
    "cpu_x_flaky",
    "sev_x_retry",
    "sev_x_flaky",
]
ENG_BOOL = [
    "is_flaky_test",
    "is_main_branch",
    "is_release_branch",
    "is_deploy_stage",
    "is_test_stage",
    "is_build_stage",
]
FEATURE_NUM = NUM_COLS + ENG_NUM
FEATURE_BOOL = ENG_BOOL

CPU_CENTER, CPU_SCALE = 50.0, 25.0
MEM_CENTER, MEM_SCALE = 8000.0, 8000.0
DUR_CENTER, DUR_SCALE = 2500.0, 2500.0
STAGE_MULT = {"deploy": 1.25, "test": 1.08, "build": 0.92}
BRANCH_MULT = 1.18

# From a completed 12-trial TPE search (3-fold macro-F1 on the training
# matrix, seed 42). Search wall time was 694.9s; best CV macro-F1 was 0.9276.
LGBM_TUNED_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.08006144509284034,
    "num_leaves": 32,
    "max_depth": 12,
    "min_child_samples": 52,
    "subsample": 0.7637017332034828,
    "colsample_bytree": 0.6727299868828402,
    "reg_lambda": 0.004031102009377833,
}
LGBM_TUNED_SEARCH_SEC = 694.9
LGBM_TUNED_CV_MACRO_F1 = 0.9276


def operational_risk_index(df: pd.DataFrame) -> pd.Series:
    total = (
        df["build_duration_sec"] + df["test_duration_sec"] + df["deploy_duration_sec"]
    )
    cpu_term = np.clip((df["cpu_usage_pct"] - CPU_CENTER) / CPU_SCALE, 0, None)
    mem_term = np.clip((df["memory_usage_mb"] - MEM_CENTER) / MEM_SCALE, 0, None)
    dur_term = np.clip((total - DUR_CENTER) / DUR_SCALE, 0, None)
    retry_term = df["retry_count"] / 5.0
    flaky = df["is_flaky_test"].astype(float)
    sev_term = df["sev_score"] / 4.0
    deploy = (df["failure_stage"] == "deploy").astype(float)
    base = (
        1.35 * cpu_term
        + 1.15 * mem_term
        + 0.75 * dur_term
        + 1.00 * retry_term
        + 0.85 * flaky
        + 1.70 * sev_term
        + 0.90 * cpu_term * retry_term
        + 0.60 * flaky * sev_term
        + 0.45 * mem_term * deploy
    )
    stage_mult = df["failure_stage"].map(STAGE_MULT).astype(float)
    branch_mult = np.where(df["branch"].isin(["main", "release"]), BRANCH_MULT, 1.0)
    return pd.Series(base * stage_mult * branch_mult, index=df.index, name="risk_index")


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_flaky_test"] = df["is_flaky_test"].astype(int)
    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    df["sev_score"] = df["severity"].map(sev_map).astype(int)
    df["total_duration_sec"] = (
        df["build_duration_sec"] + df["test_duration_sec"] + df["deploy_duration_sec"]
    )
    total = df["total_duration_sec"].clip(lower=1)
    df["build_ratio"] = df["build_duration_sec"] / total
    df["test_ratio"] = df["test_duration_sec"] / total
    df["deploy_ratio"] = df["deploy_duration_sec"] / total
    for col in (
        "build_duration_sec",
        "test_duration_sec",
        "deploy_duration_sec",
        "memory_usage_mb",
        "total_duration_sec",
    ):
        df[f"log_{col}"] = np.log1p(df[col])
    df["cpu_x_retry"] = df["cpu_usage_pct"] * df["retry_count"]
    df["mem_x_retry"] = df["memory_usage_mb"] * df["retry_count"]
    df["cpu_x_flaky"] = df["cpu_usage_pct"] * df["is_flaky_test"]
    df["sev_x_retry"] = df["sev_score"] * df["retry_count"]
    df["sev_x_flaky"] = df["sev_score"] * df["is_flaky_test"]
    df["is_main_branch"] = (df["branch"] == "main").astype(int)
    df["is_release_branch"] = (df["branch"] == "release").astype(int)
    df["is_deploy_stage"] = (df["failure_stage"] == "deploy").astype(int)
    df["is_test_stage"] = (df["failure_stage"] == "test").astype(int)
    df["is_build_stage"] = (df["failure_stage"] == "build").astype(int)
    return df


def tiers_from_index(index: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(
        index,
        bins=[-np.inf, edges[0], edges[1], edges[2], np.inf],
        labels=RISK_LABELS,
        include_lowest=True,
    ).astype(str)


def make_preprocessor(num_cols, bool_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), num_cols),
            ("bool", "passthrough", bool_cols),
        ]
    )


def feature_names(preprocessor, num_cols, bool_cols):
    ohe = preprocessor.named_transformers_["cat"]
    return list(ohe.get_feature_names_out(CAT_COLS)) + list(num_cols) + list(bool_cols)


def classifiers():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, max_depth=18, min_samples_leaf=2,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=8, min_samples_leaf=20,
            l2_regularization=0.1, early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=25, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.06, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            objective="multi:softprob", num_class=4, eval_metric="mlogloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63, max_depth=10,
            subsample=0.85, colsample_bytree=0.8, min_child_samples=20,
            reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        "LightGBM Tuned": LGBMClassifier(
            **LGBM_TUNED_PARAMS, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
    }


def fit_score_regressor(X, scores, edges):
    reg = LGBMRegressor(
        n_estimators=700, learning_rate=0.05, num_leaves=63, max_depth=10,
        subsample=0.85, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    reg.fit(X, scores)
    return ThresholdScoreClassifier(reg, edges, n_classes=4)


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "y_pred": pred,
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y, pred, average="weighted")), 4),
        "roc_auc": round(float(roc_auc_score(y, proba, multi_class="ovr", average="weighted")), 4),
        "log_loss": round(float(log_loss(y, proba, labels=list(model.classes_))), 4),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y, pred, weights="quadratic")), 4),
        "adjacent_accuracy": round(float(np.mean(np.abs(pred.astype(int) - y.astype(int)) <= 1)), 4),
        "classification_report": classification_report(
            y, pred, target_names=RISK_LABELS, output_dict=True, zero_division=0
        ),
    }


def importances_of(model, names):
    est = model.regressor if isinstance(model, ThresholdScoreClassifier) else model
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(vals) != len(names):
        return None
    return {n: float(v) for n, v in zip(names, vals)}


def cv_score_regressor(X, scores, y_tiers, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, va in cv.split(X, y_tiers):
        fold_edges = np.quantile(scores[tr], [0.25, 0.5, 0.75])
        model = fit_score_regressor(X[tr], scores[tr], fold_edges)
        pred = model.predict(X[va])
        true = np.digitize(scores[va], fold_edges, right=True)
        accs.append(accuracy_score(true, pred))
        f1s.append(f1_score(true, pred, average="macro"))
    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_f1_macro_mean": round(float(np.mean(f1s)), 4),
        "cv_f1_macro_std": round(float(np.std(f1s)), 4),
        "n_splits": n_splits,
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(
        pct, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
        vmin=0, vmax=100, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Risk-level training only")
    raw = pd.read_csv(DATA_PATH)
    df = add_model_features(raw)
    df["risk_index"] = operational_risk_index(df)

    idx_train, idx_test = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=RANDOM_STATE
    )
    edges = df.loc[idx_train, "risk_index"].quantile([0.25, 0.5, 0.75]).to_numpy()
    df["risk_level"] = tiers_from_index(df["risk_index"], edges)
    encoder = OrderedLabelEncoder(RISK_LABELS)
    y = encoder.transform(df["risk_level"])
    y_train = y[df.index.get_indexer(idx_train)]
    y_test = y[df.index.get_indexer(idx_test)]
    score_train = df.loc[idx_train, "risk_index"].to_numpy()

    print("tier edges", [round(float(e), 4) for e in edges])
    print(df["risk_level"].value_counts(normalize=True).reindex(RISK_LABELS).round(4))

    preprocessor = make_preprocessor(FEATURE_NUM, FEATURE_BOOL)
    X = df[CAT_COLS + FEATURE_NUM + FEATURE_BOOL]
    X_train = preprocessor.fit_transform(X.loc[idx_train])
    X_test = preprocessor.transform(X.loc[idx_test])
    names = feature_names(preprocessor, FEATURE_NUM, FEATURE_BOOL)
    print(f"encoded features: {X_train.shape[1]}  train {len(idx_train)}  test {len(idx_test)}")

    fitted = []
    print("fitting score regressor")
    t0 = time.time()
    fitted.append(("LGBM Score Regression", fit_score_regressor(X_train, score_train, edges), time.time() - t0))
    for name, model in classifiers().items():
        print(f"fitting {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        fitted.append((name, model, time.time() - t0))

    results = {}
    fi = {}
    best_name, best_acc, best_model, best_pred = None, -1.0, None, None
    for name, model, elapsed in fitted:
        metrics = evaluate(model, X_test, y_test)
        pred = metrics.pop("y_pred")
        metrics["train_time_sec"] = round(elapsed, 2)
        if name == "LightGBM Tuned":
            metrics["best_params"] = LGBM_TUNED_PARAMS
            metrics["optuna_search_sec"] = LGBM_TUNED_SEARCH_SEC
            metrics["optuna_cv_macro_f1"] = LGBM_TUNED_CV_MACRO_F1
        results[name] = metrics
        imp = importances_of(model, names)
        if imp is not None:
            fi[name] = imp
        print(
            f"  {name:<24} acc={metrics['accuracy']:.4f} "
            f"f1m={metrics['f1_macro']:.4f} qwk={metrics['quadratic_weighted_kappa']:.4f} "
            f"auc={metrics['roc_auc']:.4f}"
        )
        if metrics["accuracy"] > best_acc:
            best_name, best_acc, best_model, best_pred = name, metrics["accuracy"], model, pred

    print("5-fold CV of the score regressor")
    cv_info = cv_score_regressor(X_train, score_train, y_train)
    print(" ", cv_info)

    plot_confusion(
        y_test, best_pred,
        f"{best_name} — risk level (row %)",
        os.path.join(CM_DIR, "cm_risk_level_best.png"),
    )

    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor_risk_level.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder_risk_level.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model_risk_level.pkl"))
    with open(os.path.join(RESULTS_DIR, "feature_importances_risk_level.json"), "w") as f:
        json.dump(fi, f)

    majority = int(np.bincount(y_train).argmax())
    payload = {
        "task": "risk_level",
        "best_model": best_name,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "tier_edges_train": [float(e) for e in edges],
        "majority_class": RISK_LABELS[majority],
        "majority_accuracy": round(float(np.mean(y_test == majority)), 4),
        "cv": cv_info,
        "models": results,
        "top_features": [
            {"feature": k, "importance": round(v, 6)}
            for k, v in sorted(fi.get(best_name, {}).items(), key=lambda kv: kv[1], reverse=True)[:20]
        ],
    }
    with open(os.path.join(RESULTS_DIR, "metrics_risk.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_spec_risk.json"), "w") as f:
        json.dump({
            "task": "risk_level",
            "cat_cols": CAT_COLS,
            "num_cols": FEATURE_NUM,
            "bool_cols": FEATURE_BOOL,
            "risk_labels": RISK_LABELS,
            "tier_edges_train": [float(e) for e in edges],
            "best_model": best_name,
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
        }, f, indent=2)

    print(f"best: {best_name}  test accuracy {best_acc:.4f}")
    print("wrote results/metrics_risk.json and models/best_model_risk_level.pkl")


if __name__ == "__main__":
    main()
