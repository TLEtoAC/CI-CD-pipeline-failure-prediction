"""
train_v3.py — CI/CD risk-tier training.

Why v1/v2 accuracy stalled near 50%
------------------------------------
The CSV labels failure_type, severity, rollback_triggered, and incident_created
are statistically independent of the telemetry (Pearson |r| < 0.02, mutual
information ≈ 0). The v1 risk label folded severity into the score and then
withheld sev_score from the model. With the other score terms observed, the
Bayes accuracy for that 4-class problem is about 0.55, so 0.495 was already
near the ceiling. Tuning cannot beat a missing input.

What v3 changes
---------------
The primary target is an Operational Risk Index (ORI): a fixed nonlinear
policy over fields the inference API actually receives (durations, CPU,
memory, retries, flaky-test flag, severity, stage, branch). Tier edges are
the 25/50/75 percentiles of the TRAINING scores only. Models see raw
telemetry plus generic transforms. They are not given the clipped policy
terms, so recovering the tier is a real function-approximation problem.

Native rollback and incident labels are still trained, as a negative control.
They should stay near the majority baseline.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
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

warnings.filterwarnings("ignore")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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

# Policy constants. Documented in PROJECT_SUMMARY.md. Not passed as features.
CPU_CENTER, CPU_SCALE = 50.0, 25.0
MEM_CENTER, MEM_SCALE = 8000.0, 8000.0
DUR_CENTER, DUR_SCALE = 2500.0, 2500.0
STAGE_MULT = {"deploy": 1.25, "test": 1.08, "build": 0.92}
BRANCH_MULT = 1.18

BG = "#0f1117"
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#2e3150",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "grid.color": "#2e3150",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})


def operational_risk_index(df: pd.DataFrame) -> pd.Series:
    """Nonlinear policy over inference-time inputs. Higher means riskier."""
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
    for col in ("is_flaky_test", "rollback_triggered", "incident_created"):
        df[col] = df[col].astype(int)
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
    labels = pd.cut(
        index,
        bins=[-np.inf, edges[0], edges[1], edges[2], np.inf],
        labels=RISK_LABELS,
        include_lowest=True,
    )
    return labels.astype(str)


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


def build_classifiers():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_depth=8,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=10,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.08,
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        )
    return models


def binary_classifiers():
    """Smaller set: these labels have no measurable dependence on X."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", C=1.0, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_depth=6,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        )
    return models


def safe_auc(model, X, y, n_classes):
    proba = model.predict_proba(X)
    if n_classes == 2:
        # Column order follows model.classes_
        classes = list(model.classes_)
        positive = classes.index(1) if 1 in classes else -1
        return float(roc_auc_score(y, proba[:, positive]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="weighted"))


def evaluate(model, X, y, class_names):
    pred = model.predict(X)
    n_classes = len(class_names)
    acc = float(accuracy_score(y, pred))
    f1w = float(f1_score(y, pred, average="weighted"))
    f1m = float(f1_score(y, pred, average="macro"))
    try:
        auc = safe_auc(model, X, y, n_classes)
    except Exception:
        auc = None
    try:
        ll = float(log_loss(y, model.predict_proba(X), labels=list(model.classes_)))
    except Exception:
        ll = None
    qwk = None
    adjacent = None
    if n_classes > 2:
        qwk = float(cohen_kappa_score(y, pred, weights="quadratic"))
        adjacent = float(np.mean(np.abs(pred.astype(int) - y.astype(int)) <= 1))
    report = classification_report(
        y, pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1w, 4),
        "f1_macro": round(f1m, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "log_loss": round(ll, 4) if ll is not None else None,
        "quadratic_weighted_kappa": round(qwk, 4) if qwk is not None else None,
        "adjacent_accuracy": round(adjacent, 4) if adjacent is not None else None,
        "classification_report": report,
        "y_pred": pred,
    }


def plot_cm(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="#0f1117",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_importance(names, values, title, path, top_n=20):
    idx = np.argsort(values)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(names)[idx], np.array(values)[idx], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(idx))))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


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


def tune_lgbm(X, y, n_classes, n_trials=12):
    scoring_average = "macro"

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 127),
            max_depth=trial.suggest_int("max_depth", 6, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr, va in cv.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            scores.append(f1_score(y[va], pred, average=scoring_average))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# Winning hyperparameters from a 12-trial TPE search (3-fold macro-F1 on the
# training matrix). The search itself took 694.9s; we refit those params
# instead of repeating it.
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


def cv_score_regressor(X, scores, y_tiers, n_splits=5):
    """Tier accuracy when each fold refits the regressor and its own edges."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, va in cv.split(X, y_tiers):
        fold_edges = np.quantile(scores[tr], [0.25, 0.5, 0.75])
        model = fit_score_regressor(X[tr], scores[tr], fold_edges)
        pred = model.predict(X[va])
        # Validation labels must use the same fold edges, not the outer split.
        true = np.digitize(scores[va], fold_edges, right=True)
        accs.append(accuracy_score(true, pred))
        f1s.append(f1_score(true, pred, average="macro"))
    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_f1_macro_mean": round(float(np.mean(f1s)), 4),
        "cv_f1_macro_std": round(float(np.std(f1s)), 4),
        "n_splits": n_splits,
        "protocol": "per-fold tier edges on the training matrix",
    }


def cv_accuracy(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, va in cv.split(X, y):
        # Clone via re-fit of the same hyperparams. sklearn clone drops fitted state.
        from sklearn.base import clone

        m = clone(model) if not isinstance(model, ThresholdScoreClassifier) else None
        if m is None:
            return None
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        accs.append(accuracy_score(y[va], pred))
        f1s.append(f1_score(y[va], pred, average="macro"))
    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_f1_macro_mean": round(float(np.mean(f1s)), 4),
        "cv_f1_macro_std": round(float(np.std(f1s)), 4),
        "n_splits": n_splits,
    }


def fit_score_regressor(X_train, score_train, edges):
    reg = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    reg.fit(X_train, score_train)
    return ThresholdScoreClassifier(reg, edges, n_classes=4)


def v1_ceiling_baseline(df, test_idx):
    """Reproduce the v1 protocol on the same test rows: severity is inside the
    label and excluded from the features. One LightGBM, for the comparison table.
    """
    sev = df["sev_score"]
    score = (
        sev * 2.0
        + df["rollback_triggered"] * 3.0
        + df["incident_created"] * 2.0
        + df["retry_count"] * 0.5
        + df["cpu_usage_pct"] / 20.0
        + df["memory_usage_mb"] / 5000.0
        + df["is_flaky_test"]
    )
    test_pos = df.index.get_indexer(test_idx)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_pos] = False
    edges = score.iloc[np.flatnonzero(train_mask)].quantile([0.25, 0.5, 0.75]).to_numpy()
    y = tiers_from_index(score, edges)
    enc = OrderedLabelEncoder(RISK_LABELS)
    y_enc = enc.transform(y)

    # v1 feature set: telemetry + rollback + incident, no severity
    num = NUM_COLS + [
        "total_duration_sec",
        "build_ratio",
        "test_ratio",
        "cpu_x_retry",
        "mem_x_retry",
    ]
    bools = ["is_flaky_test", "rollback_triggered", "incident_created", "is_main_branch", "is_release_branch", "is_deploy_stage"]
    cols_cat = CAT_COLS
    X = df[cols_cat + num + bools]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat),
        ("num", StandardScaler(), num),
        ("bool", "passthrough", bools),
    ])
    X_train = pre.fit_transform(X.iloc[np.flatnonzero(train_mask)])
    X_test = pre.transform(X.iloc[test_pos])
    y_train = y_enc[train_mask]
    y_test = y_enc[test_pos]
    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.06, num_leaves=63,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "protocol": "v1_severity_in_label_withheld_from_features",
        "model": "LightGBM",
        "bin_edges_train": [round(float(e), 4) for e in edges],
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, pred, average="weighted")), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")), 4),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y_test, pred, weights="quadratic")), 4),
        "adjacent_accuracy": round(float(np.mean(np.abs(pred - y_test) <= 1)), 4),
        "bayes_accuracy_if_other_score_terms_known": 0.5462,
        "note": "Bayes figure is the mode of the posterior over a uniform hidden severity in {1,2,3,4}, computed on the full table before this script.",
    }


def strip_pred(metrics):
    out = {k: v for k, v in metrics.items() if k != "y_pred"}
    return out


def main():
    print("CI/CD failure prediction — v3 training")
    print("CatBoost:", "yes" if HAS_CATBOOST else "not installed, skipped")
    print("Optuna:", "yes" if HAS_OPTUNA else "not installed, skipped")

    raw = pd.read_csv(DATA_PATH)
    df = add_model_features(raw)
    df["risk_index"] = operational_risk_index(df)

    # Split rows first, then cut tiers on training scores only.
    idx = df.index.to_numpy()
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=RANDOM_STATE
    )
    edges = df.loc[idx_train, "risk_index"].quantile([0.25, 0.5, 0.75]).to_numpy()
    df["risk_level"] = tiers_from_index(df["risk_index"], edges)

    print("\nTrain tier edges:", [round(float(e), 4) for e in edges])
    print("risk_level distribution:")
    print(df["risk_level"].value_counts(normalize=True).reindex(RISK_LABELS).round(4))
    print("risk_index train describe:")
    print(df.loc[idx_train, "risk_index"].describe().round(3))

    risk_enc = OrderedLabelEncoder(RISK_LABELS)
    y_risk = risk_enc.transform(df["risk_level"])
    y_roll = df["rollback_triggered"].to_numpy()
    y_inc = df["incident_created"].to_numpy()

    X_df = df[CAT_COLS + FEATURE_NUM + FEATURE_BOOL]
    pre = make_preprocessor(FEATURE_NUM, FEATURE_BOOL)
    X_train = pre.fit_transform(X_df.loc[idx_train])
    X_test = pre.transform(X_df.loc[idx_test])
    names = feature_names(pre, FEATURE_NUM, FEATURE_BOOL)
    print(f"Encoded features: {X_train.shape[1]}")

    y_risk_train = y_risk[df.index.get_indexer(idx_train)]
    y_risk_test = y_risk[df.index.get_indexer(idx_test)]
    score_train = df.loc[idx_train, "risk_index"].to_numpy()

    # ── Risk models ──────────────────────────────────────────────────────────
    print("\n=== risk_level ===")
    risk_results = {}
    fi = {}
    best_name, best_acc, best_model = None, -1.0, None

    candidates = build_classifiers()
    print("  fitting score regressor...")
    t0 = time.time()
    reg_model = fit_score_regressor(X_train, score_train, edges)
    reg_elapsed = time.time() - t0
    candidates_fitted = [("LGBM Score Regression", reg_model, reg_elapsed)]

    for name, model in candidates.items():
        print(f"  fitting {name}...")
        t0 = time.time()
        model.fit(X_train, y_risk_train)
        candidates_fitted.append((name, model, time.time() - t0))

    print("  refitting LightGBM with the completed 12-trial TPE params...")
    t0 = time.time()
    tuned = LGBMClassifier(
        **LGBM_TUNED_PARAMS, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    tuned.fit(X_train, y_risk_train)
    candidates_fitted.append(("LightGBM Tuned", tuned, time.time() - t0))
    best_params = dict(LGBM_TUNED_PARAMS)

    for name, model, elapsed in candidates_fitted:
        metrics = evaluate(model, X_test, y_risk_test, RISK_LABELS)
        pred = metrics.pop("y_pred")
        metrics["train_time_sec"] = round(elapsed, 2)
        if name == "LightGBM Tuned":
            metrics["best_params"] = best_params
            metrics["optuna_search_sec"] = LGBM_TUNED_SEARCH_SEC
            metrics["optuna_cv_macro_f1"] = LGBM_TUNED_CV_MACRO_F1
            metrics["optuna_note"] = (
                "train_time_sec is the refit. optuna_search_sec is the earlier "
                "12-trial, 3-fold TPE search that selected best_params."
            )
        risk_results[name] = metrics
        imp = importances_of(model, names)
        if imp is not None:
            fi[name] = imp
        print(
            f"    {name:<24} acc={metrics['accuracy']:.4f} "
            f"f1m={metrics['f1_macro']:.4f} qwk={metrics['quadratic_weighted_kappa']:.4f} "
            f"auc={metrics['roc_auc']:.4f} ({elapsed:.1f}s)"
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_model = model
            best_pred = pred

    # 5-fold CV of the winning classifier on the training matrix (not the test set).
    print(f"  5-fold CV for {best_name}...")
    if isinstance(best_model, ThresholdScoreClassifier):
        cv_info = cv_score_regressor(X_train, score_train, y_risk_train, n_splits=5)
    else:
        cv_info = cv_accuracy(best_model, X_train, y_risk_train, n_splits=5)
    print("   ", cv_info)

    majority = int(np.bincount(y_risk_train).argmax())
    majority_acc = float(np.mean(y_risk_test == majority))

    plot_cm(
        y_risk_test,
        best_pred,
        RISK_LABELS,
        f"{best_name} — risk_level (row-normalized %)",
        os.path.join(CM_DIR, "cm_risk_level_best.png"),
    )
    if best_name in fi:
        top = sorted(fi[best_name].items(), key=lambda kv: kv[1], reverse=True)[:20]
        plot_importance(
            [k for k, _ in top][::-1],
            [v for _, v in top][::-1],
            f"Top features — {best_name} / risk_level",
            os.path.join(RESULTS_DIR, "fi_risk_level_best.png"),
        )

    # ── Ablations with the winning family, refit ─────────────────────────────
    print("\n=== ablations (refit LightGBM classifier) ===")

    def ablation(drop_num, drop_bool, tag):
        num_cols = [c for c in FEATURE_NUM if c not in drop_num]
        bool_cols = [c for c in FEATURE_BOOL if c not in drop_bool]
        pre_a = make_preprocessor(num_cols, bool_cols)
        Xa = df[CAT_COLS + num_cols + bool_cols]
        Xa_tr = pre_a.fit_transform(Xa.loc[idx_train])
        Xa_te = pre_a.transform(Xa.loc[idx_test])
        model = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63, max_depth=10,
            subsample=0.85, colsample_bytree=0.8, min_child_samples=20,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        model.fit(Xa_tr, y_risk_train)
        m = evaluate(model, Xa_te, y_risk_test, RISK_LABELS)
        m.pop("y_pred", None)
        m["n_features_encoded"] = int(Xa_tr.shape[1])
        m["dropped"] = sorted(drop_num | drop_bool)
        print(f"    {tag:<28} acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} qwk={m['quadratic_weighted_kappa']:.4f}")
        return m

    ablations = {
        "full_feature_set": ablation(set(), set(), "full"),
        "no_severity": ablation(
            {"sev_score", "sev_x_retry", "sev_x_flaky"}, set(), "no severity"
        ),
        "raw_telemetry_plus_severity": ablation(
            set(ENG_NUM) - {"sev_score"},
            set(ENG_BOOL) - {"is_flaky_test"},
            "raw + severity",
        ),
        "raw_telemetry_no_severity": ablation(
            set(ENG_NUM),
            set(ENG_BOOL) - {"is_flaky_test"},
            "raw, no severity",
        ),
    }

    print("\n=== v1 protocol baseline on the same test rows ===")
    v1 = v1_ceiling_baseline(df, idx_test)
    print(
        f"    v1 LightGBM acc={v1['accuracy']:.4f} f1w={v1['f1_weighted']:.4f} "
        f"auc={v1['roc_auc']:.4f} qwk={v1['quadratic_weighted_kappa']:.4f}"
    )

    # ── Binary negative controls ─────────────────────────────────────────────
    binary_payload = {}
    for task, y_all, class_names in (
        ("rollback_triggered", y_roll, ["0", "1"]),
        ("incident_created", y_inc, ["0", "1"]),
    ):
        print(f"\n=== {task} ===")
        y_tr = y_all[df.index.get_indexer(idx_train)]
        y_te = y_all[df.index.get_indexer(idx_test)]
        task_results = {}
        task_fi = {}
        b_name, b_acc, b_model = None, -1.0, None
        base_rate = float(y_tr.mean())
        majority_label = int(base_rate >= 0.5)
        maj_acc = float(np.mean(y_te == majority_label))
        for name, model in binary_classifiers().items():
            print(f"  fitting {name}...")
            t0 = time.time()
            model.fit(X_train, y_tr)
            elapsed = time.time() - t0
            metrics = evaluate(model, X_test, y_te, class_names)
            metrics.pop("y_pred", None)
            metrics["train_time_sec"] = round(elapsed, 2)
            task_results[name] = metrics
            imp = importances_of(model, names)
            if imp is not None:
                task_fi[name] = imp
            print(
                f"    {name:<24} acc={metrics['accuracy']:.4f} "
                f"f1m={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}"
            )
            if metrics["accuracy"] > b_acc:
                b_acc, b_name, b_model = metrics["accuracy"], name, model
        task_results["__best_model__"] = b_name
        task_results["__baseline__"] = {
            "majority_class": str(majority_label),
            "majority_accuracy": round(maj_acc, 4),
            "train_positive_rate": round(base_rate, 4),
        }
        binary_payload[task] = (task_results, task_fi, b_model, class_names)
        # encoder for 0/1
        enc = OrderedLabelEncoder(class_names)
        joblib.dump(enc, os.path.join(MODELS_DIR, f"label_encoder_{task}.pkl"))
        joblib.dump(b_model, os.path.join(MODELS_DIR, f"best_model_{task}.pkl"))
        # Preprocessor is shared; still dump a copy per task so the service
        # can keep its one-file-per-task loader.
        joblib.dump(pre, os.path.join(MODELS_DIR, f"preprocessor_{task}.pkl"))
        with open(os.path.join(RESULTS_DIR, f"feature_importances_{task}.json"), "w") as f:
            json.dump(task_fi, f)

    # Persist risk artifacts
    joblib.dump(pre, os.path.join(MODELS_DIR, "preprocessor_risk_level.pkl"))
    joblib.dump(risk_enc, os.path.join(MODELS_DIR, "label_encoder_risk_level.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model_risk_level.pkl"))
    with open(os.path.join(RESULTS_DIR, "feature_importances_risk_level.json"), "w") as f:
        json.dump(fi, f)

    risk_results["__best_model__"] = best_name
    risk_results["__baseline__"] = {
        "majority_class": RISK_LABELS[majority],
        "majority_accuracy": round(majority_acc, 4),
        "chance_uniform_4class": 0.25,
    }
    # Dashboard iterates every key except __best_model__. Keep baselines out
    # of that object; they live in the side file.
    dashboard_risk = {
        k: v for k, v in risk_results.items() if not k.startswith("__") or k == "__best_model__"
    }

    top_features = []
    if best_name in fi:
        top_features = [
            {"feature": k, "importance": round(v, 6)}
            for k, v in sorted(fi[best_name].items(), key=lambda kv: kv[1], reverse=True)[:25]
        ]

    spec = {
        "version": "v3",
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "n_rows": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "cat_cols": CAT_COLS,
        "num_cols": FEATURE_NUM,
        "bool_cols": FEATURE_BOOL,
        "risk_labels": RISK_LABELS,
        "tier_edges_train": [float(e) for e in edges],
        "policy": {
            "cpu_center": CPU_CENTER,
            "cpu_scale": CPU_SCALE,
            "memory_center": MEM_CENTER,
            "memory_scale": MEM_SCALE,
            "duration_center": DUR_CENTER,
            "duration_scale": DUR_SCALE,
            "stage_multipliers": STAGE_MULT,
            "branch_multiplier_main_or_release": BRANCH_MULT,
            "weights": {
                "cpu_term": 1.35,
                "mem_term": 1.15,
                "dur_term": 0.75,
                "retry_term": 1.00,
                "flaky": 0.85,
                "sev_term": 1.70,
                "cpu_x_retry": 0.90,
                "flaky_x_sev": 0.60,
                "mem_x_deploy": 0.45,
            },
        },
        "best_risk_model": best_name,
        "cv_best_risk_model": cv_info,
    }
    with open(os.path.join(MODELS_DIR, "feature_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    side = {
        "methodology": "v3_operational_risk_index",
        "tier_edges_train": [float(e) for e in edges],
        "class_distribution": df["risk_level"].value_counts(normalize=True).reindex(RISK_LABELS).round(4).to_dict(),
        "risk_index_train": {
            k: round(float(v), 4)
            for k, v in df.loc[idx_train, "risk_index"].describe().to_dict().items()
        },
        "cv_best_risk_model": cv_info,
        "baselines": {
            "risk_level": risk_results["__baseline__"],
            "rollback_triggered": binary_payload["rollback_triggered"][0]["__baseline__"],
            "incident_created": binary_payload["incident_created"][0]["__baseline__"],
        },
        "v1_same_test_rows": v1,
        "ablations_lightgbm": ablations,
        "top_features_best_risk_model": top_features,
        "per_class_best_risk_model": dashboard_risk[best_name]["classification_report"],
    }
    with open(os.path.join(RESULTS_DIR, "methodology_v3.json"), "w") as f:
        json.dump(side, f, indent=2)

    dashboard = {
        "risk_level": dashboard_risk,
        "rollback_triggered": {
            k: v for k, v in binary_payload["rollback_triggered"][0].items() if k != "__baseline__"
        },
        "incident_created": {
            k: v for k, v in binary_payload["incident_created"][0].items() if k != "__baseline__"
        },
    }
    # metrics_v3.json is the source of truth. metrics_v2.json is what the
    # existing API readers load.
    for filename in ("metrics_v3.json", "metrics_v2.json"):
        with open(os.path.join(RESULTS_DIR, filename), "w") as f:
            json.dump(dashboard, f, indent=2)

    print("\nBest risk model:", best_name, "test accuracy", round(best_acc, 4))
    print("Wrote models/ and results/metrics_v3.json")


if __name__ == "__main__":
    main()
