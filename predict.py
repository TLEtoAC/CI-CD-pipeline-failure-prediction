"""
predict.py — CLI inference tool for CI/CD Pipeline Failure Prediction
Usage:
    python predict.py \\
        --ci_tool Jenkins --language Python --os ubuntu-latest \\
        --cloud_provider AWS --branch main --failure_stage build \\
        --build_duration_sec 1200 --test_duration_sec 600 --deploy_duration_sec 300 \\
        --cpu_usage_pct 75.0 --memory_usage_mb 4096 --retry_count 2 \\
        --is_flaky_test 0 --rollback_triggered 0 --incident_created 0 \\
        --severity MEDIUM
"""

import argparse
import json
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR  = "models"
RESULTS_DIR = "results"

# Matching feature sets from train.py
CAT_COLS  = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]
NUM_COLS  = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
             "cpu_usage_pct", "memory_usage_mb", "retry_count"]
ENG_NUM   = ["total_duration_sec", "build_ratio", "test_ratio", "cpu_x_retry", "mem_x_retry"]
ENG_BOOL  = ["is_main_branch", "is_release_branch", "is_deploy_stage"]

TASKS = ["risk_level", "rollback_triggered", "incident_created"]

# Task → bool cols used during training
TASK_BOOL = {
    "risk_level":          ["is_flaky_test", "rollback_triggered", "incident_created"] + ENG_BOOL,
    "rollback_triggered":  ["is_flaky_test"] + ENG_BOOL,
    "incident_created":    ["is_flaky_test"] + ENG_BOOL,
}

SEV_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
RED   = "\033[91m"
RISK_COLORS = {"LOW": "\033[94m", "MEDIUM": "\033[93m", "HIGH": "\033[91m", "CRITICAL": "\033[31m"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict CI/CD pipeline risk level, rollback likelihood, and incident likelihood.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Categorical
    p.add_argument("--ci_tool",        default="Jenkins",
                   choices=["Jenkins", "GitHub Actions", "GitLab CI", "Azure DevOps", "CircleCI"])
    p.add_argument("--language",       default="Python",
                   choices=["Python", "Go", "NodeJS", "C++", "Rust", "Java"])
    p.add_argument("--os",             default="ubuntu-latest",
                   choices=["ubuntu-latest", "windows-latest", "macos-latest"])
    p.add_argument("--cloud_provider", default="AWS",
                   choices=["AWS", "GCP", "Azure", "On-Prem"])
    p.add_argument("--branch",         default="main",
                   choices=["main", "develop", "feature-x", "release"])
    p.add_argument("--failure_stage",  default="build",
                   choices=["build", "test", "deploy"])

    # Numeric
    p.add_argument("--build_duration_sec",  type=int,   default=1200)
    p.add_argument("--test_duration_sec",   type=int,   default=600)
    p.add_argument("--deploy_duration_sec", type=int,   default=300)
    p.add_argument("--cpu_usage_pct",       type=float, default=60.0)
    p.add_argument("--memory_usage_mb",     type=int,   default=4096)
    p.add_argument("--retry_count",         type=int,   default=1)

    # Boolean / context inputs
    p.add_argument("--is_flaky_test",      type=int, default=0, choices=[0, 1])
    p.add_argument("--rollback_triggered", type=int, default=0, choices=[0, 1])
    p.add_argument("--incident_created",   type=int, default=0, choices=[0, 1])
    p.add_argument("--severity",           default="MEDIUM",
                   choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    return p.parse_args()


def engineer_features(args) -> pd.DataFrame:
    """Mirror the feature engineering from train.py."""
    sev_score = SEV_MAP[args.severity]
    total     = args.build_duration_sec + args.test_duration_sec + args.deploy_duration_sec

    row = {
        # Raw features
        "ci_tool":             args.ci_tool,
        "language":            args.language,
        "os":                  args.os,
        "cloud_provider":      args.cloud_provider,
        "branch":              args.branch,
        "failure_stage":       args.failure_stage,
        "build_duration_sec":  args.build_duration_sec,
        "test_duration_sec":   args.test_duration_sec,
        "deploy_duration_sec": args.deploy_duration_sec,
        "cpu_usage_pct":       args.cpu_usage_pct,
        "memory_usage_mb":     args.memory_usage_mb,
        "retry_count":         args.retry_count,
        # Boolean context
        "is_flaky_test":       args.is_flaky_test,
        "rollback_triggered":  args.rollback_triggered,
        "incident_created":    args.incident_created,
        # Engineered numeric
        "total_duration_sec":  total,
        "build_ratio":         args.build_duration_sec / (total + 1),
        "test_ratio":          args.test_duration_sec  / (total + 1),
        "cpu_x_retry":         args.cpu_usage_pct * args.retry_count,
        "mem_x_retry":         args.memory_usage_mb * args.retry_count,
        # Engineered boolean
        "is_main_branch":    int(args.branch == "main"),
        "is_release_branch": int(args.branch == "release"),
        "is_deploy_stage":   int(args.failure_stage == "deploy"),
    }
    return pd.DataFrame([row])


def get_task_feature_cols(task: str):
    return CAT_COLS + NUM_COLS + ENG_NUM + TASK_BOOL[task]


def load_and_predict(task: str, X_full: pd.DataFrame):
    prep_path  = os.path.join(MODELS_DIR, f"preprocessor_{task}.pkl")
    le_path    = os.path.join(MODELS_DIR, f"label_encoder_{task}.pkl")
    model_path = os.path.join(MODELS_DIR, f"best_model_{task}.pkl")

    for path in [prep_path, le_path, model_path]:
        if not os.path.exists(path):
            print(f"  ✗ Model artifact not found: {path}")
            print("    → Run  python train.py  first to train the models.")
            sys.exit(1)

    preprocessor = joblib.load(prep_path)
    le           = joblib.load(le_path)
    model        = joblib.load(model_path)

    feat_cols = get_task_feature_cols(task)
    X = X_full[feat_cols]
    X_enc = preprocessor.transform(X)

    pred_enc = model.predict(X_enc)[0]
    pred_lbl = le.inverse_transform([pred_enc])[0]

    probas = {}
    if hasattr(model, "predict_proba"):
        raw    = model.predict_proba(X_enc)[0]
        labels = le.inverse_transform(np.arange(len(le.classes_)))
        probas = {str(lbl): round(float(p), 4) for lbl, p in zip(labels, raw)}

    # Pull best model name from saved metrics
    best_model_name = "Unknown"
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        best_model_name = m.get(task, {}).get("__best_model__", "Unknown")

    return str(pred_lbl), probas, best_model_name


def print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║    CI/CD Pipeline Failure Prediction  —  Inference       ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")


def print_results(args, results):
    print(f"{BOLD}━━━ Input Parameters ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    for k, v in vars(args).items():
        print(f"  {k:<26} : {v}")

    print(f"\n{BOLD}━━━ Predictions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

    for task, (pred, probas, model_name) in results.items():
        task_label = task.replace("_", " ").title()
        print(f"\n  {BOLD}{CYAN}[{task_label}]{RESET}  (best model: {model_name})")

        # Color coding
        if task == "risk_level":
            color = RISK_COLORS.get(pred, "")
            icon  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(pred, "⚪")
        elif task in ("rollback_triggered", "incident_created"):
            is_yes = str(pred).lower() in ("1", "true")
            color  = RED if is_yes else GREEN
            icon   = "⚠️ " if is_yes else "✅"
            pred   = ("Yes" if is_yes else "No")
        else:
            color, icon = YELLOW, "🔸"

        print(f"  {icon}  Prediction : {color}{BOLD}{pred}{RESET}")

        if probas:
            sorted_p = sorted(probas.items(), key=lambda x: x[1], reverse=True)
            print(f"  {'Class':<30} {'Bar':32} {'Prob':>6}")
            for cls, prob in sorted_p:
                bar_len = int(prob * 28)
                bar = "█" * bar_len + "░" * (28 - bar_len)
                is_top = (cls == str(pred).replace("Yes","1").replace("No","0") or
                          cls == pred or
                          (pred in ("Yes","No") and
                           ((pred == "Yes" and cls in ("1","True","true")) or
                            (pred == "No"  and cls in ("0","False","false")))))
                hl = GREEN if is_top else ""
                print(f"    {hl}{cls:<28}{RESET} {bar} {prob*100:5.1f}%")

    print(f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")


def main():
    print_banner()
    args  = parse_args()
    X_df  = engineer_features(args)

    print(f"{BOLD}Running inference across all tasks…{RESET}\n")
    results = {}
    for task in TASKS:
        pred, probas, model_name = load_and_predict(task, X_df)
        results[task] = (pred, probas, model_name)

    print_results(args, results)


if __name__ == "__main__":
    main()
