"""
eda.py — Data cleaning + Exploratory Data Analysis for CI/CD pipeline logs.

Usage
    python eda.py
    python eda.py --data path/to/data.csv --out-dir plots

Outputs
    cleaned/ci_cd_pipeline_cleaned.csv   cleaned dataset
    cleaned/cleaning_report.json         cleaning summary
    plots/*.png                          EDA figures
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths / style
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, "ci_cd_pipeline_failure_logs_dataset.csv")
PLOTS_DIR = os.path.join(ROOT, "plots")
CLEAN_DIR = os.path.join(ROOT, "cleaned")

EXPECTED_COLS = [
    "pipeline_id",
    "run_id",
    "timestamp",
    "ci_tool",
    "repository",
    "branch",
    "commit_hash",
    "author",
    "language",
    "os",
    "cloud_provider",
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "failure_stage",
    "failure_type",
    "error_code",
    "error_message",
    "severity",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
    "is_flaky_test",
    "rollback_triggered",
    "incident_created",
]

CAT_COLS = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]
NUM_COLS = [
    "build_duration_sec",
    "test_duration_sec",
    "deploy_duration_sec",
    "cpu_usage_pct",
    "memory_usage_mb",
    "retry_count",
]
BOOL_COLS = ["is_flaky_test", "rollback_triggered", "incident_created"]
TARGET_COLS = ["failure_type", "severity", "incident_created", "rollback_triggered"]

# Valid category sets (case-insensitive match → canonical form)
CANONICAL = {
    "severity": {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "critical": "CRITICAL"},
    "failure_stage": {"build": "build", "test": "test", "deploy": "deploy"},
    "branch": {
        "main": "main",
        "develop": "develop",
        "release": "release",
        "feature-x": "feature-x",
    },
    "ci_tool": {
        "jenkins": "Jenkins",
        "github actions": "GitHub Actions",
        "gitlab ci": "GitLab CI",
        "azure devops": "Azure DevOps",
        "circleci": "CircleCI",
    },
    "language": {
        "python": "Python",
        "go": "Go",
        "nodejs": "NodeJS",
        "c++": "C++",
        "rust": "Rust",
        "java": "Java",
    },
    "os": {
        "windows-latest": "windows-latest",
        "ubuntu-latest": "ubuntu-latest",
        "macos-latest": "macos-latest",
    },
    "cloud_provider": {
        "on-prem": "On-Prem",
        "aws": "AWS",
        "gcp": "GCP",
        "azure": "Azure",
    },
}

# Soft numeric bounds used to flag / clip extreme outliers
NUM_BOUNDS = {
    "build_duration_sec": (0, 7200),
    "test_duration_sec": (0, 7200),
    "deploy_duration_sec": (0, 7200),
    "cpu_usage_pct": (0, 100),
    "memory_usage_mb": (0, 65536),
    "retry_count": (0, 20),
}

SEV_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEV_PAL = {
    "LOW": "#06b6d4",
    "MEDIUM": "#f59e0b",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

BG = "#0f1117"
AX_BG = "#1a1d2e"
TEXT = "#e0e0e0"
GRID = "#2e3150"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": AX_BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "legend.facecolor": AX_BG,
        "legend.edgecolor": GRID,
    }
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def save_fig(fig: plt.Figure, plots_dir: str, name: str) -> None:
    path = os.path.join(plots_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ Saved {path}")


def _strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    obj = df.select_dtypes(include=["object"]).columns
    for c in obj:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "nan", "None", "NULL", "null", "NaN"]), c] = np.nan
    return df


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def _canonicalize(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    key = series.astype(str).str.strip().str.lower()
    return key.map(mapping)


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame, clip_outliers: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean CI/CD failure logs and return (cleaned_df, report)."""
    report: dict[str, Any] = {
        "rows_in": int(len(df)),
        "cols_in": int(df.shape[1]),
        "steps": [],
    }
    out = df.copy()

    # 1) Column presence
    missing_cols = [c for c in EXPECTED_COLS if c not in out.columns]
    extra_cols = [c for c in out.columns if c not in EXPECTED_COLS]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    out = out[EXPECTED_COLS]
    report["steps"].append(
        {
            "name": "select_columns",
            "dropped_extra": extra_cols,
            "missing_required": missing_cols,
        }
    )

    # 2) Strip / normalize string empties
    nulls_before = int(out.isna().sum().sum())
    out = _strip_object_cols(out)
    report["steps"].append(
        {
            "name": "strip_strings",
            "nulls_before": nulls_before,
            "nulls_after_strip": int(out.isna().sum().sum()),
        }
    )

    # 3) Drop exact duplicates + duplicate run_id
    n_dup = int(out.duplicated().sum())
    out = out.drop_duplicates()
    n_dup_run = int(out["run_id"].duplicated().sum())
    out = out.drop_duplicates(subset=["run_id"], keep="first")
    report["steps"].append(
        {
            "name": "deduplicate",
            "exact_duplicates_removed": n_dup,
            "duplicate_run_ids_removed": n_dup_run,
        }
    )

    # 4) Parse timestamp
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)
    bad_ts = int(out["timestamp"].isna().sum())
    report["steps"].append({"name": "parse_timestamp", "invalid_timestamps": bad_ts})

    # 5) Booleans
    for col in BOOL_COLS:
        out[col] = _to_bool(out[col])
    report["steps"].append({"name": "cast_booleans", "columns": BOOL_COLS})

    # 6) Numerics
    for col in NUM_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    report["steps"].append(
        {
            "name": "cast_numerics",
            "nulls_after_cast": {c: int(out[c].isna().sum()) for c in NUM_COLS},
        }
    )

    # 7) Canonicalize categoricals
    cat_fixes: dict[str, Any] = {}
    for col, mapping in CANONICAL.items():
        before_null = int(out[col].isna().sum())
        mapped = _canonicalize(out[col], mapping)
        unknown = int(mapped.isna().sum() - before_null)
        out[col] = mapped
        cat_fixes[col] = {"unknown_or_invalid": unknown}
    report["steps"].append({"name": "canonicalize_categoricals", "detail": cat_fixes})

    # 8) Impute
    imputed: dict[str, Any] = {}
    for col in NUM_COLS:
        n = int(out[col].isna().sum())
        if n:
            fill = float(out[col].median())
            out[col] = out[col].fillna(fill)
            imputed[col] = {"count": n, "strategy": "median", "value": fill}
    for col in list(CANONICAL.keys()) + ["failure_type", "error_code", "error_message",
                                         "pipeline_id", "repository", "commit_hash", "author"]:
        if col not in out.columns:
            continue
        n = int(out[col].isna().sum())
        if n:
            mode = out[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "UNKNOWN"
            out[col] = out[col].fillna(fill)
            imputed[col] = {"count": n, "strategy": "mode", "value": str(fill)}
    # Drop rows that still lack a usable timestamp (rare)
    before_ts = len(out)
    out = out.dropna(subset=["timestamp"])
    imputed["timestamp_rows_dropped"] = before_ts - len(out)
    report["steps"].append({"name": "impute", "detail": imputed})

    # 9) Outlier handling (IQR flag + optional soft clip to domain bounds)
    outlier_info: dict[str, Any] = {}
    for col, (lo, hi) in NUM_BOUNDS.items():
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = int(((out[col] < fence_lo) | (out[col] > fence_hi)).sum())
        domain_out = int(((out[col] < lo) | (out[col] > hi)).sum())
        if clip_outliers and domain_out:
            out[col] = out[col].clip(lo, hi)
        outlier_info[col] = {
            "iqr_outliers": iqr_out,
            "domain_outliers": domain_out,
            "clipped_to": [lo, hi] if clip_outliers else None,
        }
    report["steps"].append({"name": "outliers", "detail": outlier_info})

    # 10) Derived features useful for EDA / modeling preview
    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    out["sev_score"] = out["severity"].map(sev_map).astype("Int64")
    out["total_duration_sec"] = (
        out["build_duration_sec"] + out["test_duration_sec"] + out["deploy_duration_sec"]
    )
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.day_name()
    out["date"] = out["timestamp"].dt.date.astype(str)

    # Final dtypes
    for col in BOOL_COLS:
        out[col] = out[col].astype(int)
    for col in ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
                "memory_usage_mb", "retry_count"]:
        out[col] = out[col].round().astype(int)
    out["cpu_usage_pct"] = out["cpu_usage_pct"].astype(float)

    report["rows_out"] = int(len(out))
    report["cols_out"] = int(out.shape[1])
    report["rows_removed"] = report["rows_in"] - report["rows_out"]
    report["nulls_remaining"] = {c: int(out[c].isna().sum()) for c in out.columns if out[c].isna().any()}
    return out, report


def print_profile(df: pd.DataFrame) -> None:
    print("\n── Dataset profile ──")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Null cells remaining: {int(df.isna().sum().sum())}")
    print("\n  Target balances:")
    for col in ["failure_type", "severity", "failure_stage", "incident_created", "rollback_triggered"]:
        print(f"\n  [{col}]")
        vc = df[col].value_counts(normalize=True).mul(100).round(1)
        for k, v in vc.items():
            print(f"    {k}: {v}%")
    print("\n  Numeric summary:")
    print(df[NUM_COLS + ["total_duration_sec"]].describe().round(2).to_string())


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
def plot_missingness(df_raw: pd.DataFrame, plots_dir: str) -> None:
    print("[1/10] Missingness overview...")
    miss = df_raw.isna().mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in miss.values]
    ax.barh(miss.index.astype(str), miss.values, color=colors)
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Values by Column (raw data)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(5, miss.max() * 1.2 if len(miss) else 5))
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, plots_dir, "01_missingness.png")


def plot_targets(df: pd.DataFrame, plots_dir: str) -> None:
    print("[2/10] Target distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Target Variable Distributions", fontsize=16, fontweight="bold", y=1.02)

    specs = [
        ("failure_type", "Failure Type", sns.color_palette("viridis", 10)),
        ("severity", "Severity Level", [SEV_PAL[s] for s in SEV_ORDER if s in df["severity"].unique()]),
        ("incident_created", "Incident Created", ["#06b6d4", "#ef4444"]),
    ]
    for ax, (col, title, colors) in zip(axes, specs):
        if col == "severity":
            counts = df[col].value_counts().reindex([s for s in SEV_ORDER if s in df[col].unique()])
        else:
            counts = df[col].value_counts()
        bars = ax.barh(counts.index.astype(str), counts.values, color=colors[: len(counts)])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Count")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + max(counts.values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, plots_dir, "02_target_distributions.png")


def plot_numeric_distributions(df: pd.DataFrame, plots_dir: str) -> None:
    print("[3/10] Numeric feature distributions...")
    cols = NUM_COLS + ["total_duration_sec"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Numeric Feature Distributions", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    pal = sns.color_palette("viridis", len(cols))
    for ax, col, color in zip(axes, cols, pal):
        data = df[col]
        ax.hist(data, bins=40, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(data.mean(), color="#f59e0b", lw=1.5, ls="--", label=f"Mean {data.mean():.1f}")
        ax.axvline(data.median(), color="#06b6d4", lw=1.5, ls=":", label=f"Median {data.median():.1f}")
        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    save_fig(fig, plots_dir, "03_numeric_distributions.png")


def plot_correlation(df: pd.DataFrame, plots_dir: str) -> None:
    print("[4/10] Correlation heatmap...")
    corr_cols = NUM_COLS + ["total_duration_sec", "sev_score"] + BOOL_COLS
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, linecolor=BG, annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    save_fig(fig, plots_dir, "04_correlation_heatmap.png")


def plot_categorical_vs_failure(df: pd.DataFrame, plots_dir: str) -> None:
    print("[5/10] Categorical vs failure type...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Categorical Features — Failure Type Breakdown", fontsize=15, fontweight="bold")
    axes = axes.flatten()
    ftypes = df["failure_type"].unique()
    pal = dict(zip(ftypes, sns.color_palette("tab10", len(ftypes))))
    for ax, col in zip(axes, CAT_COLS):
        ct = pd.crosstab(df[col], df["failure_type"])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot(
            kind="bar", stacked=True, ax=ax,
            color=[pal.get(c, "#888") for c in ct_pct.columns],
            edgecolor="none", width=0.75, legend=False,
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("% of Failures")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[t]) for t in ftypes]
    fig.legend(handles, list(ftypes), loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    save_fig(fig, plots_dir, "05_categorical_vs_failure_type.png")


def plot_duration_vs_severity(df: pd.DataFrame, plots_dir: str) -> None:
    print("[6/10] Duration vs severity...")
    duration_cols = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec", "total_duration_sec"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Pipeline Duration vs Severity", fontsize=15, fontweight="bold")
    order = [s for s in SEV_ORDER if s in df["severity"].unique()]
    for ax, col in zip(axes, duration_cols):
        data_by_sev = [df.loc[df["severity"] == s, col].values for s in order]
        bp = ax.boxplot(
            data_by_sev, patch_artist=True,
            medianprops={"color": "#fff", "linewidth": 2},
            whiskerprops={"color": TEXT},
            capprops={"color": TEXT},
            flierprops={"marker": ".", "markerfacecolor": "#888", "markersize": 3, "alpha": 0.4},
        )
        for patch, sev in zip(bp["boxes"], order):
            patch.set_facecolor(SEV_PAL[sev])
            patch.set_alpha(0.85)
        ax.set_xticklabels(order)
        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel("Seconds")
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, plots_dir, "06_duration_vs_severity.png")


def plot_resource_vs_incident(df: pd.DataFrame, plots_dir: str) -> None:
    print("[7/10] Resource usage vs incident...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Resource Usage vs Incident Created", fontsize=15, fontweight="bold")
    inc_pal = {0: "#06b6d4", 1: "#ef4444"}
    labels = {0: "No Incident", 1: "Incident Created"}
    for ax, col, label in zip(
        axes,
        ["cpu_usage_pct", "memory_usage_mb"],
        ["CPU Usage (%)", "Memory Usage (MB)"],
    ):
        for val in [0, 1]:
            ax.hist(
                df.loc[df["incident_created"] == val, col],
                bins=40, alpha=0.65, color=inc_pal[val],
                label=labels[val], edgecolor="none",
            )
        ax.set_title(f"{label} by Incident Status", fontsize=12, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, plots_dir, "07_resource_usage_vs_incident.png")


def plot_temporal(df: pd.DataFrame, plots_dir: str) -> None:
    print("[8/10] Temporal patterns...")
    daily = df.groupby("date").size()
    hour_sev = pd.crosstab(df["hour"], df["severity"])
    hour_sev = hour_sev.reindex(columns=[s for s in SEV_ORDER if s in hour_sev.columns])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Temporal Failure Patterns", fontsize=15, fontweight="bold")

    axes[0].plot(pd.to_datetime(daily.index), daily.values, color="#22d3ee", lw=1.5)
    axes[0].fill_between(pd.to_datetime(daily.index), daily.values, alpha=0.25, color="#22d3ee")
    axes[0].set_title("Failures per Day", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Count")
    axes[0].grid(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    bottom = np.zeros(len(hour_sev))
    for sev in hour_sev.columns:
        axes[1].bar(hour_sev.index, hour_sev[sev], bottom=bottom, color=SEV_PAL[sev], label=sev, width=0.85)
        bottom += hour_sev[sev].values
    axes[1].set_title("Failures by Hour of Day (stacked by severity)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, plots_dir, "08_temporal_patterns.png")


def plot_stage_branch_heatmaps(df: pd.DataFrame, plots_dir: str) -> None:
    print("[9/10] Stage / branch heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Incident Rate by Context", fontsize=15, fontweight="bold")

    stage_sev = pd.crosstab(df["failure_stage"], df["severity"], normalize="index") * 100
    stage_sev = stage_sev.reindex(columns=[s for s in SEV_ORDER if s in stage_sev.columns])
    sns.heatmap(stage_sev, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0],
                linewidths=0.5, linecolor=BG, cbar_kws={"label": "%"})
    axes[0].set_title("Severity Mix by Failure Stage (%)", fontsize=12, fontweight="bold")

    branch_inc = (
        df.groupby(["branch", "failure_stage"])["incident_created"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )
    sns.heatmap(branch_inc, annot=True, fmt=".1f", cmap="magma", ax=axes[1],
                linewidths=0.5, linecolor=BG, cbar_kws={"label": "Incident %"})
    axes[1].set_title("Incident Rate by Branch × Stage (%)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, plots_dir, "09_stage_branch_heatmaps.png")


def plot_flaky_retry(df: pd.DataFrame, plots_dir: str) -> None:
    print("[10/10] Flaky tests & retries...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Flaky Tests and Retry Behavior", fontsize=15, fontweight="bold")

    ct = pd.crosstab(df["retry_count"], df["is_flaky_test"])
    ct.plot(kind="bar", ax=axes[0], color=["#06b6d4", "#ef4444"], edgecolor="none", width=0.75)
    axes[0].set_title("Retry Count by Flaky Flag", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Retry Count")
    axes[0].set_ylabel("Count")
    axes[0].legend(["Not Flaky", "Flaky"], fontsize=9)
    axes[0].grid(axis="y")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    rates = (
        df.groupby("is_flaky_test")[["rollback_triggered", "incident_created"]]
        .mean()
        .mul(100)
    )
    rates.index = ["Not Flaky", "Flaky"]
    rates.plot(kind="bar", ax=axes[1], color=["#f59e0b", "#ef4444"], edgecolor="none", width=0.7)
    axes[1].set_title("Rollback / Incident Rate by Flaky Flag", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(rates.index, rotation=0)
    axes[1].legend(["Rollback", "Incident"], fontsize=9)
    axes[1].grid(axis="y")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, plots_dir, "10_flaky_retry_analysis.png")


def run_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    plot_missingness(df_raw, plots_dir)
    plot_targets(df_clean, plots_dir)
    plot_numeric_distributions(df_clean, plots_dir)
    plot_correlation(df_clean, plots_dir)
    plot_categorical_vs_failure(df_clean, plots_dir)
    plot_duration_vs_severity(df_clean, plots_dir)
    plot_resource_vs_incident(df_clean, plots_dir)
    plot_temporal(df_clean, plots_dir)
    plot_stage_branch_heatmaps(df_clean, plots_dir)
    plot_flaky_retry(df_clean, plots_dir)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean CI/CD failure logs and run EDA")
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to raw CSV")
    p.add_argument("--plots-dir", default=PLOTS_DIR, help="Directory for plot PNGs")
    p.add_argument("--clean-dir", default=CLEAN_DIR, help="Directory for cleaned CSV + report")
    p.add_argument("--no-clip", action="store_true", help="Do not clip domain outliers")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.clean_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"Loading {args.data} ...")
    raw = pd.read_csv(args.data)
    print(f"  Raw shape: {raw.shape}")

    print("\nCleaning data ...")
    clean, report = clean_data(raw, clip_outliers=not args.no_clip)
    print(f"  Clean shape: {clean.shape}  (removed {report['rows_removed']} rows)")

    clean_path = os.path.join(args.clean_dir, "ci_cd_pipeline_cleaned.csv")
    report_path = os.path.join(args.clean_dir, "cleaning_report.json")
    # CSV-friendly timestamp
    clean_to_save = clean.copy()
    clean_to_save["timestamp"] = clean_to_save["timestamp"].astype(str)
    clean_to_save.to_csv(clean_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ Saved {clean_path}")
    print(f"  ✓ Saved {report_path}")

    print_profile(clean)

    print("\nGenerating EDA plots ...")
    run_eda(raw, clean, args.plots_dir)
    print(f"\n✅ Done. Plots → '{args.plots_dir}/'  |  Cleaned data → '{args.clean_dir}/'")


if __name__ == "__main__":
    main()
