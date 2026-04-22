"""
eda.py — Exploratory Data Analysis for CI/CD Pipeline Failure Prediction
Generates plots saved to the `plots/` directory.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH = "ci_cd_pipeline_failure_logs_dataset.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = "viridis"
BG_COLOR = "#0f1117"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#7c3aed"

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


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df["is_flaky_test"] = df["is_flaky_test"].astype(int)
df["rollback_triggered"] = df["rollback_triggered"].astype(int)
df["incident_created"] = df["incident_created"].astype(int)
print(f"  Shape: {df.shape}")


# ─────────────────────────────────────────────
# 1. Target distributions
# ─────────────────────────────────────────────
print("\n[1/6] Target class distributions...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Target Variable Distributions", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)

targets = [
    ("failure_type", "Failure Type"),
    ("severity", "Severity Level"),
    ("incident_created", "Incident Created"),
]

colors_list = [
    sns.color_palette("viridis", 10),
    sns.color_palette("magma", 4),
    ["#7c3aed", "#06b6d4"],
]

for ax, (col, title), colors in zip(axes, targets, colors_list):
    counts = df[col].value_counts()
    bars = ax.barh(counts.index.astype(str), counts.values, color=colors[:len(counts)])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.grid(axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save(fig, "01_target_distributions.png")


# ─────────────────────────────────────────────
# 2. Numeric feature distributions
# ─────────────────────────────────────────────
print("[2/6] Numeric feature distributions...")
num_cols = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec",
            "cpu_usage_pct", "memory_usage_mb", "retry_count"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Numeric Feature Distributions", fontsize=16, fontweight="bold", color=TEXT_COLOR)
axes = axes.flatten()

pal = sns.color_palette("viridis", len(num_cols))
for ax, col, color in zip(axes, num_cols, pal):
    data = df[col]
    ax.hist(data, bins=40, color=color, alpha=0.85, edgecolor="none")
    ax.axvline(data.mean(), color="#f59e0b", linewidth=1.5, linestyle="--", label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.median(), color="#06b6d4", linewidth=1.5, linestyle=":", label=f"Median: {data.median():.1f}")
    ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save(fig, "02_numeric_distributions.png")


# ─────────────────────────────────────────────
# 3. Correlation heatmap
# ─────────────────────────────────────────────
print("[3/6] Correlation heatmap...")
corr_df = df[num_cols + ["is_flaky_test", "rollback_triggered", "incident_created"]].copy()
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1, ax=ax,
    linewidths=0.5, linecolor="#0f1117",
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
save(fig, "03_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 4. Categorical features vs failure_type
# ─────────────────────────────────────────────
print("[4/6] Categorical feature breakdowns...")
cat_cols = ["ci_tool", "language", "os", "cloud_provider", "branch", "failure_stage"]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Categorical Features — Failure Type Breakdown", fontsize=15, fontweight="bold", color=TEXT_COLOR)
axes = axes.flatten()

failure_types = df["failure_type"].unique()
pal = dict(zip(failure_types, sns.color_palette("tab10", len(failure_types))))

for ax, col in zip(axes, cat_cols):
    ct = pd.crosstab(df[col], df["failure_type"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", stacked=True, ax=ax,
                color=[pal.get(c, "#888") for c in ct_pct.columns],
                edgecolor="none", width=0.75)
    ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("% of Failures")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save(fig, "04_categorical_vs_failure_type.png")


# ─────────────────────────────────────────────
# 5. Duration vs severity box plots
# ─────────────────────────────────────────────
print("[5/6] Duration vs severity box plots...")
duration_cols = ["build_duration_sec", "test_duration_sec", "deploy_duration_sec"]
sev_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
sev_pal = {"LOW": "#06b6d4", "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Pipeline Duration vs Severity", fontsize=15, fontweight="bold", color=TEXT_COLOR)

for ax, col in zip(axes, duration_cols):
    data_by_sev = [df[df["severity"] == s][col].values for s in sev_order]
    bp = ax.boxplot(data_by_sev, patch_artist=True, notch=False,
                    medianprops={"color": "#fff", "linewidth": 2},
                    whiskerprops={"color": TEXT_COLOR},
                    capprops={"color": TEXT_COLOR},
                    flierprops={"marker": ".", "markerfacecolor": "#888", "markersize": 3, "alpha": 0.4})
    for patch, sev in zip(bp["boxes"], sev_order):
        patch.set_facecolor(sev_pal[sev])
        patch.set_alpha(0.8)
    ax.set_xticklabels(sev_order)
    ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save(fig, "05_duration_vs_severity.png")


# ─────────────────────────────────────────────
# 6. Resource usage vs incident_created
# ─────────────────────────────────────────────
print("[6/6] Resource usage vs incident created...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Resource Usage vs Incident Created", fontsize=15, fontweight="bold", color=TEXT_COLOR)

resource_cols = ["cpu_usage_pct", "memory_usage_mb"]
resource_labels = ["CPU Usage (%)", "Memory Usage (MB)"]
inc_pal = {0: "#06b6d4", 1: "#ef4444"}
inc_labels = {0: "No Incident", 1: "Incident Created"}

for ax, col, label in zip(axes, resource_cols, resource_labels):
    for val in [0, 1]:
        data = df[df["incident_created"] == val][col]
        ax.hist(data, bins=40, alpha=0.65, color=inc_pal[val], label=inc_labels[val], edgecolor="none")
    ax.set_title(f"{label} by Incident Status", fontsize=12, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save(fig, "06_resource_usage_vs_incident.png")

print(f"\n✅ EDA complete! All plots saved to '{PLOTS_DIR}/'")
