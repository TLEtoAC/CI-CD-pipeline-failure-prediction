# EDA Script ↔ Syllabus Concept Mapping

This document maps the data-science syllabus topics to **`eda.py`**: where each idea appears in the code, and how it helps the CI/CD failure analysis.

> **Scope note:** `eda.py` covers cleaning, exploration, transformation, and visualization. Topics that belong to **modeling** (PCA, train/test split, confusion matrix, linear regression) are **not** in `eda.py`; they appear in `train_risk.py` / `train_v3.py` and are flagged at the end.

---

## Quick index

| Syllabus topic | In `eda.py`? | Primary location |
|----------------|--------------|------------------|
| Types of attributes | Yes | `CAT_COLS`, `NUM_COLS`, `BOOL_COLS` |
| Data sources & data quality | Yes | CSV load + `cleaning_report.json` |
| Data cleaning | Yes | `clean_data()` |
| Identifying outliers | Yes | IQR + domain bounds in `clean_data()` |
| Measures of centre & spread | Yes | `describe()`, mean/median lines, box plots |
| Hypothesis testing vs EDA | Yes (EDA side) | All `plot_*` + `print_profile` |
| Data transformation | Yes | Casts, canonicalize, derived features |
| Structural inspection | Yes | Column checks, `shape`, profile |
| Data encoding | Yes | Bool→int, severity→`sev_score` |
| NumPy arrays / vectorized ops | Yes | Masks, `np.zeros`, correlation mask |
| Pandas DataFrames (`loc`, grouping) | Yes | Throughout cleaning & plots |
| Lists & dictionaries | Yes | `CANONICAL`, `report`, mappings |
| Descriptive stats / distribution / correlation | Yes | Profile + plots 02–04 |
| Aggregation & grouping | Yes | `groupby`, `crosstab` |
| Feature engineering (extraction) | Yes | Derived columns in step 10 |
| Feature scaling / selection / CV / metrics | No | Training scripts |
| Visualization (bar, box, hist, line, heatmap) | Yes | Plots 01–10 |
| Linear / multiple regression | No | Training scripts |

---

## Unit-style foundations (attributes, quality, cleaning, EDA)

### 1. Types of attributes

**Where**

```68:78:eda.py
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
```

| Attribute type | Examples in dataset | How used |
|----------------|---------------------|----------|
| **Nominal categorical** | `ci_tool`, `language`, `os`, `cloud_provider` | Crosstabs, stacked bars |
| **Ordinal categorical** | `severity` (`LOW`→`CRITICAL`) | Ordered plots + `sev_score` |
| **Numeric continuous** | `cpu_usage_pct`, durations, memory | Histograms, correlation, outliers |
| **Numeric discrete** | `retry_count` | Bar charts vs flaky flag |
| **Binary / boolean** | `is_flaky_test`, `incident_created` | Encoded to 0/1 for analysis |
| **Temporal** | `timestamp` | Daily trends, hour-of-day patterns |
| **Identifier / text** | `run_id`, `error_message` | Dedup / kept, not modeled in EDA plots |

**How it helps:** Treating each column by type avoids wrong ops (e.g. mean of `ci_tool`) and drives the right chart and cleaning rule per field.

---

### 2. Data sources and data quality

**Where**

- **Source:** `pd.read_csv(...)` in `main()` — file `ci_cd_pipeline_failure_logs_dataset.csv` (pipeline telemetry / failure logs).
- **Quality checks:** missingness plot, null counts in the cleaning report, duplicate removal, invalid timestamp count, domain/IQR outlier counts.

```372:385:eda.py
def plot_missingness(df_raw: pd.DataFrame, plots_dir: str) -> None:
    ...
    miss = df_raw.isna().mean().mul(100).sort_values(ascending=False)
    ...
    save_fig(fig, plots_dir, "01_missingness.png")
```

```679:688:eda.py
    clean_path = os.path.join(args.clean_dir, "ci_cd_pipeline_cleaned.csv")
    report_path = os.path.join(args.clean_dir, "cleaning_report.json")
    ...
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
```

**How it helps:** Documents *where* data came from and *what was wrong* before modeling, so later risk scores are not biased by garbage rows or silent type errors.

---

### 3. Data cleaning

**Where:** entire `clean_data()` (steps 1–9).

| Step | Code idea | Quality issue fixed |
|------|-----------|---------------------|
| 1 | Keep only `EXPECTED_COLS` | Schema drift / unexpected columns |
| 2 | `_strip_object_cols` | Whitespace, empty strings → NaN |
| 3 | `drop_duplicates` / `run_id` | Duplicate pipeline runs |
| 4 | `pd.to_datetime(..., errors="coerce")` | Bad timestamps |
| 5–6 | `_to_bool`, `pd.to_numeric` | Mixed / string types |
| 7 | `_canonicalize` + `CANONICAL` | `"aws"` vs `"AWS"`, etc. |
| 8 | Median / mode impute; drop bad timestamps | Missing values |
| 9 | IQR flag + domain `clip` | Extreme / impossible values |

**How it helps:** Produces a consistent table (`cleaned/ci_cd_pipeline_cleaned.csv`) that EDA and training can trust.

---

### 4. Identifying outliers

**Where**

```312:327:eda.py
    # 9) Outlier handling (IQR flag + optional soft clip to domain bounds)
    ...
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        ...
        if clip_outliers and domain_out:
            out[col] = out[col].clip(lo, hi)
```

Also visible visually in:

- Histograms (`plot_numeric_distributions`) — long tails
- Box plots (`plot_duration_vs_severity`) — fliers beyond whiskers

**How it helps:** Separates *statistically unusual* (IQR) from *physically impossible* (e.g. CPU > 100%). Clipping domain outliers stops one bad row from dominating means/correlations.

---

### 5. Measures of centre and spread

**Where**

| Measure | In code | Purpose |
|---------|---------|---------|
| **Mean** | `data.mean()` on histograms | Typical duration / CPU |
| **Median** | `data.median()`; also impute fill | Robust centre when skewed |
| **Mode** | `out[col].mode()` for categoricals | Impute categories |
| **Std / quartiles / min / max** | `df[...].describe()` in `print_profile` | Spread & range |
| **IQR** | `q3 - q1` in outlier step | Outlier fences |
| **Box-plot IQR / whiskers** | `ax.boxplot(...)` | Centre + spread by severity |

```427:428:eda.py
        ax.axvline(data.mean(), color="#f59e0b", lw=1.5, ls="--", label=f"Mean {data.mean():.1f}")
        ax.axvline(data.median(), color="#06b6d4", lw=1.5, ls=":", label=f"Median {data.median():.1f}")
```

**How it helps:** Mean vs median on the same plot shows skewness; `describe()` and box plots show whether severity groups differ in duration spread (useful before risk modeling).

---

### 6. Hypothesis testing vs Exploratory Data Analysis

**In this script we practice EDA, not formal hypothesis tests.**

| Approach | Goal | How `eda.py` relates |
|----------|------|----------------------|
| **EDA** | Explore patterns, generate questions | All `plot_*`, `print_profile`, correlation heatmap |
| **Hypothesis testing** | Confirm a pre-stated claim with p-values | Not implemented here |

Examples of *exploratory* questions the plots support:

- Are failure types balanced? → target bars  
- Do longer builds associate with higher severity? → box plots  
- Do flaky tests coincide with more incidents? → flaky/retry bars  
- Is there a daily/hourly failure rhythm? → temporal line + stacked bars  

**How it helps:** EDA tells you *what is worth testing or modeling later* without locking into a single null hypothesis too early.

---

### 7. Data transformation

**Where**

| Transform | Location | Effect |
|-----------|----------|--------|
| Type casting | bool / numeric / datetime | Machine-usable types |
| String normalize + canonicalize | `_strip_object_cols`, `_canonicalize` | One spelling per category |
| Ordinal encode severity | `sev_map` → `sev_score` | Numeric severity for correlation |
| Bool → int (0/1) | final dtypes on `BOOL_COLS` | Correlation / grouping friendly |
| Feature construction | `total_duration_sec`, `hour`, `day_of_week`, `date` | New analysis axes |
| Clip to bounds | `Series.clip(lo, hi)` | Cap impossible numerics |

```329:337:eda.py
    sev_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    out["sev_score"] = out["severity"].map(sev_map).astype("Int64")
    out["total_duration_sec"] = (
        out["build_duration_sec"] + out["test_duration_sec"] + out["deploy_duration_sec"]
    )
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.day_name()
    out["date"] = out["timestamp"].dt.date.astype(str)
```

**How it helps:** Raw logs become analysis-ready; derived time and total-duration fields unlock trend and severity comparisons.

---

### 8. Structural inspection

**Where**

- Expected schema list `EXPECTED_COLS`
- `raw.shape` / clean shape prints
- Missing-column raise in `clean_data`
- `print_profile`: shape, time span, class balances, `describe()`

```220:225:eda.py
    missing_cols = [c for c in EXPECTED_COLS if c not in out.columns]
    ...
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    out = out[EXPECTED_COLS]
```

**How it helps:** Fails fast if the CSV schema changes, and gives a one-screen structural summary before plots.

---

### 9. Data encoding

**Where**

| Encoding | Code | Why |
|----------|------|-----|
| Label / ordinal for severity | `sev_map` → `sev_score` | Ordered levels for corr & heatmaps |
| Boolean encoding | `_to_bool` then `.astype(int)` | Stable 0/1 targets |
| Category canonicalization | `CANONICAL` dictionaries | Controlled category levels |
| Mapping dicts for plot labels | `inc_pal`, `SEV_PAL`, `labels` | Human-readable legends |

**How it helps:** Models and numeric EDA need numbers; encoding keeps meaning (LOW < CRITICAL) while enabling `corr()` and rate calculations.

> Full **one-hot encoding** of categoricals for ML is done later in `train_risk.py` (`OneHotEncoder`), not in `eda.py`.

---

## Unit IV — Programming with Python / NumPy / Pandas

### 10. NumPy arrays, indexing, vectorized operations

**Where**

```447:447:eda.py
    mask = np.triu(np.ones_like(corr, dtype=bool))
```

```564:567:eda.py
    bottom = np.zeros(len(hour_sev))
    for sev in hour_sev.columns:
        axes[1].bar(hour_sev.index, hour_sev[sev], bottom=bottom, ...)
        bottom += hour_sev[sev].values
```

Also: boolean masks like `(out[col] < fence_lo) | (out[col] > fence_hi)`, and `.values` when feeding box plots.

**How it helps:** Vectorized masks count outliers without Python `for` loops over every cell; NumPy builds the upper-triangle mask for a cleaner correlation heatmap.

**Not in `eda.py`:** PCA, explicit matrix subsetting as a teaching demo — those are modeling / advanced linear-algebra topics.

---

### 11. Categorical data & Pandas DataFrames (`loc`, grouping, summaries)

**Where**

| Pandas skill | Example in `eda.py` |
|--------------|---------------------|
| Import CSV | `pd.read_csv` |
| Column select | `out[EXPECTED_COLS]`, `df[col]` |
| `loc` filtering | `df.loc[df["severity"] == s, col]` |
| `isna` / `fillna` / `dropna` | Imputation & missingness |
| `value_counts` | Target balances |
| `describe` | Numeric summary |
| `groupby` + agg | Daily counts; incident rates |
| `crosstab` | Stage×severity, retry×flaky |
| `corr` | Correlation matrix |
| Extend DataFrame | Add `sev_score`, `hour`, … |
| Sort (implicit) | `sort_values` on missingness % |

```497:497:eda.py
        data_by_sev = [df.loc[df["severity"] == s, col].values for s in order]
```

```591:596:eda.py
    branch_inc = (
        df.groupby(["branch", "failure_stage"])["incident_created"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )
```

**How it helps:** DataFrames are the working structure for cleaning and every EDA chart; `loc` + `groupby` answer “by severity / by branch / by day” questions.

---

### 12. Python lists & dictionaries (data-science applications)

**Where**

| Structure | Examples | Role |
|-----------|----------|------|
| **Lists** | `EXPECTED_COLS`, `NUM_COLS`, `SEV_ORDER`, plot column lists | Ordered schemas & plot loops |
| **Dicts** | `CANONICAL`, `NUM_BOUNDS`, `SEV_PAL`, `sev_map`, cleaning `report` | Lookups, config, JSON-serializable audit |

```81:116:eda.py
CANONICAL = {
    "severity": {"low": "LOW", "medium": "MEDIUM", ...},
    ...
}
```

```213:217:eda.py
    report: dict[str, Any] = {
        "rows_in": int(len(df)),
        "cols_in": int(df.shape[1]),
        "steps": [],
    }
```

**How it helps:** Lists drive reproducible column order; dicts encode business rules (valid labels, color maps) and produce the cleaning report without hard-coding every branch.

---

## Unit V — Working with data, EDA, feature engineering

### 13. Data manipulation with Pandas

Covered throughout `clean_data`, `print_profile`, and plot helpers:

- **Import:** `read_csv`
- **Inspect:** shape prints, `describe()`, value balances (similar role to `head`/`info` in interactive work)
- **Aggregate / group:** `groupby`, `crosstab`
- **Merge / concat:** not required for this single-file pipeline (one CSV)

**How it helps:** Turns 45k log rows into summaries (rates, daily counts, category mixes) used for decisions and plots.

---

### 14. Exploratory Data Analysis (descriptive, distribution, correlation, trends)

| EDA goal | Function | Output file |
|----------|----------|-------------|
| Missingness | `plot_missingness` | `01_missingness.png` |
| Class balance | `plot_targets` | `02_target_distributions.png` |
| Distributions | `plot_numeric_distributions` | `03_numeric_distributions.png` |
| Correlation | `plot_correlation` | `04_correlation_heatmap.png` |
| Categorical patterns | `plot_categorical_vs_failure` | `05_categorical_vs_failure_type.png` |
| Duration vs severity | `plot_duration_vs_severity` | `06_duration_vs_severity.png` |
| Resource vs incident | `plot_resource_vs_incident` | `07_resource_usage_vs_incident.png` |
| Trends over time | `plot_temporal` | `08_temporal_patterns.png` |
| Context heatmaps | `plot_stage_branch_heatmaps` | `09_stage_branch_heatmaps.png` |
| Flaky / retry patterns | `plot_flaky_retry` | `10_flaky_retry_analysis.png` |

**How it helps:** Builds intuition for failure drivers (stage, branch, flaky tests, resources) before training the risk-tier model.

---

### 15. Feature engineering (as started in EDA)

| Syllabus item | In `eda.py`? | Details |
|---------------|--------------|---------|
| **Feature extraction** | Yes | `total_duration_sec`, `hour`, `day_of_week`, `date`, `sev_score` |
| **Feature selection** | Partial | Column lists define what EDA focuses on; formal selection is in training |
| **Feature scaling** | No | `StandardScaler` in `train_risk.py` |

**How it helps:** Extracted features make temporal and severity analysis possible; they preview fields later used (or extended) in `train_risk.py` (`add_model_features`).

---

### 16. Train–test split, cross-validation, model evaluation

**Not in `eda.py`.** These belong to the modeling pipeline:

- Train/test split & CV → `train_risk.py` / `train_v3.py` (`train_test_split`, `StratifiedKFold`)
- Confusion matrix, accuracy, precision, recall → metrics / CM plots under `results/`

**How the EDA still helps those steps:** clean labels, balanced class views, and correlation checks reduce leakage surprises and guide which features to feed the classifier.

---

## Unit VI — Data visualization (& regression note)

### 17. Charts used in `eda.py` (Matplotlib & Seaborn)

| Chart type | Syllabus name | Function | Why it helps here |
|------------|---------------|----------|-------------------|
| Horizontal / vertical **bar** | Bar chart | Targets, missingness, flaky rates, stacked hour bars | Counts & rates by category |
| **Histogram** | Histogram | Numeric dists; CPU/mem by incident | Shape, skew, overlap of groups |
| **Box plot** | Box plot | Duration vs severity | Median, IQR, outliers per class |
| **Line graph** | Line graph | Failures per day | Trend / seasonality over time |
| **Heatmap** | Core viz (2D matrix) | Correlation; stage×severity; branch×stage | Relationships in a grid |
| **Stacked bar** | Bar + composition | Categorical × failure type; hour × severity | Mix within a group |
| **Legend / titles / labels** | Chart legend | Every `plot_*` | Readable multi-series plots |

**3D visualization:** not used — 2D plots are enough for this tabular CI/CD EDA.

**Regression (linear / multiple):** not in `eda.py`. Correlation heatmaps show *association strength*, which is related but not the same as fitting a regression line. Regression-style modeling, if needed, lives in the training scripts / research summaries.

---

## End-to-end flow (how concepts connect)

```text
CSV data source
    → structural inspection (schema, shape)
    → data cleaning + quality report
    → outlier detection (IQR / domain)
    → transforms & encoding (types, sev_score, time parts)
    → measures of centre/spread (describe, mean/median, boxplots)
    → EDA visualizations (distribution, correlation, trends, groups)
    → cleaned CSV ready for train_risk.py (scaling, split, metrics)
```

---

## Concepts only lightly / not covered in `eda.py`

Use this checklist if you need full syllabus coverage in the project:

| Topic | Where to look instead |
|-------|------------------------|
| PCA | Not implemented in current risk training path |
| One-hot + StandardScaler | `train_risk.py` → `make_preprocessor` |
| Train/test split & Stratified K-Fold | `train_risk.py` |
| Confusion matrix / accuracy / F1 / etc. | `train_risk.py` + `results/` |
| Linear / multiple regression | Not the primary model (tree/boosting classifiers); see research docs if discussed |
| Merging multiple DataFrames | Single-source CSV — N/A for this EDA |
| Interactive `head()` / `info()` | Equivalent info printed via `print_profile`; run interactively in a notebook if needed |

---

## How to use this document

1. Run `python eda.py`.
2. Open plots under `plots/` and the audit under `cleaned/cleaning_report.json`.
3. For each syllabus bullet above, point to the **Where** section and the matching PNG or report step.
4. For modeling metrics and scaling, walk through `train_risk.py` after EDA.
