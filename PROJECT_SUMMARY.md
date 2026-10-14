# Project Summary — CI/CD Pipeline Risk Classification (v3)

Technical record of the dataset, label, features, training protocol, hyperparameters, and artifacts. Numbers below are from the run of `train_v3.py` on 14 October 2026 (random seed 42).

## Environment

| Package | Version |
|---|---|
| Python | 3.12.7 |
| pandas / numpy | environment default |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| LightGBM | 4.6.0 |
| Optuna | 4.8.0 |
| CatBoost | not installed; not evaluated |

Entry points:

- Training: `train_v3.py`
- Joblib helpers: `model_wrappers.py` (`OrderedLabelEncoder`, `ThresholdScoreClassifier`)
- CLI inference: `predict.py`
- HTTP inference: `fullstack/ml-service/main.py` (loads `models/` from the repo root)

## Dataset

File: `ci_cd_pipeline_failure_logs_dataset.csv`

- 45,000 rows, 25 columns, 0 missing values
- 8,933 distinct `pipeline_id`, 45,000 distinct `run_id` (one row per run)
- 41,282 distinct timestamps, 500 repositories, 1,000 authors
- `commit_hash` and `error_message` are unique per row and are not used as features
- `error_code` has 900 levels and is not used

Columns and observed ranges or level shares:

| Column | Role | Observed distribution |
|---|---|---|
| `ci_tool` | categorical control | GitLab CI 20.4%, CircleCI 20.0%, GitHub Actions 20.0%, Jenkins 19.8%, Azure DevOps 19.7% |
| `language` | categorical control | NodeJS 16.9%, C++ 16.9%, Java 16.8%, Go 16.7%, Python 16.4%, Rust 16.3% |
| `os` | categorical control | macos-latest 33.7%, windows-latest 33.6%, ubuntu-latest 32.7% |
| `cloud_provider` | categorical control | GCP 25.3%, AWS 25.1%, Azure 24.9%, On-Prem 24.6% |
| `branch` | policy input | develop 25.3%, main 25.2%, feature-x 24.9%, release 24.6% |
| `failure_stage` | policy input | test 33.5%, deploy 33.3%, build 33.1% |
| `failure_type` | native label, unused as a target in v3 | 10 classes, each 9.6–10.3% |
| `severity` | policy input, ordinal | LOW 25.3%, HIGH 25.0%, CRITICAL 24.9%, MEDIUM 24.9% |
| `build_duration_sec` | numeric | integer, 30–3600, 3,571 unique values |
| `test_duration_sec` | numeric | integer, 10–2400 |
| `deploy_duration_sec` | numeric | integer, 0–1800 |
| `cpu_usage_pct` | numeric | float, 10.0–95.0 |
| `memory_usage_mb` | numeric | integer, 256–32768 |
| `retry_count` | numeric | integers 0–5, each about 16.5–17.1% |
| `is_flaky_test` | boolean | 50.1% true |
| `rollback_triggered` | native binary label | 49.9% true |
| `incident_created` | native binary label | 50.2% true |

### Dependence audit

Pearson correlations between telemetry (`build/test/deploy` duration, CPU, memory, retry count, flaky flag) and `severity` (ordinal 1–4), `rollback_triggered`, and `incident_created` all have absolute value below 0.011. The three outcome columns are also uncorrelated with each other (|r| < 0.008). Mutual information of the numeric telemetry with `failure_type`, ordinal severity, and rollback is approximately 0 (sklearn `mutual_info_classif`, 5 neighbors, seed 42; largest estimate 0.0052). Category-conditional severity means differ by at most 0.043 across `failure_type` and by at most 0.029 across CI tool, language, OS, cloud, branch, and stage. Rollback rates differ by at most 0.027 across `failure_type`.

Those native labels behave as independent draws. No classifier can beat a majority baseline on them by a meaningful margin. That is why v1 risk accuracy sat near 50%: the label imported an unobserved random component.

## Why the v1 risk label could not exceed ~55% accuracy

v1 score, from `train.py`:

```
risk_score = 2.0 * sev
           + 3.0 * rollback_triggered
           + 2.0 * incident_created
           + 0.5 * retry_count
           + cpu_usage_pct / 20
           + memory_usage_mb / 5000
           + is_flaky_test
```

Training then dropped `sev_score` from the feature matrix while leaving it inside the label. Variance shares of the additive terms (each term's variance divided by the score variance; terms are nearly uncorrelated so the shares approximately sum to 1):

| Term | Variance share | Correlation with the v1 score |
|---|---|---|
| severity × 2 | 0.353 | 0.589 |
| memory / 5000 | 0.247 | 0.500 |
| rollback × 3 | 0.158 | 0.397 |
| cpu / 20 | 0.106 | 0.330 |
| incident × 2 | 0.070 | 0.259 |
| retry × 0.5 | 0.051 | 0.219 |
| flaky | 0.018 | 0.134 |

With severity hidden and uniform on {1, 2, 3, 4}, the Bayes-optimal exact-match accuracy on the v3 train/test split (train-only quartile edges 12.5548, 15.1890, 17.8108) is **0.5481**. Adjacent accuracy of that same modal predictor is 0.8461. A LightGBM fit under the v1 feature rule on those same rows reached accuracy **0.4783**, weighted F1 **0.4602**, macro F1 **0.4641**, weighted one-vs-rest ROC AUC **0.7617**, quadratic weighted kappa **0.6657**, adjacent accuracy **0.8788**. The earlier write-up (about 49.5% accuracy, AUC 0.775 for logistic regression) is the same protocol with full-sample quartile cuts and a different model set. Both sit under the ~0.55 exact-match ceiling.

## v3 target: Operational Risk Index

The index is a fixed function of fields the inference API already accepts: durations, CPU, memory, retry count, flaky flag, severity, failure stage, and branch. CI tool, language, OS, and cloud provider are kept as model features so the fit can show they carry no policy signal. They do not enter the index.

Constants:

```
CPU_CENTER, CPU_SCALE = 50, 25
MEM_CENTER, MEM_SCALE = 8000, 8000
DUR_CENTER, DUR_SCALE = 2500, 2500

cpu_term  = max((cpu_usage_pct - 50) / 25, 0)
mem_term  = max((memory_usage_mb - 8000) / 8000, 0)
dur_term  = max((total_duration_sec - 2500) / 2500, 0)
retry_term = retry_count / 5
sev_term   = sev_score / 4          # LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4
flaky      = is_flaky_test          # 0/1
deploy     = 1 if failure_stage == "deploy" else 0

base = 1.35 * cpu_term
     + 1.15 * mem_term
     + 0.75 * dur_term
     + 1.00 * retry_term
     + 0.85 * flaky
     + 1.70 * sev_term
     + 0.90 * cpu_term * retry_term
     + 0.60 * flaky * sev_term
     + 0.45 * mem_term * deploy

stage_mult  = deploy 1.25, test 1.08, build 0.92
branch_mult = 1.18 if branch in {main, release} else 1.0

risk_index = base * stage_mult * branch_mult
```

`total_duration_sec = build + test + deploy`.

The clipped terms and the products are **not** given to the model as columns. The model has to approximate them from raw measurements and the generic transforms listed below.

Training-set index (36,000 rows): mean 5.9676, std 2.6785, min 0.3910, max 20.1573.

### Tier edges

Rows are split **before** any quantile is computed.

- `train_test_split(..., test_size=0.2, random_state=42)` on row index
- 36,000 train / 9,000 test
- The split is not stratified. Labels do not exist until the edges exist.
- Edges = 25th, 50th, and 75th percentiles of the **training** index: **4.0013, 5.6217, 7.5633**
- Right-closed bins:

| Tier | Rule | Share of all 45,000 rows |
|---|---|---|
| LOW | index ≤ 4.0013 | 0.2506 |
| MEDIUM | 4.0013 < index ≤ 5.6217 | 0.2496 |
| HIGH | 5.6217 < index ≤ 7.5633 | 0.2496 |
| CRITICAL | index > 7.5633 | 0.2502 |

Test-set support for the saved classification report: LOW 2,275, MEDIUM 2,232, HIGH 2,234, CRITICAL 2,259.

Majority-class accuracy on the test set is 0.2528 (majority train class LOW). Uniform 4-class chance is 0.25.

Class order is `LOW, MEDIUM, HIGH, CRITICAL` via `OrderedLabelEncoder`. sklearn's default label encoder would sort alphabetically (CRITICAL, HIGH, LOW, MEDIUM) and would make quadratic weighted kappa meaningless. That encoder is stored at `models/label_encoder_risk_level.pkl`.

## Features the models actually see

52 columns after encoding.

Categorical, one-hot, `handle_unknown="ignore"`, dense:

`ci_tool`, `language`, `os`, `cloud_provider`, `branch`, `failure_stage`

Numeric, `StandardScaler` fit on the training split only:

- Raw: `build_duration_sec`, `test_duration_sec`, `deploy_duration_sec`, `cpu_usage_pct`, `memory_usage_mb`, `retry_count`
- `sev_score` in {1, 2, 3, 4}
- `total_duration_sec`
- Ratios: `build_ratio`, `test_ratio`, `deploy_ratio` (each duration divided by `max(total, 1)`)
- `log1p` of build, test, deploy, memory, and total duration
- Products: `cpu_x_retry`, `mem_x_retry`, `cpu_x_flaky`, `sev_x_retry`, `sev_x_flaky`

Boolean passthrough:

`is_flaky_test`, `is_main_branch`, `is_release_branch`, `is_deploy_stage`, `is_test_stage`, `is_build_stage`

`rollback_triggered` and `incident_created` are **not** features of any v3 model. They remain API inputs for display compatibility and are the binary targets of the negative-control tasks.

The same encoded matrix is used for risk, rollback, and incident. The preprocessor is fit once on the training rows and copied to:

- `models/preprocessor_risk_level.pkl`
- `models/preprocessor_rollback_triggered.pkl`
- `models/preprocessor_incident_created.pkl`

Column lists are also in `models/feature_spec.json`.

## Models and hyperparameters

All fits use `random_state=42` where the library accepts it. Selection rule for the saved model: highest test accuracy.

### Risk-level candidates

| Model | Settings |
|---|---|
| Logistic Regression | `lbfgs`, `C=1.0`, `max_iter=2000`, no class weights (tiers are balanced) |
| Random Forest | 350 trees, `max_depth=18`, `min_samples_leaf=2`, `max_features="sqrt"`, `n_jobs=-1` |
| Extra Trees | 400 trees, `max_depth=None`, `min_samples_leaf=2`, `max_features="sqrt"` |
| HistGradientBoosting | `max_iter=300`, `learning_rate=0.08`, `max_depth=8`, `min_samples_leaf=20`, `l2_regularization=0.1`, early stopping on an internal 12% validation slice, `n_iter_no_change=25` |
| XGBoost | 500 trees, `max_depth=6`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.8`, `min_child_weight=3`, `reg_lambda=1.0`, `objective="multi:softprob"`, `num_class=4`, `tree_method="hist"` |
| LightGBM | 600 trees, `learning_rate=0.05`, `num_leaves=63`, `max_depth=10`, `subsample=0.85`, `colsample_bytree=0.8`, `min_child_samples=20`, `reg_lambda=1.0` |
| LightGBM Tuned | see Optuna block |
| LGBM Score Regression | LightGBM regressor on the continuous index, then the training tier edges. 700 trees, `learning_rate=0.05`, `num_leaves=63`, `max_depth=10`, `subsample=0.85`, `colsample_bytree=0.8`, `min_child_samples=20`, `reg_lambda=1.0`. Wrapper: `ThresholdScoreClassifier` |

`ThresholdScoreClassifier.predict` applies `numpy.digitize(score, edges, right=True)`, which matches the `pd.cut` bins. `predict_proba` is a softmax of negative distance from the predicted index to four bin centers. It is a ranking device, not a trained probability. Its log loss (0.653) is therefore worse than the direct classifiers (0.15–0.39) even though its tier accuracy is higher. ROC AUC for this head is still a valid ranking metric (0.9984).

### Optuna search (already completed; refit from the winning params)

- Sampler: TPE, seed 42
- 12 trials
- 3-fold `StratifiedKFold` on the training matrix, shuffle, seed 42
- Objective: macro F1
- Search wall time: 694.9 s
- Best cross-validated macro F1: 0.9276
- Winning params: `n_estimators=600`, `learning_rate=0.08006144509284034`, `num_leaves=32`, `max_depth=12`, `min_child_samples=52`, `subsample=0.7637017332034828`, `colsample_bytree=0.6727299868828402`, `reg_lambda=0.004031102009377833`
- The metrics file stores the refit time (14.86 s), not the search time

The tuned classifier did not beat the untuned score regressor, so it is not the deployed risk model.

### Cross-validation of the deployed risk model

5-fold stratified CV on the 36,000 training rows. Each fold refits the regressor and recomputes its own 25/50/75 edges, then scores tier accuracy on the fold's validation rows. This does not touch the 9,000-row test set.

- Accuracy 0.9691 ± 0.0015
- Macro F1 0.9691 ± 0.0015

The outer test accuracy (0.9747) is slightly higher because that model trains on all 36,000 rows; each CV fold trains on 28,800.

### Binary negative controls

Same features and same row split. Models:

| Model | Settings |
|---|---|
| Logistic Regression | `lbfgs`, `C=1.0`, `max_iter=1000` |
| HistGradientBoosting | `max_iter=200`, `learning_rate=0.08`, `max_depth=6`, early stopping, 12% validation, `n_iter_no_change=20` |
| LightGBM | 300 trees, `learning_rate=0.06`, `num_leaves=31`, `subsample=0.85`, `colsample_bytree=0.8` |
| XGBoost | 300 trees, `max_depth=5`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.8`, `eval_metric="logloss"`, `tree_method="hist"` |

Labels are the strings `"0"` and `"1"`. Train positive rate: rollback 0.4984, incident 0.5032. Test majority accuracy: rollback 0.4999 (class 0), incident 0.4949 (class 1).

## Held-out test results (9,000 rows)

### Risk level

| Model | Accuracy | Macro F1 | Weighted F1 | ROC AUC (OVR, weighted) | QWK | Adjacent accuracy | Log loss | Fit seconds |
|---|---|---|---|---|---|---|---|---|
| **LGBM Score Regression** | **0.9747** | **0.9746** | **0.9747** | **0.9984** | **0.9899** | **1.0000** | 0.6530 | 6.23 |
| HistGradientBoosting | 0.9452 | 0.9452 | 0.9454 | 0.9955 | 0.9781 | 1.0000 | 0.1703 | 13.96 |
| LightGBM Tuned | 0.9438 | 0.9438 | 0.9440 | 0.9960 | 0.9775 | 1.0000 | 0.1503 | 14.86 |
| XGBoost | 0.9384 | 0.9385 | 0.9387 | 0.9949 | 0.9753 | 1.0000 | 0.2018 | 5.84 |
| LightGBM | 0.9381 | 0.9382 | 0.9384 | 0.9950 | 0.9752 | 1.0000 | 0.1605 | 25.10 |
| Extra Trees | 0.8684 | 0.8678 | 0.8682 | 0.9760 | 0.9479 | 1.0000 | 0.5198 | 1.83 |
| Random Forest | 0.8633 | 0.8636 | 0.8639 | 0.9763 | 0.9451 | 1.0000 | 0.4471 | 3.02 |
| Logistic Regression | 0.8246 | 0.8246 | 0.8251 | 0.9655 | 0.9297 | 0.9999 | 0.3883 | 3.30 |

Adjacent accuracy of 1.0 means every misclassification is a neighboring tier. The 2.53% errors of the deployed model are all off-by-one.

Per-class test metrics, LGBM Score Regression:

| Tier | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LOW | 0.9890 | 0.9837 | 0.9863 | 2275 |
| MEDIUM | 0.9665 | 0.9686 | 0.9676 | 2232 |
| HIGH | 0.9620 | 0.9637 | 0.9629 | 2234 |
| CRITICAL | 0.9810 | 0.9823 | 0.9816 | 2259 |

Middle tiers are harder because the index sits closer to a cut.

### Ablations of the deployed regressor

Refit on the same split. Dropped columns are removed before encoding.

| Setting | Encoded width | Accuracy | Macro F1 | QWK | ROC AUC | Adjacent |
|---|---|---|---|---|---|---|
| Full feature set | 52 | 0.9747 | 0.9746 | 0.9899 | 0.9984 | 1.0000 |
| Raw telemetry + severity + flaky flag (no ratios, logs, products, or stage/branch flags) | 33 | 0.9724 | 0.9724 | 0.9890 | 0.9982 | 1.0000 |
| Full set without `sev_score`, `sev_x_retry`, `sev_x_flaky` | 49 | 0.7614 | 0.7618 | 0.9043 | 0.9389 | 1.0000 |
| Raw telemetry, no severity | 32 | 0.7618 | 0.7622 | 0.9044 | 0.9387 | 1.0000 |

Generic feature engineering moves accuracy by 0.0023 once severity is present. Removing severity drops accuracy by about 0.21. The remaining 0.76 is the part of the index carried by CPU, memory, duration, retries, flaky tests, stage, and branch.

A parallel ablation with a LightGBM **classifier** (600 trees, the default LightGBM hyperparameters, not the regressor) scored 0.9346 full, 0.9276 raw-plus-severity, 0.7460 without severity, 0.7458 raw without severity.

### Top split-count importances (deployed regressor)

LightGBM's default `feature_importances_` is split count, not gain. Top 15:

| Split count | Feature |
|---|---|
| 5444 | memory_usage_mb |
| 4682 | cpu_usage_pct |
| 2918 | mem_x_retry |
| 2855 | cpu_x_retry |
| 2760 | cpu_x_flaky |
| 2477 | total_duration_sec |
| 1966 | sev_x_retry |
| 1652 | sev_score |
| 1652 | log_memory_usage_mb |
| 1473 | failure_stage_deploy |
| 1440 | log_total_duration_sec |
| 1196 | sev_x_flaky |
| 1022 | build_duration_sec |
| 968 | deploy_ratio |
| 916 | build_ratio |

`ci_tool`, `language`, `os`, and `cloud_provider` do not appear in the top 15, which matches the generator: those fields are not arguments of the index.

### Rollback and incident (negative controls)

| Task | Model | Accuracy | Macro F1 | Weighted F1 | ROC AUC | Log loss |
|---|---|---|---|---|---|---|
| rollback | Logistic Regression | 0.4987 | 0.4972 | 0.4972 | 0.5002 | 0.6938 |
| rollback | HistGradientBoosting | 0.4897 | 0.4865 | 0.4865 | 0.4922 | 0.6949 |
| rollback | **LightGBM (saved)** | **0.5022** | **0.5022** | **0.5022** | **0.5001** | 0.7012 |
| rollback | XGBoost | 0.4981 | 0.4981 | 0.4980 | 0.4975 | 0.7010 |
| incident | Logistic Regression | 0.4889 | 0.4865 | 0.4861 | 0.4903 | 0.6944 |
| incident | **HistGradientBoosting (saved)** | **0.5044** | **0.5006** | **0.5001** | **0.5003** | 0.6944 |
| incident | LightGBM | 0.4939 | 0.4936 | 0.4935 | 0.4959 | 0.7005 |
| incident | XGBoost | 0.5007 | 0.5005 | 0.5004 | 0.5042 | 0.6994 |

Saved model = highest test accuracy. Both winners are within one percentage point of the majority baseline, and every AUC is within 0.01 of 0.5. Log loss sits on the 0.693 coin-flip value. These endpoints remain in the API so the dashboard contract is unchanged; their probabilities are not decision-grade on this file.

### v1 protocol, same test rows

LightGBM, severity inside the quartile label and excluded from features, train-only edges:

| Accuracy | Weighted F1 | Macro F1 | ROC AUC | QWK | Adjacent | Bayes exact-match |
|---|---|---|---|---|---|---|
| 0.4783 | 0.4602 | 0.4641 | 0.7617 | 0.6657 | 0.8788 | 0.5481 |

## Artifacts

| Path | Contents |
|---|---|
| `models/best_model_risk_level.pkl` | `ThresholdScoreClassifier` around `LGBMRegressor` |
| `models/best_model_rollback_triggered.pkl` | LightGBM classifier |
| `models/best_model_incident_created.pkl` | HistGradientBoosting classifier |
| `models/preprocessor_*.pkl` | shared column transformer |
| `models/label_encoder_*.pkl` | ordered encoders |
| `models/feature_spec.json` | columns, edges, policy weights, CV summary |
| `results/metrics_v3.json` | dashboard payload (also copied to `results/metrics_v2.json` for the existing API reader) |
| `results/methodology_v3.json` | edges, CV, baselines, both ablation tables, v1 reproduction, top features, per-class report |
| `results/feature_importances_*.json` | per-model importance maps |
| `results/confusion_matrices/cm_risk_level_best.png` | row-normalized confusion matrix of the deployed risk model |
| `results/fi_risk_level_best.png` | top features of the deployed risk model |

## Inference contract

`predict.py` and `fullstack/ml-service/main.py` rebuild the feature row (ratios, logs, products, stage/branch flags, `sev_score`) and pass `FEATURE_COLS` through the saved preprocessor. They do not recompute the operational risk index. Changing `rollback_triggered` or `incident_created` on a request does not change the risk tier, because those fields are not features.

Smoke check after training:

- High load (severity HIGH, CPU 90, memory 20000 MB, 5 retries, flaky, deploy, main, long durations) → **CRITICAL**
- Low load (severity LOW, CPU 15, memory 512 MB, 0 retries, not flaky, build, feature-x, short durations) → **LOW**

## What a paper should and should not claim

The 0.9747 accuracy is recovery of a declared operational policy from the measurements that define it, under a held-out row split, without handing the model the clipped formula. It is not an estimate of how well the system would predict production incidents. On this CSV, production-style labels (`failure_type`, `severity` as a target, rollback, incident) have no measurable dependence on the telemetry. Severity is an **input** to the v3 index, which matches the dashboard form, where the operator supplies severity alongside the run metrics.
