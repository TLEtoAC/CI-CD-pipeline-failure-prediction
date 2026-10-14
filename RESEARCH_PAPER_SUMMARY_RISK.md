# Research Paper Context: Pipeline Risk-Tier Prediction Only

Brief for drafting a paper about the risk-level model trained by `train_risk.py`. This experiment does not train rollback or incident classifiers. Figures are from that script: 80/20 split, seed 42, 36,000 train rows, 9,000 test rows. Metrics file: `results/metrics_risk.json`.

## Working title

Four-class CI/CD risk tiers from an operational index: regress-then-threshold versus direct classifiers

## Abstract points

A synthetic corpus of 45,000 CI/CD runs has native outcome tags that are independent of telemetry, so a risk label that hides one of its own inputs cannot exceed about 55% exact-match accuracy. This study defines an Operational Risk Index from measurements available at inference (CPU, memory, duration, retries, flaky tests, severity, stage, branch), cuts it into four ordered tiers on the training set only, and compares eight models. A LightGBM regressor of the continuous index, passed through the training quartile edges, reaches held-out accuracy **0.9747**, macro F1 **0.9746**, weighted one-vs-rest ROC AUC **0.9984**, and quadratic weighted kappa **0.9899**. Every error is a neighboring tier. Five-fold cross-validation on the training matrix is **0.9691 ± 0.0015**. The strongest direct classifier, histogram gradient boosting, scores 0.9452. Logistic regression scores 0.8246. The result is recovery of a declared policy, not a forecast of production incidents.

## Problem

CI dashboards need an ordered risk tier (LOW, MEDIUM, HIGH, CRITICAL) from a single run. On `ci_cd_pipeline_failure_logs_dataset.csv` the columns `failure_type`, `severity`, `rollback_triggered`, and `incident_created` have absolute Pearson correlation below 0.011 with durations, CPU, memory, retries, and the flaky flag, and mutual information near 0. An earlier risk score put severity inside the label and withheld it from the features. With severity uniform on {1, 2, 3, 4} and the other score terms observed, the Bayes exact-match accuracy of that protocol on this split is **0.5481**. A LightGBM reproduction scored **0.4783** accuracy and **0.7617** AUC (adjacent accuracy 0.8788). That ceiling is why this paper changes the target instead of searching more hyperparameters.

This script trains only the risk tier. Rollback and incident are omitted because those bits have no dependence on the features.

## Data

- File: `ci_cd_pipeline_failure_logs_dataset.csv`
- 45,000 rows, 25 columns, 0 missing values
- One row per `run_id` (45,000 unique). `pipeline_id` has 8,933 levels. No sequential label exists for a recurrent model.
- Unused identity fields: timestamp (41,282 distinct), repository (500), author (1,000), commit hash, error message (both unique per row), error code (900 levels)

Feature-side ranges and shares:

| Field | Distribution |
|---|---|
| CI tool | 5 levels, each about 20% (Jenkins, GitHub Actions, GitLab CI, Azure DevOps, CircleCI) |
| Language | 6 levels, each about 16–17% (Python, Go, NodeJS, C++, Rust, Java) |
| OS | ubuntu-latest 32.7%, windows-latest 33.6%, macos-latest 33.7% |
| Cloud | AWS, Azure, GCP, On-Prem, each about 25% |
| Branch | main, develop, feature-x, release, each about 25% |
| Failure stage | build, test, deploy, each about 33% |
| Build / test / deploy duration (s) | 30–3600 / 10–2400 / 0–1800 |
| CPU | 10–95% |
| Memory | 256–32,768 MB |
| Retry count | integers 0–5, each about 16.5–17.1% |
| Flaky test | 50.1% true |
| Severity | about 25% in each of LOW, MEDIUM, HIGH, CRITICAL |

Severity is an **input** to the index, matching the dashboard form. It is not the prediction target.

## Target definition

Ordinal map: LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4.

```
total = build_duration_sec + test_duration_sec + deploy_duration_sec
cpu_term   = max((cpu_usage_pct - 50) / 25, 0)
mem_term   = max((memory_usage_mb - 8000) / 8000, 0)
dur_term   = max((total - 2500) / 2500, 0)
retry_term = retry_count / 5
sev_term   = sev_score / 4
flaky      = is_flaky_test in {0, 1}
deploy     = 1 if failure_stage == "deploy" else 0

base = 1.35*cpu_term + 1.15*mem_term + 0.75*dur_term
     + 1.00*retry_term + 0.85*flaky + 1.70*sev_term
     + 0.90*cpu_term*retry_term
     + 0.60*flaky*sev_term
     + 0.45*mem_term*deploy

stage_mult  = 1.25 deploy, 1.08 test, 0.92 build
branch_mult = 1.18 if branch in {main, release} else 1.0
risk_index  = base * stage_mult * branch_mult
```

Training-set index (36,000 rows): mean 5.9676, standard deviation 2.6785, min 0.3910, max 20.1573.

### Split and tier edges

1. Split row indices first: `test_size=0.2`, `random_state=42`. Not stratified, because the label does not exist yet.
2. Edges are the training 25th, 50th, and 75th percentiles: **4.0013, 5.6217, 7.5633**. Test scores do not move the cuts.
3. Right-closed bins:

| Tier | Rule | Share of 45,000 rows |
|---|---|---|
| LOW | index ≤ 4.0013 | 0.2506 |
| MEDIUM | 4.0013 < index ≤ 5.6217 | 0.2496 |
| HIGH | 5.6217 < index ≤ 7.5633 | 0.2496 |
| CRITICAL | index > 7.5633 | 0.2502 |

Test support: LOW 2,275, MEDIUM 2,232, HIGH 2,234, CRITICAL 2,259. Majority-class test accuracy is **0.2528** (train majority class LOW). Uniform chance is 0.25.

Class order stored in the label encoder is LOW < MEDIUM < HIGH < CRITICAL. Alphabetical encoding would make quadratic weighted kappa invalid.

The clipped terms and the three policy products are not columns. The model has to approximate them.

## Features (52 after encoding)

Categorical, one-hot, unknown levels ignored, fit on train: `ci_tool`, `language`, `os`, `cloud_provider`, `branch`, `failure_stage`. Tool, language, OS, and cloud are controls. They are not arguments of the index.

Numeric, standard-scaled on train:

- `build_duration_sec`, `test_duration_sec`, `deploy_duration_sec`, `cpu_usage_pct`, `memory_usage_mb`, `retry_count`
- `sev_score`
- `total_duration_sec`
- `build_ratio`, `test_ratio`, `deploy_ratio` (duration / max(total, 1))
- `log1p` of build, test, deploy, memory, and total duration
- `cpu_x_retry`, `mem_x_retry`, `cpu_x_flaky`, `sev_x_retry`, `sev_x_flaky`

Boolean passthrough: `is_flaky_test`, `is_main_branch`, `is_release_branch`, `is_deploy_stage`, `is_test_stage`, `is_build_stage`.

`rollback_triggered` and `incident_created` are not features.

## Models and selection

Software: Python 3.12.7, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0. Seed 42 wherever the estimator accepts it. Selection rule: highest test accuracy.

| Model | Hyperparameters |
|---|---|
| Logistic regression | LBFGS, C=1.0, max_iter=2000, no class weights |
| Random forest | 350 trees, max_depth=18, min_samples_leaf=2, max_features=sqrt |
| Extra trees | 400 trees, unlimited depth, min_samples_leaf=2, max_features=sqrt |
| HistGradientBoosting | 300 iterations, learning_rate=0.08, max_depth=8, min_samples_leaf=20, l2=0.1, early stopping on an internal 12% slice, patience 25 |
| XGBoost | 500 trees, max_depth=6, learning_rate=0.06, subsample=0.85, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1, multi:softprob, tree_method=hist |
| LightGBM | 600 trees, learning_rate=0.05, num_leaves=63, max_depth=10, subsample=0.85, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1 |
| LightGBM tuned | 12-trial TPE search, 3-fold stratified macro-F1, search time 694.9 s, best CV macro-F1 0.9276. Params: n_estimators=600, learning_rate=0.080061445, num_leaves=32, max_depth=12, min_child_samples=52, subsample=0.763702, colsample_bytree=0.672730, reg_lambda=0.004031. Refit time on the full training matrix is the `train_time_sec` in the table |
| LGBM score regression | LightGBM regressor on the continuous index (700 trees, learning_rate=0.05, num_leaves=63, max_depth=10, subsample=0.85, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1), then `digitize` at the training edges |

`predict_proba` for the regressor is a softmax of distance from the predicted index to the four bin centers. It supports ranking (AUC). Its log loss (0.653) is not comparable to the classifiers' log loss (0.15–0.39), because those classifiers were trained on a proper scoring rule.

Cross-validation of the winner: 5 stratified folds on the 36,000 training rows. Each fold refits the regressor and its own quartile edges. The 9,000 test rows are untouched. Accuracy **0.9691 ± 0.0015**, macro F1 **0.9691 ± 0.0015**. The outer test score is slightly higher because that fit uses all 36,000 rows; each fold uses 28,800.

## Held-out results

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | QWK | Adjacent | Log loss | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LGBM score regression** | **0.9747** | **0.9746** | **0.9747** | **0.9984** | **0.9899** | **1.0000** | 0.6530 | 6.08 |
| HistGradientBoosting | 0.9452 | 0.9452 | 0.9454 | 0.9955 | 0.9781 | 1.0000 | 0.1703 | 7.16 |
| LightGBM tuned | 0.9438 | 0.9438 | 0.9440 | 0.9960 | 0.9775 | 1.0000 | 0.1503 | 11.15 |
| XGBoost | 0.9384 | 0.9385 | 0.9387 | 0.9949 | 0.9753 | 1.0000 | 0.2018 | 4.53 |
| LightGBM | 0.9381 | 0.9382 | 0.9384 | 0.9950 | 0.9752 | 1.0000 | 0.1605 | 20.14 |
| Extra trees | 0.8684 | 0.8678 | 0.8682 | 0.9760 | 0.9479 | 1.0000 | 0.5198 | 1.66 |
| Random forest | 0.8633 | 0.8636 | 0.8639 | 0.9763 | 0.9451 | 1.0000 | 0.4471 | 2.49 |
| Logistic regression | 0.8246 | 0.8246 | 0.8251 | 0.9655 | 0.9297 | 0.9999 | 0.3883 | 0.39 |

Adjacent accuracy 1.0 means the 2.53% of mistakes are all one tier away. The 12-trial search lifts flat LightGBM by 0.0057 accuracy and does not change the winner. Regress-then-threshold beats the best flat classifier by 0.0295 accuracy and raises kappa from 0.9781 to 0.9899. That head matches a target that is a thresholded continuous score.

Per-class test metrics, deployed model:

| Tier | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| LOW | 0.9890 | 0.9837 | 0.9863 | 2275 |
| MEDIUM | 0.9665 | 0.9686 | 0.9676 | 2232 |
| HIGH | 0.9620 | 0.9637 | 0.9629 | 2234 |
| CRITICAL | 0.9810 | 0.9823 | 0.9816 | 2259 |

Middle tiers are harder because more mass sits next to a cut.

### Feature importance

LightGBM split counts on the deployed regressor, top of the ranking: `memory_usage_mb` 5444, `cpu_usage_pct` 4682, `mem_x_retry` 2918, `cpu_x_retry` 2855, `cpu_x_flaky` 2760, `total_duration_sec` 2477, `sev_x_retry` 1966, `sev_score` 1652, `log_memory_usage_mb` 1652, `failure_stage_deploy` 1473. CI tool, language, OS, and cloud provider are absent from the top 15, which is the expected pattern for controls that do not enter the index. Deploy stage appears because deploy runs are multiplied by 1.25 and memory pressure is up-weighted on deploy.

Confusion matrix: `results/confusion_matrices/cm_risk_level_best.png`.

A separate ablation of this same regressor (same split, recorded in `results/methodology_v3.json` under `ablations_score_regression`) shows:

| Feature set | Encoded columns | Accuracy | Macro F1 | QWK | AUC |
|---|---:|---:|---:|---:|---:|
| Full | 52 | 0.9747 | 0.9746 | 0.9899 | 0.9984 |
| Raw telemetry + severity + flaky flag | 33 | 0.9724 | 0.9724 | 0.9890 | 0.9982 |
| Full set without severity columns | 49 | 0.7614 | 0.7618 | 0.9043 | 0.9389 |
| Raw telemetry, no severity | 32 | 0.7618 | 0.7622 | 0.9044 | 0.9387 |

Hand-built ratios and products add 0.0023 accuracy. Removing severity costs about 0.21. A scorer that must run before severity is known should be described at the 0.76 operating point, not at 0.97.

## Comparison with the withheld-severity protocol

| Protocol | Model | Accuracy | AUC | QWK | Adjacent |
|---|---|---:|---:|---:|---:|
| v1: severity inside the quartile label, excluded from features. Train-only edges 12.5548, 15.1890, 17.8108 | LightGBM | 0.4783 | 0.7617 | 0.6657 | 0.8788 |
| Bayes ceiling of that protocol (modal tier, severity uniform and hidden) | — | 0.5481 | — | — | 0.8461 |
| v3 ORI, this script | LGBM score regression | 0.9747 | 0.9984 | 0.9899 | 1.0000 |

The earlier four-model report (logistic regression 0.495 accuracy / 0.488 weighted F1 / 0.775 AUC; random forest 0.495 / 0.473 / 0.771; XGBoost 0.489 / 0.476 / 0.768; LightGBM 0.483 / 0.470 / 0.764) used full-sample quartile cuts on the v1 score. Cite it as the previous protocol. Cite 0.4783 / 0.7617 as the split-aligned reproduction.

## Deployment note

`train_risk.py` writes `models/best_model_risk_level.pkl`, `models/preprocessor_risk_level.pkl`, and `models/label_encoder_risk_level.pkl`. `predict.py` and the FastAPI service load those files. A long, hot, flaky deploy on `main` with severity HIGH scores CRITICAL. A short, cool build on a feature branch with severity LOW scores LOW.

## Discussion points that are supported

- The accuracy jump from ~0.48 to 0.9747 follows from giving the model the inputs the label actually depends on, and from regressing the continuous index before cutting tiers.
- Ordered metrics matter. Under the old protocol, adjacent accuracy was already 0.8788 while exact accuracy was 0.4783.
- A short hyperparameter search on flat LightGBM moved accuracy by less than a point. The decision head mattered more.
- Control columns with no role in the policy do not rank in the top features.

## Limitations to state

- The file is synthetic. Independence of the native tags is a measured property of this CSV.
- The index weights are specified, not learned from incidents. Accuracy is fidelity to that specification.
- One outer split (seed 42). The stability number is the 5-fold training CV, standard deviation 0.0015.
- Regression probabilities are geometric. Report accuracy, F1, kappa, and AUC for that model. Do not rank it by log loss.
- CatBoost and neural nets were not run.
- This paper's experiment does not claim a rollback or incident predictor.

## Claims to avoid

- Do not write that the model predicts production failures at 97% accuracy.
- Do not write that logistic regression is the best risk model. Under this protocol it is the weakest of the eight.
- Do not describe the index weights as learned coefficients. The learned object is the regressor that maps telemetry onto the resulting tiers.

## Suggested section order

1. Introduction: ordered CI risk, and the ceiling when a label hides an independent input.
2. Data and the independence audit.
3. Operational Risk Index, train-only quartile cuts, feature list.
4. Models, including regress-then-threshold, and the Optuna settings.
5. Results table, per-class scores, importance, severity ablation.
6. One paragraph on the saved artifact.
7. Limitations.
