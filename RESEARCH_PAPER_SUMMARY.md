# Research Paper Context: CI/CD Pipeline Risk Classification

This file is a self-contained brief for drafting a research paper. Every number is from the v3 experiment unless a sentence names the earlier v1 write-up. Do not invent metrics, datasets, or baselines beyond what is written here. The companion file `PROJECT_SUMMARY.md` holds formulas, hyperparameters, and file paths.

## Working title

Recovering an operational CI/CD risk policy from pipeline telemetry: a controlled study on a synthetic log corpus with uncorrelated native labels

## What the study actually shows

A 45,000-row synthetic CI/CD log table has native outcomes (`failure_type`, `severity`, `rollback_triggered`, `incident_created`) that are statistically independent of the run telemetry. A four-class risk label that hides severity inside the target and withholds it from the features has an exact-match Bayes ceiling of about 0.55. Models in that protocol land near 0.48–0.50 accuracy with ROC AUC about 0.76.

Replacing that target with an explicit Operational Risk Index (ORI), computed only from fields available at inference time, and training a LightGBM regressor that is then cut at training-set quartiles, yields:

- Held-out accuracy **0.9747**
- Macro F1 **0.9746**
- Weighted one-vs-rest ROC AUC **0.9984**
- Quadratic weighted kappa **0.9899**
- Adjacent-tier accuracy **1.0000** (every error is off by one tier)
- 5-fold training CV accuracy **0.9691 ± 0.0015**

The same models on the native rollback and incident bits stay at accuracy ≈ 0.50 and AUC ≈ 0.50.

The accuracy gain is policy recovery, not discovery of a hidden failure mechanism in the synthetic outcomes.

## Abstract points (use these figures)

- Corpus: 45,000 synthetic pipeline runs, 25 columns, no missing values.
- Diagnosis: absolute Pearson correlation between telemetry and severity, rollback, and incident is below 0.011. Mutual information is approximately 0.
- Prior protocol ceiling: with severity latent and uniform on four levels, Bayes exact-match accuracy is **0.5481** on the evaluation split. A reproduced LightGBM under that protocol scores **0.4783** accuracy and **0.7617** AUC.
- Proposed target: a nonlinear ORI over CPU, memory, duration, retries, flaky tests, severity, stage, and branch. Tiers are training-set quartiles (edges 4.0013, 5.6217, 7.5633), applied unchanged to the test set.
- Model comparison on 9,000 held-out runs: logistic regression 0.8246, random forest 0.8633, extra trees 0.8684, XGBoost 0.9384, LightGBM 0.9381, tuned LightGBM 0.9438, histogram gradient boosting 0.9452, LightGBM index regression then tier cuts **0.9747**.
- Ablation of the winning regressor: dropping hand-built ratios and products costs 0.0023 accuracy (0.9747 → 0.9724). Dropping severity costs about 0.21 accuracy (0.9747 → 0.7614).
- Negative controls: best rollback model accuracy 0.5022 (AUC 0.5001) against a 0.4999 majority baseline; best incident model accuracy 0.5044 (AUC 0.5003) against a 0.4949 majority baseline.

## Introduction — problem framing

CI systems emit a mix of configuration context (tool, language, OS, cloud, branch), resource traces (CPU, memory, stage durations, retries), and outcome tags (failure class, severity, rollback, incident). A monitoring product that treats every tag as a supervised target will look accurate only when the tag is a function of the inputs the model is allowed to see.

On this corpus the outcome tags are not functions of the inputs. Publishing a 50% four-class accuracy without the ceiling overstates how much signal was left on the table, and it understates how much of the label was an independent coin flip. The useful supervised problem is: given the measurements an operator already types into the dashboard, reproduce a declared risk policy, and state clearly which inputs the policy requires.

Motivation to keep: failed or risky pipelines delay release. A tier (LOW / MEDIUM / HIGH / CRITICAL) is an action scale (proceed, watch, gate, stop). The paper's contribution is the measurement design and the model comparison under that design, plus a negative result on the native bits.

## Data

Source file: `ci_cd_pipeline_failure_logs_dataset.csv`.

Identity columns not used as features: `pipeline_id` (8,933 levels), `run_id` (unique), `timestamp` (41,282 distinct), `repository` (500), `author` (1,000), `commit_hash` (unique), `error_message` (unique), `error_code` (900). Each row is one run. There is no ordered failure sequence to model with an RNN; `run_id` does not repeat.

Context fields are close to uniform:

- CI tool, five levels, each about 20%
- Language, six levels, each about 16–17%
- OS, three `*-latest` images, each about 33%
- Cloud, four levels, each about 25%
- Branch: main, develop, feature-x, release, each about 25%
- Failure stage: build, test, deploy, each about 33%

Numeric spans: build duration 30–3600 s, test 10–2400 s, deploy 0–1800 s, CPU 10–95%, memory 256–32768 MB, retry count integers 0 through 5 at roughly equal frequency. Flaky tests, rollbacks, and incidents are each about 50%. Severity is about 25% in each of LOW, MEDIUM, HIGH, CRITICAL. Failure type has ten classes, each about 10% (Network Error, Security Scan Failure, Resource Exhaustion, Dependency Error, Build Failure, Deployment Failure, Test Failure, Permission Error, Configuration Error, Timeout).

### Independence evidence (put in a table or appendix)

- Max |Pearson r| of durations, CPU, memory, retries, and the flaky flag with ordinal severity, rollback, or incident: **< 0.011**
- |r| among severity, rollback, and incident: **< 0.008**
- Mutual information (5-neighbor estimate) of those numerics with failure type, severity, and rollback: all ≤ **0.0052**, most estimates 0
- Spread of mean severity across a categorical field: at most 0.043 (failure type) and at most 0.029 for tool, language, OS, cloud, branch, and stage
- Spread of rollback rate across failure type: 0.027

## Methods

### Prior label (v1), reproduced on the v3 split

```
v1_score = 2·severity + 3·rollback + 2·incident + 0.5·retries
         + cpu/20 + memory/5000 + flaky
```

Quartile tiers. Features include rollback, incident, retries, CPU, memory, durations, and flaky tests. Features exclude severity. Severity is the largest term (about 35% of score variance) and is independent of everything else, so the label contains an irreducible four-way noise source.

Reproduction: LightGBM, same 36k/9k split, quartile edges fit on train only (12.5548, 15.1890, 17.8108).

| Accuracy | Weighted F1 | Macro F1 | AUC | QWK | Adjacent accuracy | Bayes exact-match |
|---|---|---|---|---|---|---|
| 0.4783 | 0.4602 | 0.4641 | 0.7617 | 0.6657 | 0.8788 | 0.5481 |

The earlier report (logistic regression accuracy 0.495, weighted F1 0.488, AUC 0.775; random forest 0.495 / 0.473 / 0.771; XGBoost 0.489 / 0.476 / 0.768; LightGBM 0.483 / 0.470 / 0.764) used full-sample quartile cuts and did not report quadratic weighted kappa. Cite it as the previous protocol, and cite 0.4783 / 0.7617 as the split-aligned reproduction. The Bayes number 0.5481 is the modal tier when severity is redrawn uniformly from {1, 2, 3, 4} and the other v1 terms are known. Adjacent accuracy of that mode is 0.8461; the mode maximizes exact match, so a different decision rule can score higher on the adjacent metric (the reproduction's adjacent accuracy is 0.8788).

### Operational Risk Index (v3)

Ordinal severity map: LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4.

```
cpu_term   = max((cpu - 50) / 25, 0)
mem_term   = max((memory_mb - 8000) / 8000, 0)
dur_term   = max((total_duration_sec - 2500) / 2500, 0)
retry_term = retries / 5
sev_term   = sev_score / 4

base = 1.35·cpu_term + 1.15·mem_term + 0.75·dur_term
     + 1.00·retry_term + 0.85·flaky + 1.70·sev_term
     + 0.90·cpu_term·retry_term
     + 0.60·flaky·sev_term
     + 0.45·mem_term·1[stage = deploy]

risk_index = base · stage_mult · branch_mult
stage_mult  = 1.25 deploy, 1.08 test, 0.92 build
branch_mult = 1.18 if branch ∈ {main, release} else 1.0
```

Training index: mean 5.9676, standard deviation 2.6785, min 0.3910, max 20.1573.

Protocol details that belong in the paper:

1. Split first (20% test, seed 42, 36,000 / 9,000). Do not stratify, because the label is defined from training quantiles.
2. Tier edges are the training 25/50/75 percentiles: **4.0013, 5.6217, 7.5633**. Right-closed. Test rows never influence the cuts.
3. The model does not receive `cpu_term`, `mem_term`, `dur_term`, or the three policy products. It receives raw measurements, `sev_score`, duration ratios, `log1p` transforms, generic products (`cpu×retry`, `memory×retry`, `cpu×flaky`, `sev×retry`, `sev×flaky`), and stage/branch indicators, plus one-hot tool, language, OS, and cloud as controls.
4. Preprocessing fit on train only: one-hot with unknown levels ignored, standard scaler on numerics, booleans unchanged. 52 encoded columns.
5. Metrics: accuracy, macro and weighted F1, weighted one-vs-rest ROC AUC, log loss, quadratic weighted kappa, adjacent accuracy (share of predictions within one tier). Class order for kappa is LOW < MEDIUM < HIGH < CRITICAL.
6. Model choice: maximum test accuracy. Majority baseline on this split: 0.2528.

### Model families

Linear baseline (logistic regression, LBFGS, C=1). Bagged trees (random forest, 350 trees, depth 18; extra trees, 400 trees). Histogram gradient boosting (300 iterations, learning rate 0.08, depth 8, internal early stopping). XGBoost (500 trees, depth 6, learning rate 0.06, hist). LightGBM (600 trees, 63 leaves, learning rate 0.05). A 12-trial TPE search (Optuna, 3-fold macro F1, 694.9 s, best CV macro F1 0.9276) produced a second LightGBM: 600 trees, learning rate 0.0801, 32 leaves, depth 12, min child samples 52, subsample 0.764, column sample 0.673, L2 0.00403.

The winning method is indirect classification: regress the continuous index with LightGBM (700 trees, learning rate 0.05, 63 leaves, depth 10), then apply the training edges. Probabilities reported by that head are a softmax over distance to bin centers. Use them for ranking (AUC 0.9984). Do not compare that head's log loss (0.653) with the classifiers' log loss (0.15–0.39); the classifiers were trained on a proper scoring rule and the regressor was not.

CatBoost was unavailable in the training environment and is not a result.

### Validation

Outer test metrics are one split. Stability of the winner: 5-fold CV on the training matrix only, with tier edges refit inside each fold, accuracy 0.9691 ± 0.0015, macro F1 0.9691 ± 0.0015.

## Results to include

### Main comparison (test n = 9,000)

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | QWK | Adjacent | Log loss | Train seconds |
|---|---|---|---|---|---|---|---|---|
| LGBM score regression | 0.9747 | 0.9746 | 0.9747 | 0.9984 | 0.9899 | 1.0000 | 0.6530 | 6.23 |
| HistGradientBoosting | 0.9452 | 0.9452 | 0.9454 | 0.9955 | 0.9781 | 1.0000 | 0.1703 | 13.96 |
| LightGBM tuned | 0.9438 | 0.9438 | 0.9440 | 0.9960 | 0.9775 | 1.0000 | 0.1503 | 14.86 |
| XGBoost | 0.9384 | 0.9385 | 0.9387 | 0.9949 | 0.9753 | 1.0000 | 0.2018 | 5.84 |
| LightGBM | 0.9381 | 0.9382 | 0.9384 | 0.9950 | 0.9752 | 1.0000 | 0.1605 | 25.10 |
| Extra trees | 0.8684 | 0.8678 | 0.8682 | 0.9760 | 0.9479 | 1.0000 | 0.5198 | 1.83 |
| Random forest | 0.8633 | 0.8636 | 0.8639 | 0.9763 | 0.9451 | 1.0000 | 0.4471 | 3.02 |
| Logistic regression | 0.8246 | 0.8246 | 0.8251 | 0.9655 | 0.9297 | 0.9999 | 0.3883 | 3.30 |

Reading the table: every nonlinear model is far above the 0.25 chance line. Logistic regression, which cannot represent the rectified terms and the stage/branch multipliers unless they are pre-built, is the weakest at 0.8246 and still has QWK 0.9297 because its mistakes are almost all adjacent. Gradient boosting sits in a band from 0.938 to 0.945. Mapping a regression of the index through the same cuts gains another three points, to 0.9747, which is the right inductive bias for a thresholded continuous score. The 12-trial search improved LightGBM classification from 0.9381 to 0.9438 and did not change the winner.

Per-class F1 of the winner: LOW 0.9863 (support 2275), MEDIUM 0.9676 (2232), HIGH 0.9629 (2234), CRITICAL 0.9816 (2259). Precision and recall for LOW are 0.9890 and 0.9837; for CRITICAL 0.9810 and 0.9823; the middle tiers sit between 0.962 and 0.969. Errors concentrate at the cuts.

### Ablation (winning regressor, refit)

| Feature set | Encoded columns | Accuracy | Macro F1 | QWK | AUC |
|---|---|---|---|---|---|
| Full | 52 | 0.9747 | 0.9746 | 0.9899 | 0.9984 |
| Raw measurements, severity, flaky flag only | 33 | 0.9724 | 0.9724 | 0.9890 | 0.9982 |
| Full, severity columns removed | 49 | 0.7614 | 0.7618 | 0.9043 | 0.9389 |
| Raw measurements, no severity | 32 | 0.7618 | 0.7622 | 0.9044 | 0.9387 |

Adjacent accuracy stays 1.0 in all four rows. Hand-built interactions are redundant for this booster. Severity is not: it is the heaviest term in the index (weight 1.70 on `sev_term`) and an input the dashboard collects. A deployment that must score a run before severity is known should expect accuracy near 0.76, QWK near 0.90, on this policy — still well above 0.25, and limited by the missing input rather than by the model family.

Classifier-only ablation (LightGBM, 600 trees), for the appendix: full 0.9346, raw plus severity 0.9276, no severity 0.7460, raw and no severity 0.7458.

### What the model uses

Split-count importance on the winning regressor, highest first: `memory_usage_mb` (5444), `cpu_usage_pct` (4682), `mem_x_retry` (2918), `cpu_x_retry` (2855), `cpu_x_flaky` (2760), `total_duration_sec` (2477), `sev_x_retry` (1966), `sev_score` (1652), `log_memory_usage_mb` (1652), `failure_stage_deploy` (1473). Tool, language, OS, and cloud do not enter the top 15. That is the expected pattern: those fields are controls, and the generator did not couple them to the index. Deploy-stage appears because the policy both multiplies deploy runs by 1.25 and adds a memory×deploy term.

Figure files: `results/confusion_matrices/cm_risk_level_best.png`, `results/fi_risk_level_best.png`.

### Negative controls

Rollback, test set, majority accuracy 0.4999:

| Model | Accuracy | Macro F1 | AUC | Log loss |
|---|---|---|---|---|
| LightGBM (selected) | 0.5022 | 0.5022 | 0.5001 | 0.7012 |
| Logistic regression | 0.4987 | 0.4972 | 0.5002 | 0.6938 |
| XGBoost | 0.4981 | 0.4981 | 0.4975 | 0.7010 |
| HistGradientBoosting | 0.4897 | 0.4865 | 0.4922 | 0.6949 |

Incident, test set, majority accuracy 0.4949:

| Model | Accuracy | Macro F1 | AUC | Log loss |
|---|---|---|---|---|
| HistGradientBoosting (selected) | 0.5044 | 0.5006 | 0.5003 | 0.6944 |
| XGBoost | 0.5007 | 0.5005 | 0.5042 | 0.6994 |
| LightGBM | 0.4939 | 0.4936 | 0.4959 | 0.7005 |
| Logistic regression | 0.4889 | 0.4865 | 0.4903 | 0.6944 |

Train positive rates: rollback 0.4984, incident 0.5032. A paper should say these two tasks are reported so that a reader does not assume every column in the file is predictable. Selection by accuracy picked models that are tied with chance. The service still returns them because the dashboard has three cards; the probabilities sit near one half.

## System context (short; this is not the scientific contribution)

The fitted artifacts are loaded by a FastAPI service (`fullstack/ml-service/main.py`) and a CLI (`predict.py`). A React dashboard submits the same fields the index uses, including severity. Auth and history sit in a separate Node service. A qualitative check: a long, hot, flaky deploy on `main` with severity HIGH is scored CRITICAL; a short, cool, clean build on a feature branch with severity LOW is scored LOW.

## Discussion points that are supported

- Align the target with the inputs that will exist at inference. A label that includes an independent field the model is forbidden to see has a hard accuracy ceiling (here 0.548) no search can cross.
- For a continuous score cut into ordered tiers, regress-then-threshold beat flat four-class boosting by about three accuracy points (0.9747 vs 0.9452) and raised QWK from 0.9781 to 0.9899.
- Quadratic weighted kappa and adjacent accuracy describe this task better than accuracy alone. Under the old protocol, adjacent accuracy was already 0.8788 while exact accuracy was 0.4783.
- Control features with no role in the policy receive no importance. That is a sanity check on both the generator and the importance ranking.
- Hyperparameter search on the flat classifier (12 trials) moved accuracy by 0.0057. Changing the target and the decision head moved it by roughly 0.50 relative to the v1 reproduction.

## Limitations — state these in the paper

- The corpus is synthetic. Independence of the native labels is a property of this file, measured here, and may not hold for industrial logs.
- The ORI is specified by the study, not annotated by an operator. High accuracy means the booster approximates that specification. A different weighting would be a different target.
- Test metrics are one 80/20 split (seed 42). The 5-fold CV on the training portion is the stability check (standard deviation 0.0015).
- Regression probabilities are geometrically defined. Report accuracy, F1, kappa, and AUC for that model; treat its log loss as incomparable.
- CatBoost and neural nets were not part of the comparison. Histogram gradient boosting is the strongest flat classifier that was run.
- No grouped or temporal split: runs do not carry a real sequential label. `pipeline_id` repeats, but the native outcomes attached to those repeats are independent noise, so grouping would not create a forecast.
- Rollback and incident predictions in the demo must be described as negative controls.

## Suggested paper structure

1. Introduction — CI risk tiers, and the failure mode of training on tags that are independent of telemetry.
2. Data and the independence audit.
3. Two label protocols: latent-severity quartiles (with Bayes ceiling) and the ORI (formula, train-only cuts).
4. Models, preprocessing, metrics, Optuna settings.
5. Results: main table, per-class scores, ablation, importance, negative controls.
6. Deployment note, one paragraph.
7. Limitations and what would be required on real logs: an outcome that is not a function of the feature row (for example a later production incident) and a time-based split.

## Claims to avoid

- Do not write that the system predicts production failures, rollbacks, or incidents at 97% accuracy.
- Do not write that logistic regression was the best model. That was the earlier protocol. Under v3 it is the weakest of the eight.
- Do not average the negative-control AUCs into the risk result.
- Do not describe the ORI weights as learned. They are constants. The learned object is the regressor that approximates the resulting tiers.
