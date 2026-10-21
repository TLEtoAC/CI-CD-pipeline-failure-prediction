const CHAT_KB = [
  {
    patterns: ["what is risk level", "risk level mean", "risk score", "what does risk"],
    answer: "**Risk Level** is our primary prediction target, classified into four tiers:\n\n🟢 **LOW** — Pipeline is stable, minimal chance of failure\n🟡 **MEDIUM** — Some risk factors present; monitor closely\n🟠 **HIGH** — Multiple risk signals detected; review before deploying\n🔴 **CRITICAL** — Pipeline failure is highly likely; immediate attention required\n\nThe risk score is computed from severity, retry count, CPU/memory pressure, and incident history."
  },
  {
    patterns: ["rollback", "rollback triggered", "what is rollback"],
    answer: "**Rollback Triggered** is a binary classifier on the dataset column of the same name. On the 9,000-row test split the best model (LightGBM) scores **50.2%** accuracy and **0.50 AUC**, against a 50.0% majority baseline.\n\nChanging CPU, retries, or severity does not move this probability. The column is independent of the telemetry."
  },
  {
    patterns: ["incident", "incident created", "what is incident"],
    answer: "**Incident Created** is a binary classifier on the dataset column of the same name. The saved model is histogram gradient boosting at **50.4%** accuracy and **0.50 AUC**, against a 49.5% majority baseline.\n\nIt returns Yes or No, but both probabilities stay near one half."
  },
  {
    patterns: ["confidence", "confidence score", "probability"],
    answer: "The **Confidence Score** is the model's probability estimate for its top prediction. For example, a confidence of 78% means the model assigns 78% probability to the predicted class.\n\nAll class probabilities are shown in the breakdown chart, summing to 100%."
  },
  {
    patterns: ["ci tool", "jenkins", "github actions", "gitlab", "azure devops", "circleci"],
    answer: "The **CI Tool** field captures the CI/CD platform running the pipeline:\n\n| Tool | Common Use |\n|---|---|\n| Jenkins | On-prem, highly configurable |\n| GitHub Actions | Native GitHub integration |\n| GitLab CI | GitLab ecosystem |\n| Azure DevOps | Microsoft stack |\n| CircleCI | Cloud-native |\n\nDifferent tools can correlate with different failure patterns based on your infrastructure."
  },
  {
    patterns: ["cpu", "memory", "resource", "cpu usage", "memory usage"],
    answer: "**CPU Usage %** and **Memory Usage (MB)** capture resource pressure during the pipeline run.\n\nHigh CPU + high retry count is a strong signal for HIGH/CRITICAL risk. These are combined into interaction features:\n- `cpu_x_retry = cpu_usage × retry_count`\n- `mem_x_retry = memory_mb × retry_count`"
  },
  {
    patterns: ["retry", "retry count", "retries"],
    answer: "**Retry Count** is how many times the pipeline retried failed steps. It's one of the strongest predictors:\n\n- 0 retries → usually stable runs\n- 1–2 retries → moderate concern\n- 3+ retries → high risk signal\n\nRetries interact multiplicatively with CPU and memory to amplify the risk score."
  },
  {
    patterns: ["flaky", "flaky test", "is_flaky"],
    answer: "**Flaky Test** (is_flaky_test = 1) indicates the pipeline contains known intermittently-failing tests.\n\nFlaky tests inflate retry counts and can mask real failures. Marking this flag helps the model distinguish genuine failures from test instability."
  },
  {
    patterns: ["severity", "what is severity"],
    answer: "**Severity** reflects the assessed impact of the failure:\n\n| Level | Score | Meaning |\n|---|---|---|\n| LOW | 1 | Minimal impact |\n| MEDIUM | 2 | Moderate impact |\n| HIGH | 3 | Significant impact |\n| CRITICAL | 4 | Business-critical |\n\nSeverity carries a 2× weight in the risk formula and is the single strongest predictor."
  },
  {
    patterns: ["model", "which model", "algorithm", "logistic regression", "xgboost", "lightgbm", "random forest"],
    answer: "The deployed **risk** model is a LightGBM regressor of the operational risk index, then cut into four tiers. Held-out accuracy is **0.9747**.\n\nDirect classifiers on the same tier: HistGradientBoosting 0.9452, tuned LightGBM 0.9438, XGBoost 0.9384, LightGBM 0.9381, extra trees 0.8684, random forest 0.8633, logistic regression 0.8246.\n\nRollback is LightGBM at 0.5022. Incident is HistGradientBoosting at 0.5044. Both match chance."
  },
  {
    patterns: ["accuracy", "f1", "auc", "metrics", "performance"],
    answer: "**Held-out performance** on 9,000 test rows:\n\n**Risk level** — LightGBM score regression, accuracy **0.9747**, AUC **0.9984**. Majority baseline is 0.25.\n\n**Rollback** — LightGBM, accuracy **0.5022**, AUC **0.5001**.\n\n**Incident** — HistGradientBoosting, accuracy **0.5044**, AUC **0.5003**.\n\nRollback and incident match chance because those columns are independent of the features."
  },
  {
    patterns: ["branch", "main", "develop", "release", "feature"],
    answer: "**Branch** indicates which Git branch triggered the pipeline:\n\n- `main` → Production branch; failures here are critical\n- `release` → Release candidate; high scrutiny\n- `develop` → Integration branch; moderate risk\n- `feature-x` → Feature branch; usually lower stakes\n\nThe model encodes `is_main_branch` and `is_release_branch` as binary flags."
  },
  {
    patterns: ["failure stage", "build", "test", "deploy", "stage"],
    answer: "**Failure Stage** is where in the pipeline the failure occurred:\n\n🔨 **Build** — Compilation/dependency errors\n🧪 **Test** — Test suite failures\n🚀 **Deploy** — Deployment/rollout failures\n\nDeploy-stage failures are typically most severe (encoded as `is_deploy_stage = 1`)."
  },
  {
    patterns: ["how to use", "how do i", "getting started", "tutorial", "help"],
    answer: "**How to use this dashboard:**\n\n1. **Predict Tab** → Fill in your pipeline parameters and click **Run Prediction** to get instant risk scores\n2. **Analytics Tab** → Explore model performance metrics, accuracy charts, and feature importances\n3. **History** → Review your past predictions\n4. **Chatbot** → Ask me anything about predictions, features, or model behavior!\n\n💡 **Tip:** Try setting severity=CRITICAL + retry_count=5 to see a CRITICAL risk prediction."
  },
  {
    patterns: ["dataset", "data", "training data", "how many"],
    answer: "**Dataset:** `ci_cd_pipeline_failure_logs_dataset.csv`\n\n- **45,000 records** of CI/CD pipeline runs\n- Split: 80% train / 20% test (36,000 / 9,000)\n- **Features:** ~47 engineered columns including log transforms and interactions\n- **Synthetic dataset:** rollback and incident labels are randomly assigned, explaining their near-random AUC\n\nThe risk_level target was **engineered** from meaningful signals to achieve an outstanding AUC = 0.9995."
  },
  {
    patterns: ["feature importance", "important feature", "top feature", "which feature"],
    answer: "**Top Features for Risk Level Prediction:**\n\n1. `severity_ordinal` — Highest importance (2× weight)\n2. `cpu_x_retry` — CPU × retry count interaction\n3. `mem_x_retry` — Memory × retry count interaction  \n4. `retry_count` — Raw retry count\n5. `cpu_usage_pct` — Raw CPU pressure\n\nCheck the **Analytics → Feature Importances** tab for interactive charts."
  },
];

function chatbotReply(message, history) {
  const msg = message.toLowerCase();

  for (const entry of CHAT_KB) {
    if (entry.patterns.some(p => msg.includes(p))) {
      return entry.answer;
    }
  }

  if (history && history.length > 0) {
    const lastBot = [...history].reverse().find(m => m.role === "assistant");
    if (lastBot && (msg.includes("why") || msg.includes("explain") || msg.includes("reason"))) {
      return "Based on the prediction shown, the key contributing factors are typically:\n\n1. **Severity level** — the single strongest predictor (2× weight)\n2. **Retry count** — repeated failures signal instability\n3. **CPU/Memory pressure** — resource exhaustion triggers cascades\n4. **Failure stage** — deploy-stage failures carry highest risk\n\nFor a detailed feature importance breakdown, check the **Analytics** tab.";
    }
  }

  if (msg.match(/^(hi|hello|hey|howdy|yo|sup)\b/)) {
    return "👋 Hello! I'm your CI/CD Prediction Assistant. I can help you:\n\n- Understand prediction results\n- Explain what each feature means\n- Guide you through the dashboard\n- Answer questions about model performance\n\nWhat would you like to know?";
  }

  if (msg.match(/thank|thanks|thx/)) {
    return "You're welcome! 😊 Feel free to ask anything else about the CI/CD prediction system.";
  }

  return "I'm not sure I understand that fully. Here are some topics I know about:\n\n- **Predictions**: risk level, rollback, incident created\n- **Features**: severity, retry count, CPU/memory, branch, failure stage\n- **Model**: accuracy, AUC, F1-score, algorithms used\n- **Usage**: how to use the dashboard\n\nTry asking something like *\"What does risk level mean?\"* or *\"How is confidence calculated?\"*";
}

exports.chat = (req, res) => {
  const { message, history } = req.body || {};
  if (!message) return res.status(400).json({ error: "message is required" });
  const reply = chatbotReply(message.trim(), history || []);
  res.json({ reply, timestamp: new Date().toISOString() });
};
