const CHAT_KB = [
  {
    patterns: ["what is risk level", "risk level mean", "risk score", "what does risk"],
    answer: "**Risk Level** is our primary prediction target, classified into four tiers:\n\n🟢 **LOW** — Pipeline is stable, minimal chance of failure\n🟡 **MEDIUM** — Some risk factors present; monitor closely\n🟠 **HIGH** — Multiple risk signals detected; review before deploying\n🔴 **CRITICAL** — Pipeline failure is highly likely; immediate attention required\n\nThe risk score is computed from severity, retry count, CPU/memory pressure, and incident history."
  },
  {
    patterns: ["rollback", "rollback triggered", "what is rollback"],
    answer: "**Rollback Triggered** predicts whether a deployment rollback will be necessary. A rollback reverts the system to the last stable state when a deployment causes issues.\n\n⚠️ A **Yes** prediction means the model detected conditions that historically led to rollbacks (high retry counts, critical severity, deploy-stage failures)."
  },
  {
    patterns: ["incident", "incident created", "what is incident"],
    answer: "**Incident Created** predicts whether the pipeline failure will escalate into a formal incident report. Incidents typically involve:\n- Production outages\n- SLA breaches\n- On-call alerts\n\nThis is a binary prediction: **Yes** or **No**."
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
    answer: "The system trained **5 models** per prediction task:\n\n| Model | Task 1 AUC |\n|---|---|\n| Logistic Regression ⭐ | **0.9995** |\n| Random Forest | 0.9958 |\n| XGBoost | 0.9987 |\n| LightGBM | 0.9986 |\n| LightGBM_Tuned | 0.9987 |\n\nLogistic Regression is the best model for Risk Level prediction (highest AUC). LightGBM Tuned leads for Rollback prediction."
  },
  {
    patterns: ["accuracy", "f1", "auc", "metrics", "performance"],
    answer: "**Model Performance** on 9,000 test samples:\n\n**Risk Level (4-class)**\n- Best Accuracy: 98.19% (vs 25% random baseline)\n- Best AUC: **0.9995** ✅\n\n> Note: 98.19% accuracy on 4 classes is outstanding — the models successfully learned the complex interactions!\n\n**Rollback & Incident** tasks have AUC ≈ 0.50 because these columns are randomly assigned in the synthetic dataset."
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
