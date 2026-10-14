// src/pages/PredictPage.jsx — Main prediction dashboard
import { useState } from "react";
import { predictAPI } from "../api";
import toast from "react-hot-toast";
import { Play, AlertTriangle, CheckCircle, XCircle, Info, ChevronDown, ChevronUp } from "lucide-react";

const DEFAULTS = {
  ci_tool: "Jenkins", language: "Python", os: "ubuntu-latest",
  cloud_provider: "AWS", branch: "main", failure_stage: "build",
  build_duration_sec: 1200, test_duration_sec: 600, deploy_duration_sec: 300,
  cpu_usage_pct: 60, memory_usage_mb: 4096, retry_count: 1,
  is_flaky_test: 0, rollback_triggered: 0, incident_created: 0, severity: "MEDIUM",
};

const RISK_META = {
  LOW:      { color: "#10b981", bg: "rgba(16,185,129,0.1)",  icon: "🟢", label: "Low Risk" },
  MEDIUM:   { color: "#f59e0b", bg: "rgba(245,158,11,0.1)", icon: "🟡", label: "Medium Risk" },
  HIGH:     { color: "#ef4444", bg: "rgba(239,68,68,0.1)",  icon: "🟠", label: "High Risk" },
  CRITICAL: { color: "#ff3b6b", bg: "rgba(255,59,107,0.1)", icon: "🔴", label: "Critical Risk" },
};

function ProbBar({ label, value, isTop, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
      <span style={{ width: 80, fontSize: 12, color: isTop ? color : "var(--text-secondary)", fontWeight: isTop ? 700 : 400 }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${value * 100}%`, height: "100%", background: isTop ? color : "rgba(255,255,255,0.15)", borderRadius: 3, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ width: 42, fontSize: 12, textAlign: "right", color: isTop ? color : "var(--text-muted)", fontWeight: isTop ? 700 : 400 }}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function ResultCard({ title, result, isRisk }) {
  const [expanded, setExpanded] = useState(false);
  if (!result) return null;

  let meta = { color: "var(--accent)", bg: "rgba(99,102,241,0.1)", icon: "⚪", label: result.prediction };
  if (isRisk) meta = RISK_META[result.prediction] || meta;
  const isYes = String(result.prediction) === "1" || result.prediction === "True";
  if (!isRisk) {
    meta = isYes
      ? { color: "#ef4444", bg: "rgba(239,68,68,0.1)", icon: "⚠️", label: "Yes" }
      : { color: "#10b981", bg: "rgba(16,185,129,0.1)", icon: "✅", label: "No" };
  }

  const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : null;

  return (
    <div className="result-card" style={{ borderColor: meta.color + "55", background: meta.bg }}>
      <div className="result-card-header">
        <div>
          <span className="result-icon">{meta.icon}</span>
          <div>
            <div className="result-task">{title}</div>
            <div className="result-prediction" style={{ color: meta.color }}>{meta.label}</div>
          </div>
        </div>
        <div className="result-right">
          {confidence && (
            <div className="confidence-ring" style={{ "--conf-color": meta.color, "--conf-pct": confidence }}>
              <svg viewBox="0 0 36 36" width="60" height="60">
                <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
                <circle cx="18" cy="18" r="15" fill="none" stroke={meta.color} strokeWidth="3"
                  strokeDasharray={`${parseFloat(confidence) * 0.942} 94.2`}
                  strokeLinecap="round" transform="rotate(-90 18 18)" style={{ transition: "stroke-dasharray 0.8s ease" }} />
                <text x="18" y="22" textAnchor="middle" fill={meta.color} fontSize="8" fontWeight="700">{confidence}%</text>
              </svg>
              <span className="conf-label">confidence</span>
            </div>
          )}
          <span className="result-model">{result.model_used}</span>
          <button className="expand-btn" onClick={() => setExpanded(p => !p)}>
            {expanded ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
          </button>
        </div>
      </div>
      {expanded && result.probabilities && Object.keys(result.probabilities).length > 0 && (
        <div className="prob-breakdown">
          <div className="prob-title">Class Probabilities</div>
          {Object.entries(result.probabilities)
            .sort(([,a],[,b]) => b - a)
            .map(([cls, prob]) => {
              const displayLabel = isRisk ? cls : (cls === "1" || cls === "True" ? "Yes" : "No");
              const isTop = cls === String(result.prediction) || (isYes && (cls === "1" || cls === "True")) || (!isYes && (cls === "0" || cls === "False"));
              return <ProbBar key={cls} label={displayLabel} value={prob} isTop={isTop} color={meta.color} />;
            })}
        </div>
      )}
    </div>
  );
}

export default function PredictPage() {
  const [form, setForm]       = useState(DEFAULTS);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResults(null);
    try {
      const payload = { ...form,
        build_duration_sec:  parseInt(form.build_duration_sec),
        test_duration_sec:   parseInt(form.test_duration_sec),
        deploy_duration_sec: parseInt(form.deploy_duration_sec),
        cpu_usage_pct:       parseFloat(form.cpu_usage_pct),
        memory_usage_mb:     parseInt(form.memory_usage_mb),
        retry_count:         parseInt(form.retry_count),
        is_flaky_test:       parseInt(form.is_flaky_test),
        rollback_triggered:  parseInt(form.rollback_triggered),
        incident_created:    parseInt(form.incident_created),
      };
      const { data } = await predictAPI.predict(payload);
      setResults(data);
      toast.success("Prediction complete!");
    } catch (err) {
      toast.error(err.response?.data?.error || "Prediction failed. Is the ML service running?");
    } finally { setLoading(false); }
  };

  const Sel = ({ id, label, val, formKey, opts }) => (
    <div className="input-group">
      <label className="input-label">{label}</label>
      <select id={id} className="select" value={val} onChange={e => set(formKey, e.target.value)}>
        {opts.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
  const Num = ({ id, label, val, formKey, min, max, step }) => (
    <div className="input-group">
      <label className="input-label">{label}</label>
      <input id={id} type="number" className="input" value={val} min={min} max={max} step={step || 1} onChange={e => set(formKey, e.target.value)} />
    </div>
  );

  return (
    <div className="page-content fade-in">
      <div className="page-header">
        <div>
          <h2 className="page-title">Pipeline Risk Predictor</h2>
          <p className="page-desc">Enter your CI/CD pipeline parameters to get an instant risk assessment.</p>
        </div>
      </div>

      <div className="predict-layout">
        {/* Form */}
        <div className="card predict-form-card">
          <h3 className="section-title">Pipeline Parameters</h3>
          <form onSubmit={handleSubmit}>
            <div className="form-section">
              <div className="form-section-title">🔧 Infrastructure</div>
              <div className="form-grid-2">
                <Sel id="ci-tool" label="CI Tool" val={form.ci_tool} formKey="ci_tool" opts={["Jenkins","GitHub Actions","GitLab CI","Azure DevOps","CircleCI"]} />
                <Sel id="language" label="Language" val={form.language} formKey="language" opts={["Python","Go","NodeJS","C++","Rust","Java"]} />
                <Sel id="os" label="Operating System" val={form.os} formKey="os" opts={["ubuntu-latest","windows-latest","macos-latest"]} />
                <Sel id="cloud-provider" label="Cloud Provider" val={form.cloud_provider} formKey="cloud_provider" opts={["AWS","GCP","Azure","On-Prem"]} />
                <Sel id="branch" label="Branch" val={form.branch} formKey="branch" opts={["main","develop","feature-x","release"]} />
                <Sel id="failure-stage" label="Failure Stage" val={form.failure_stage} formKey="failure_stage" opts={["build","test","deploy"]} />
              </div>
            </div>

            <div className="form-section">
              <div className="form-section-title">⏱️ Durations & Resources</div>
              <div className="form-grid-3">
                <Num id="build-dur" label="Build Duration (s)" val={form.build_duration_sec} formKey="build_duration_sec" min={0} />
                <Num id="test-dur" label="Test Duration (s)" val={form.test_duration_sec} formKey="test_duration_sec" min={0} />
                <Num id="deploy-dur" label="Deploy Duration (s)" val={form.deploy_duration_sec} formKey="deploy_duration_sec" min={0} />
                <Num id="cpu-usage" label="CPU Usage %" val={form.cpu_usage_pct} formKey="cpu_usage_pct" min={0} max={100} step={0.1} />
                <Num id="memory-mb" label="Memory (MB)" val={form.memory_usage_mb} formKey="memory_usage_mb" min={0} />
                <Num id="retry-count" label="Retry Count" val={form.retry_count} formKey="retry_count" min={0} max={20} />
              </div>
            </div>

            <div className="form-section">
              <div className="form-section-title">🚦 Context Flags</div>
              <div className="form-grid-2">
                <Sel id="severity" label="Severity" val={form.severity} formKey="severity" opts={["LOW","MEDIUM","HIGH","CRITICAL"]} />
                <Sel id="flaky-test" label="Flaky Test?" val={form.is_flaky_test} formKey="is_flaky_test" opts={[0,1]} />
                <Sel id="rollback" label="Rollback Triggered?" val={form.rollback_triggered} formKey="rollback_triggered" opts={[0,1]} />
                <Sel id="incident" label="Incident Created?" val={form.incident_created} formKey="incident_created" opts={[0,1]} />
              </div>
            </div>

            <button id="predict-btn" type="submit" className="btn btn-primary predict-btn" disabled={loading}>
              {loading ? <><span className="spinner" /> Running ML Models…</> : <><Play size={16} /> Run Prediction</>}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="predict-results">
          {!results && !loading && (
            <div className="results-empty">
              <div className="empty-icon"><Info size={40} /></div>
              <h3>Results will appear here</h3>
              <p>Fill in the pipeline parameters and click <strong>Run Prediction</strong> to get your risk assessment across all three prediction tasks.</p>
            </div>
          )}
          {loading && (
            <div className="results-loading">
              <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
              <span>Running inference across all ML models…</span>
            </div>
          )}
          {results && (
            <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <ResultCard title="Pipeline Risk Level" result={results.risk_level} isRisk={true} />
              <ResultCard title="Rollback Triggered" result={results.rollback_triggered} isRisk={false} />
              <ResultCard title="Incident Created" result={results.incident_created} isRisk={false} />
            </div>
          )}
        </div>
      </div>

      <style>{`
        .page-content { padding: 32px; max-width: 1400px; }
        .page-header { margin-bottom: 28px; }
        .page-title { font-size: 24px; font-weight: 800; margin-bottom: 4px; }
        .page-desc { color: var(--text-secondary); font-size: 14px; }
        .predict-layout { display: grid; grid-template-columns: 1fr 420px; gap: 24px; align-items: start; }
        @media (max-width: 1100px) { .predict-layout { grid-template-columns: 1fr; } }
        .predict-form-card { padding: 28px; }
        .section-title { font-size: 16px; font-weight: 700; margin-bottom: 20px; }
        .form-section { margin-bottom: 24px; }
        .form-section-title { font-size: 12px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px; }
        .form-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        .form-grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }
        @media (max-width: 600px) { .form-grid-2,.form-grid-3 { grid-template-columns: 1fr; } }
        .predict-btn { width: 100%; justify-content: center; padding: 14px; font-size: 15px; margin-top: 8px; }
        .predict-results { display: flex; flex-direction: column; gap: 16px; }
        .results-empty { background: var(--bg-card); border: 1px dashed var(--border); border-radius: var(--radius); padding: 48px 24px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 14px; }
        .empty-icon { color: var(--text-muted); }
        .results-empty h3 { font-size: 17px; font-weight: 600; }
        .results-empty p { font-size: 14px; color: var(--text-secondary); max-width: 300px; line-height: 1.6; }
        .results-loading { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 48px 24px; display: flex; flex-direction: column; align-items: center; gap: 16px; color: var(--text-secondary); font-size: 14px; }
        .result-card { border: 1px solid; border-radius: var(--radius); padding: 20px; transition: all 0.3s; }
        .result-card-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
        .result-card-header > div:first-child { display: flex; align-items: center; gap: 14px; }
        .result-icon { font-size: 28px; }
        .result-task { font-size: 12px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
        .result-prediction { font-size: 22px; font-weight: 800; }
        .result-right { display: flex; align-items: center; gap: 10px; }
        .confidence-ring { display: flex; flex-direction: column; align-items: center; gap: 2px; }
        .conf-label { font-size: 9px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
        .result-model { font-size: 10px; color: var(--text-muted); font-weight: 600; background: rgba(255,255,255,0.05); padding: 3px 7px; border-radius: 4px; white-space: nowrap; }
        .expand-btn { background: none; border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; cursor: pointer; color: var(--text-secondary); transition: all 0.2s; }
        .expand-btn:hover { border-color: var(--accent); color: var(--accent); }
        .prob-breakdown { margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.06); }
        .prob-title { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
      `}</style>
    </div>
  );
}
