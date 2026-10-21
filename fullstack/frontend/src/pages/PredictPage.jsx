// src/pages/PredictPage.jsx — Interactive lab for the three saved models
import { useState } from "react";
import { predictAPI } from "../api";
import toast from "react-hot-toast";
import { Play, GitCompare, ShieldAlert, Undo2, Siren } from "lucide-react";

const BASE = {
  ci_tool: "Jenkins",
  language: "Python",
  os: "ubuntu-latest",
  cloud_provider: "AWS",
  branch: "main",
  failure_stage: "build",
  build_duration_sec: 1200,
  test_duration_sec: 600,
  deploy_duration_sec: 300,
  cpu_usage_pct: 60,
  memory_usage_mb: 4096,
  retry_count: 1,
  is_flaky_test: 0,
  severity: "MEDIUM",
};

const PRESETS = [
  {
    id: "calm",
    name: "Calm feature build",
    blurb: "Short, cool, clean run on a feature branch.",
    values: {
      ...BASE,
      ci_tool: "GitHub Actions",
      branch: "feature-x",
      failure_stage: "build",
      build_duration_sec: 180,
      test_duration_sec: 90,
      deploy_duration_sec: 40,
      cpu_usage_pct: 18,
      memory_usage_mb: 512,
      retry_count: 0,
      is_flaky_test: 0,
      severity: "LOW",
    },
  },
  {
    id: "flaky",
    name: "Flaky release tests",
    blurb: "Release branch, flaky tests, several retries.",
    values: {
      ...BASE,
      branch: "release",
      failure_stage: "test",
      build_duration_sec: 900,
      test_duration_sec: 1600,
      deploy_duration_sec: 200,
      cpu_usage_pct: 72,
      memory_usage_mb: 8192,
      retry_count: 3,
      is_flaky_test: 1,
      severity: "HIGH",
    },
  },
  {
    id: "hot",
    name: "Hot main deploy",
    blurb: "Long, hot, flaky deploy on main.",
    values: {
      ...BASE,
      branch: "main",
      failure_stage: "deploy",
      build_duration_sec: 2800,
      test_duration_sec: 1800,
      deploy_duration_sec: 1500,
      cpu_usage_pct: 94,
      memory_usage_mb: 28000,
      retry_count: 5,
      is_flaky_test: 1,
      severity: "CRITICAL",
    },
  },
];

const TASKS = [
  {
    id: "risk_level",
    title: "Risk level",
    icon: ShieldAlert,
    accuracy: 0.9747,
    baseline: 0.2528,
    model: "LGBM score regression",
    note: "This tier is a function of the inputs on the form. Moving CPU, memory, duration, retries, flaky tests, severity, stage, or branch changes the score.",
  },
  {
    id: "rollback_triggered",
    title: "Rollback triggered",
    icon: Undo2,
    accuracy: 0.5022,
    baseline: 0.4999,
    model: "LightGBM",
    note: "Held-out accuracy is 50.2% against a 50.0% majority baseline. On this dataset the rollback bit is independent of telemetry, so the probability stays near one half no matter which scenario you run.",
  },
  {
    id: "incident_created",
    title: "Incident created",
    icon: Siren,
    accuracy: 0.5044,
    baseline: 0.4949,
    model: "HistGradientBoosting",
    note: "Held-out accuracy is 50.4% against a 49.5% majority baseline. Adding failure type, severity, and error code as features still leaves ROC AUC at about 0.51.",
  },
];

const RISK_COLOR = {
  LOW: "#10b981",
  MEDIUM: "#f59e0b",
  HIGH: "#ef4444",
  CRITICAL: "#ff3b6b",
};

function toPayload(form) {
  return {
    ...form,
    build_duration_sec: Number(form.build_duration_sec),
    test_duration_sec: Number(form.test_duration_sec),
    deploy_duration_sec: Number(form.deploy_duration_sec),
    cpu_usage_pct: Number(form.cpu_usage_pct),
    memory_usage_mb: Number(form.memory_usage_mb),
    retry_count: Number(form.retry_count),
    is_flaky_test: Number(form.is_flaky_test),
    rollback_triggered: 0,
    incident_created: 0,
  };
}

function yesNo(prediction) {
  const raw = String(prediction);
  return raw === "1" || raw === "True" || raw === "true";
}

function displayLabel(taskId, prediction) {
  if (taskId === "risk_level") return String(prediction);
  return yesNo(prediction) ? "Yes" : "No";
}

function classLabel(taskId, cls, prediction) {
  if (taskId === "risk_level") return cls;
  const positive = cls === "1" || cls === "True" || cls === "true";
  if (positive) return "Yes";
  if (cls === "0" || cls === "False" || cls === "false") return "No";
  return String(prediction);
}

function tone(taskId, prediction) {
  if (taskId === "risk_level") return RISK_COLOR[prediction] || "#6366f1";
  return yesNo(prediction) ? "#ef4444" : "#10b981";
}

function ProbList({ taskId, result }) {
  const entries = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  const color = tone(taskId, result.prediction);
  return (
    <div className="prob-list">
      {entries.map(([cls, prob]) => {
        const label = classLabel(taskId, cls, result.prediction);
        const active = label === displayLabel(taskId, result.prediction);
        return (
          <div key={cls} className="prob-row">
            <span style={{ color: active ? color : "var(--text-secondary)", fontWeight: active ? 700 : 500 }}>{label}</span>
            <div className="prob-track">
              <div style={{ width: `${Math.max(prob * 100, 1)}%`, background: active ? color : "rgba(255,255,255,0.18)" }} />
            </div>
            <span>{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default function PredictPage() {
  const [form, setForm] = useState(PRESETS[0].values);
  const [activePreset, setActivePreset] = useState(PRESETS[0].id);
  const [focus, setFocus] = useState("risk_level");
  const [results, setResults] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);

  const set = (key, value) => {
    setActivePreset(null);
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const applyPreset = (preset) => {
    setForm(preset.values);
    setActivePreset(preset.id);
  };

  const run = async (values) => {
    const { data } = await predictAPI.predict(toPayload(values));
    return data;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const data = await run(form);
      setResults(data);
      toast.success("Models scored this scenario");
    } catch (err) {
      toast.error(err.response?.data?.error || "Prediction failed. Is the ML service running?");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    setComparing(true);
    try {
      const rows = [];
      for (const preset of PRESETS) {
        const data = await run(preset.values);
        rows.push({ preset, data });
      }
      setComparison(rows);
      setResults(rows[rows.length - 1].data);
      setForm(PRESETS[PRESETS.length - 1].values);
      setActivePreset(PRESETS[PRESETS.length - 1].id);
      toast.success("Compared three scenarios");
    } catch (err) {
      toast.error(err.response?.data?.error || "Comparison failed. Is the ML service running?");
    } finally {
      setComparing(false);
    }
  };

  const focused = TASKS.find((task) => task.id === focus);
  const focusedResult = results?.[focus];

  return (
    <div className="page-content fade-in">
      <div className="page-header">
        <div>
          <h2 className="page-title">Model test bench</h2>
          <p className="page-desc">
            Score one pipeline, or run the three built-in scenarios side by side. Risk level follows the inputs. Rollback and incident stay near chance.
          </p>
        </div>
      </div>

      <div className="task-switch" role="tablist">
        {TASKS.map((task) => {
          const Icon = task.icon;
          const selected = focus === task.id;
          return (
            <button
              key={task.id}
              type="button"
              role="tab"
              aria-selected={selected}
              className={`task-tab ${selected ? "selected" : ""}`}
              onClick={() => setFocus(task.id)}
            >
              <Icon size={16} />
              <span>
                <strong>{task.title}</strong>
                <em>{(task.accuracy * 100).toFixed(1)}% held-out · baseline {(task.baseline * 100).toFixed(1)}%</em>
              </span>
            </button>
          );
        })}
      </div>

      <div className="predict-layout">
        <form className="card predict-form-card" onSubmit={handleSubmit}>
          <div className="preset-row">
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={`preset ${activePreset === preset.id ? "on" : ""}`}
                onClick={() => applyPreset(preset)}
              >
                <strong>{preset.name}</strong>
                <span>{preset.blurb}</span>
              </button>
            ))}
          </div>

          <div className="form-section">
            <div className="form-section-title">Infrastructure</div>
            <div className="form-grid-2">
              <Field label="CI tool">
                <select id="ci-tool" className="select" value={form.ci_tool} onChange={(e) => set("ci_tool", e.target.value)}>
                  {["Jenkins", "GitHub Actions", "GitLab CI", "Azure DevOps", "CircleCI"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </Field>
              <Field label="Language">
                <select id="language" className="select" value={form.language} onChange={(e) => set("language", e.target.value)}>
                  {["Python", "Go", "NodeJS", "C++", "Rust", "Java"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </Field>
              <Field label="Operating system">
                <select id="os" className="select" value={form.os} onChange={(e) => set("os", e.target.value)}>
                  {["ubuntu-latest", "windows-latest", "macos-latest"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </Field>
              <Field label="Cloud">
                <select id="cloud" className="select" value={form.cloud_provider} onChange={(e) => set("cloud_provider", e.target.value)}>
                  {["AWS", "GCP", "Azure", "On-Prem"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </Field>
              <Field label="Branch">
                <select id="branch" className="select" value={form.branch} onChange={(e) => set("branch", e.target.value)}>
                  {["main", "develop", "feature-x", "release"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </Field>
              <Field label="Stage">
                <select id="stage" className="select" value={form.failure_stage} onChange={(e) => set("failure_stage", e.target.value)}>
                  {["build", "test", "deploy"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </Field>
            </div>
          </div>

          <div className="form-section">
            <div className="form-section-title">Load</div>
            <div className="slider-grid">
              <Slider id="cpu" label="CPU" value={form.cpu_usage_pct} min={10} max={95} step={1} unit="%" onChange={(v) => set("cpu_usage_pct", Number(v))} />
              <Slider id="memory" label="Memory" value={form.memory_usage_mb} min={256} max={32768} step={256} unit=" MB" onChange={(v) => set("memory_usage_mb", Number(v))} />
              <Slider id="retries" label="Retries" value={form.retry_count} min={0} max={5} step={1} onChange={(v) => set("retry_count", Number(v))} />
              <Slider id="build" label="Build" value={form.build_duration_sec} min={30} max={3600} step={30} unit=" s" onChange={(v) => set("build_duration_sec", Number(v))} />
              <Slider id="test" label="Test" value={form.test_duration_sec} min={10} max={2400} step={10} unit=" s" onChange={(v) => set("test_duration_sec", Number(v))} />
              <Slider id="deploy" label="Deploy" value={form.deploy_duration_sec} min={0} max={1800} step={20} unit=" s" onChange={(v) => set("deploy_duration_sec", Number(v))} />
            </div>
          </div>

          <div className="form-section">
            <div className="form-section-title">Context</div>
            <div className="chip-row">
              {["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((level) => (
                <button
                  key={level}
                  type="button"
                  className={`chip ${form.severity === level ? "on" : ""}`}
                  style={form.severity === level ? { borderColor: RISK_COLOR[level], color: RISK_COLOR[level] } : undefined}
                  onClick={() => set("severity", level)}
                >
                  {level}
                </button>
              ))}
              <button
                type="button"
                className={`chip ${Number(form.is_flaky_test) === 1 ? "on" : ""}`}
                onClick={() => set("is_flaky_test", Number(form.is_flaky_test) === 1 ? 0 : 1)}
              >
                {Number(form.is_flaky_test) === 1 ? "Flaky tests" : "Stable tests"}
              </button>
            </div>
          </div>

          <div className="action-row">
            <button id="predict-btn" type="submit" className="btn btn-primary" disabled={loading || comparing}>
              {loading ? <><span className="spinner" /> Scoring…</> : <><Play size={16} /> Score this scenario</>}
            </button>
            <button id="compare-btn" type="button" className="btn btn-secondary" onClick={handleCompare} disabled={loading || comparing}>
              {comparing ? <><span className="spinner" /> Comparing…</> : <><GitCompare size={16} /> Compare scenarios</>}
            </button>
          </div>
        </form>

        <div className="predict-results">
          {!results && !loading && !comparing && (
            <div className="results-empty">
              <h3>Pick a scenario and score it</h3>
              <p>
                {focused.note}
              </p>
            </div>
          )}
          {(loading || comparing) && !results && (
            <div className="results-loading">
              <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
              <span>{comparing ? "Running calm, flaky, and hot scenarios…" : "Scoring the three models…"}</span>
            </div>
          )}
          {focusedResult && (
            <article className="focus-card" style={{ borderColor: tone(focus, focusedResult.prediction) }}>
              <header>
                <div>
                  <div className="eyebrow">{focused.title}</div>
                  <div className="big" style={{ color: tone(focus, focusedResult.prediction) }}>
                    {displayLabel(focus, focusedResult.prediction)}
                  </div>
                </div>
                <div className="meta">
                  <span>{focusedResult.model_used || focused.model}</span>
                  <span>{((focusedResult.confidence || 0) * 100).toFixed(1)}% top class</span>
                </div>
              </header>
              <ProbList taskId={focus} result={focusedResult} />
              <p className="note">{focused.note}</p>
            </article>
          )}

          {results && (
            <div className="mini-grid">
              {TASKS.filter((task) => task.id !== focus).map((task) => {
                const result = results[task.id];
                const color = tone(task.id, result.prediction);
                return (
                  <button key={task.id} type="button" className="mini" onClick={() => setFocus(task.id)}>
                    <span>{task.title}</span>
                    <strong style={{ color }}>{displayLabel(task.id, result.prediction)}</strong>
                    <em>{((result.confidence || 0) * 100).toFixed(1)}%</em>
                  </button>
                );
              })}
            </div>
          )}

          {comparison && (
            <div className="card compare-card">
              <h3>Scenario comparison</h3>
              <table>
                <thead>
                  <tr>
                    <th>Scenario</th>
                    {TASKS.map((task) => <th key={task.id}>{task.title}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {comparison.map(({ preset, data }) => (
                    <tr key={preset.id}>
                      <td>{preset.name}</td>
                      {TASKS.map((task) => {
                        const result = data[task.id];
                        return (
                          <td key={task.id} style={{ color: tone(task.id, result.prediction) }}>
                            {displayLabel(task.id, result.prediction)}
                            <small>{((result.confidence || 0) * 100).toFixed(0)}%</small>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .page-content { padding: 32px; max-width: 1280px; }
        .page-header { margin-bottom: 20px; }
        .page-title { font-size: 24px; font-weight: 800; margin-bottom: 4px; }
        .page-desc { color: var(--text-secondary); font-size: 14px; max-width: 720px; }
        .task-switch { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 18px; }
        .task-tab { display: flex; gap: 10px; align-items: center; text-align: left; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; color: var(--text-secondary); cursor: pointer; }
        .task-tab strong, .task-tab em { display: block; }
        .task-tab strong { color: var(--text-primary); font-size: 14px; }
        .task-tab em { font-style: normal; font-size: 12px; color: var(--text-muted); }
        .task-tab.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); color: var(--text-primary); }
        .predict-layout { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr); gap: 20px; align-items: start; }
        .predict-form-card { padding: 22px; }
        .preset-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 18px; }
        .preset { text-align: left; background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; cursor: pointer; color: var(--text-secondary); }
        .preset strong, .preset span { display: block; }
        .preset strong { color: var(--text-primary); font-size: 13px; }
        .preset span { font-size: 12px; margin-top: 2px; }
        .preset.on { border-color: var(--accent); background: rgba(99,102,241,0.12); }
        .form-section { margin-bottom: 18px; }
        .form-section-title { font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
        .form-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .slider-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
        .slider-field { display: flex; flex-direction: column; gap: 6px; }
        .slider-head { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); font-weight: 600; }
        .slider-head strong { color: var(--text-primary); }
        .slider-field input[type="range"] { width: 100%; accent-color: #6366f1; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .chip { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); border-radius: 999px; padding: 6px 12px; font-size: 12px; font-weight: 700; cursor: pointer; }
        .chip.on { background: rgba(99,102,241,0.12); border-color: var(--accent); color: var(--text-primary); }
        .action-row { display: flex; gap: 10px; }
        .action-row .btn { flex: 1; justify-content: center; }
        .results-empty, .results-loading { background: var(--bg-card); border: 1px dashed var(--border); border-radius: var(--radius); padding: 36px 22px; }
        .results-empty h3 { font-size: 17px; margin-bottom: 8px; }
        .results-empty p, .results-loading { color: var(--text-secondary); font-size: 14px; line-height: 1.6; }
        .results-loading { display: flex; flex-direction: column; align-items: center; gap: 12px; }
        .focus-card { background: var(--bg-card); border: 1px solid; border-radius: var(--radius); padding: 18px; }
        .focus-card header { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
        .eyebrow { font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-secondary); font-weight: 700; }
        .big { font-size: 32px; font-weight: 800; line-height: 1.1; }
        .meta { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; font-size: 12px; color: var(--text-muted); }
        .prob-list { display: flex; flex-direction: column; gap: 8px; }
        .prob-row { display: grid; grid-template-columns: 88px 1fr 48px; gap: 8px; align-items: center; font-size: 12px; }
        .prob-track { height: 8px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
        .prob-track div { height: 100%; border-radius: 99px; }
        .prob-row span:last-child { text-align: right; color: var(--text-secondary); }
        .note { margin-top: 14px; font-size: 13px; color: var(--text-secondary); line-height: 1.55; }
        .mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
        .mini { text-align: left; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; cursor: pointer; }
        .mini span, .mini em { display: block; color: var(--text-muted); font-size: 12px; font-style: normal; }
        .mini strong { display: block; font-size: 18px; margin: 2px 0; }
        .compare-card { margin-top: 12px; padding: 16px; }
        .compare-card h3 { font-size: 14px; margin-bottom: 10px; }
        .compare-card table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .compare-card th { text-align: left; color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; padding-bottom: 8px; }
        .compare-card td { padding: 8px 8px 8px 0; border-top: 1px solid var(--border); font-weight: 700; }
        .compare-card td:first-child { color: var(--text-primary); font-weight: 600; }
        .compare-card small { display: block; font-weight: 500; color: var(--text-muted); }
        @media (max-width: 980px) {
          .predict-layout, .task-switch, .preset-row { grid-template-columns: 1fr; }
          .action-row { flex-direction: column; }
        }
      `}</style>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div className="input-group">
      <label className="input-label">{label}</label>
      {children}
    </div>
  );
}

function Slider({ id, label, value, min, max, step, unit = "", onChange }) {
  return (
    <label className="slider-field" htmlFor={id}>
      <span className="slider-head"><span>{label}</span><strong>{value}{unit}</strong></span>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}
