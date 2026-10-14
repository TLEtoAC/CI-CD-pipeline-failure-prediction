// src/pages/AnalyticsPage.jsx — Model performance metrics + charts
import { useState, useEffect } from "react";
import { analyticsAPI } from "../api";
import toast from "react-hot-toast";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  LineChart, Line, ResponsiveContainer,
} from "recharts";
import { TrendingUp, Award, Zap, Target } from "lucide-react";

const TASK_LABELS = {
  risk_level: "Risk Level",
  rollback_triggered: "Rollback Triggered",
  incident_created: "Incident Created",
};

const MODEL_COLORS = {
  "Logistic Regression": "#6366f1",
  "Random Forest":       "#06b6d4",
  "Extra Trees":         "#a78bfa",
  "HistGradientBoosting":"#f472b6",
  "XGBoost":             "#f59e0b",
  "LightGBM":            "#10b981",
  "LightGBM Tuned":      "#34d399",
  "LGBM Score Regression":"#fb7185",
};

const CUSTOM_TOOLTIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#131c30", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 10, padding: "10px 14px" }}>
      <p style={{ color: "#a5b4fc", fontWeight: 700, marginBottom: 6, fontSize: 13 }}>{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color, fontSize: 12 }}>{p.name}: <strong>{(p.value * 100).toFixed(1)}%</strong></p>
      ))}
    </div>
  );
};

function StatCard({ icon: Icon, label, value, sub, color }) {
  return (
    <div className="stat-card card">
      <div className="stat-card-top">
        <div className="stat-icon" style={{ background: `${color}22`, color }}><Icon size={20} /></div>
        <span className="stat-label">{label}</span>
      </div>
      <div className="stat-value" style={{ color }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export default function AnalyticsPage() {
  const [metrics,  setMetrics]  = useState(null);
  const [task,     setTask]     = useState("risk_level");
  const [fiData,   setFiData]   = useState(null);
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    analyticsAPI.metrics()
      .then(r => setMetrics(r.data))
      .catch(() => toast.error("Could not load metrics. Is the ML service running?"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!metrics) return;
    setFiData(null);
    analyticsAPI.featureImportances(task)
      .then(r => setFiData(r.data))
      .catch(() => {});
  }, [task, metrics]);

  if (loading) return <div className="page-content" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 400 }}><div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>;

  if (!metrics) return (
    <div className="page-content">
      <div className="card" style={{ textAlign: "center", padding: 48, color: "var(--text-secondary)" }}>
        ⚠️ Could not load analytics data. Make sure the ML service is running on port 8001.
      </div>
    </div>
  );

  // Build bar chart data
  const taskData = metrics[task] || {};
  const barData = Object.entries(taskData)
    .filter(([k]) => k !== "__best_model__")
    .map(([model, m]) => ({
      model: model.replace("Logistic Regression", "LR").replace("Random Forest", "RF"),
      fullModel: model,
      Accuracy: m.accuracy,
      F1: m.f1_weighted,
      AUC: m.roc_auc,
    }));

  // Best stats
  const bestModel = taskData.__best_model__ || "—";
  const bestEntry = taskData[bestModel] || {};
  const bestAUC   = bestEntry.roc_auc   ? (bestEntry.roc_auc * 100).toFixed(1) + "%" : "—";
  const bestAcc   = bestEntry.accuracy  ? (bestEntry.accuracy * 100).toFixed(1) + "%" : "—";
  const bestF1    = bestEntry.f1_weighted ? bestEntry.f1_weighted.toFixed(3) : "—";

  // Radar data
  const radarData = Object.entries(taskData)
    .filter(([k]) => k !== "__best_model__")
    .map(([model, m]) => ({
      model, Accuracy: +(m.accuracy * 100).toFixed(1),
      F1: +(m.f1_weighted * 100).toFixed(1), AUC: +(m.roc_auc * 100).toFixed(1),
    }));

  // Feature importance
  let fiBarData = [];
  if (fiData) {
    const bestFI = fiData[bestModel] || fiData[Object.keys(fiData)[0]] || {};
    fiBarData = Object.entries(bestFI)
      .sort(([,a],[,b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 12)
      .map(([feat, val]) => ({ feature: feat.replace(/_/g, " "), importance: Math.abs(parseFloat(val) || 0) }));
  }

  return (
    <div className="page-content fade-in">
      <div className="page-header">
        <div>
          <h2 className="page-title">Analytics & Model Performance</h2>
          <p className="page-desc">Held-out test metrics for each model across the three prediction tasks.</p>
        </div>
        <div className="task-tabs">
          {Object.entries(TASK_LABELS).map(([k, v]) => (
            <button key={k} className={`task-tab ${task === k ? "active" : ""}`} onClick={() => setTask(k)}>{v}</button>
          ))}
        </div>
      </div>

      {/* Stat cards */}
      <div className="stats-grid">
        <StatCard icon={Award}    label="Best Model"     value={bestModel.replace("Logistic Regression","Logistic Reg.")} color="#6366f1" />
        <StatCard icon={TrendingUp} label="Best AUC"    value={bestAUC} sub="vs 0.50 random"  color="#06b6d4" />
        <StatCard icon={Target}   label="Best Accuracy"  value={bestAcc} sub="4-class balanced" color="#f59e0b" />
        <StatCard icon={Zap}      label="Best F1 Score"  value={bestF1}  color="#10b981" />
      </div>

      <div className="charts-grid">
        {/* Bar chart */}
        <div className="card">
          <h3 className="chart-title">Model Comparison — {TASK_LABELS[task]}</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="model" tick={{ fill: "#8899bb", fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#8899bb", fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
              <Tooltip content={<CUSTOM_TOOLTIP />} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#8899bb" }} />
              <Bar dataKey="Accuracy" fill="#6366f1" radius={[4,4,0,0]} />
              <Bar dataKey="F1"       fill="#06b6d4" radius={[4,4,0,0]} />
              <Bar dataKey="AUC"      fill="#10b981" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar chart */}
        <div className="card">
          <h3 className="chart-title">Model Radar — All Metrics</h3>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={[
              { metric: "Accuracy", ...Object.fromEntries(radarData.map(r => [r.model.replace("Logistic Regression","LR").replace("Random Forest","RF"), r.Accuracy])) },
              { metric: "F1 Score", ...Object.fromEntries(radarData.map(r => [r.model.replace("Logistic Regression","LR").replace("Random Forest","RF"), r.F1])) },
              { metric: "AUC-ROC",  ...Object.fromEntries(radarData.map(r => [r.model.replace("Logistic Regression","LR").replace("Random Forest","RF"), r.AUC])) },
            ]}>
              <PolarGrid stroke="rgba(255,255,255,0.07)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#8899bb", fontSize: 12 }} />
              {Object.entries(MODEL_COLORS).map(([model, color]) => (
                <Radar key={model} name={model.replace("Logistic Regression","LR").replace("Random Forest","RF")}
                  dataKey={model.replace("Logistic Regression","LR").replace("Random Forest","RF")}
                  stroke={color} fill={color} fillOpacity={0.08} strokeWidth={2} />
              ))}
              <Legend wrapperStyle={{ fontSize: 11, color: "#8899bb" }} />
              <Tooltip contentStyle={{ background: "#131c30", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 8 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importances */}
        {fiBarData.length > 0 && (
          <div className="card fi-card">
            <h3 className="chart-title">Top Feature Importances — {bestModel} · {TASK_LABELS[task]}</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={fiBarData} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#8899bb", fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fill: "#a5b4fc", fontSize: 11 }} width={140} />
                <Tooltip contentStyle={{ background: "#131c30", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="importance" fill="url(#fiGrad)" radius={[0,4,4,0]} />
                <defs>
                  <linearGradient id="fiGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#06b6d4" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Metrics table */}
        <div className="card">
          <h3 className="chart-title">Full Metrics Table</h3>
          <div style={{ overflowX: "auto" }}>
            <table className="metrics-table">
              <thead>
                <tr><th>Model</th><th>Accuracy</th><th>F1 (w)</th><th>AUC-ROC</th><th>Train Time</th><th></th></tr>
              </thead>
              <tbody>
                {Object.entries(taskData).filter(([k]) => k !== "__best_model__").map(([model, m]) => (
                  <tr key={model} className={model === bestModel ? "best-row" : ""}>
                    <td style={{ color: MODEL_COLORS[model] || "#fff", fontWeight: 600 }}>{model}</td>
                    <td>{(m.accuracy * 100).toFixed(2)}%</td>
                    <td>{m.f1_weighted.toFixed(4)}</td>
                    <td style={{ color: "#10b981", fontWeight: 600 }}>{m.roc_auc == null ? "—" : m.roc_auc.toFixed(4)}</td>
                    <td>{m.train_time_sec.toFixed(2)}s</td>
                    <td>{model === bestModel ? <span className="badge badge-low">⭐ Best</span> : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <style>{`
        .page-header { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 28px; }
        .page-title { font-size: 24px; font-weight: 800; }
        .page-desc { color: var(--text-secondary); font-size: 14px; margin-top: 4px; }
        .task-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
        .task-tab { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-secondary); font-family: inherit; font-size: 13px; font-weight: 600; cursor: pointer; border-radius: var(--radius-sm); padding: 8px 16px; transition: all 0.2s; }
        .task-tab.active { background: var(--accent); color: #fff; border-color: var(--accent); }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        @media (max-width: 900px) { .stats-grid { grid-template-columns: repeat(2,1fr); } }
        @media (max-width: 500px) { .stats-grid { grid-template-columns: 1fr; } }
        .stat-card { padding: 20px; }
        .stat-card-top { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
        .stat-icon { width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
        .stat-label { font-size: 12px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-value { font-size: 26px; font-weight: 800; }
        .stat-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .charts-grid { grid-template-columns: 1fr; } }
        .fi-card,.metrics-table-card { grid-column: 1 / -1; }
        .chart-title { font-size: 14px; font-weight: 700; margin-bottom: 20px; color: var(--text-primary); }
        .metrics-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .metrics-table th { color: var(--text-muted); font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left; }
        .metrics-table td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--text-secondary); }
        .metrics-table .best-row td { background: rgba(99,102,241,0.06); color: var(--text-primary); }
        .fi-card { grid-column: 1 / -1; }
      `}</style>
    </div>
  );
}
