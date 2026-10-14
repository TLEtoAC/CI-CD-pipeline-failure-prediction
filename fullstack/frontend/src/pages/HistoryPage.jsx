// src/pages/HistoryPage.jsx
import { useState, useEffect } from "react";
import { predictAPI } from "../api";
import toast from "react-hot-toast";
import { Clock, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";

const RISK_COLORS = { LOW: "#10b981", MEDIUM: "#f59e0b", HIGH: "#ef4444", CRITICAL: "#ff3b6b" };

function HistoryItem({ item }) {
  const [open, setOpen] = useState(false);
  const riskLevel = item.results?.risk_level?.prediction || "—";
  const color = RISK_COLORS[riskLevel] || "var(--accent)";
  const dt = new Date(item.timestamp);

  return (
    <div className="history-item card">
      <div className="history-header" onClick={() => setOpen(p => !p)}>
        <div className="history-meta">
          <div className="history-risk" style={{ color }}>{riskLevel}</div>
          <div className="history-info">
            <span>{item.inputs?.ci_tool} · {item.inputs?.language} · {item.inputs?.branch}</span>
            <span className="history-time"><Clock size={12} /> {dt.toLocaleString()}</span>
          </div>
        </div>
        <div className="history-badges">
          {item.results?.rollback_triggered?.prediction === "1" && <span className="badge badge-high">Rollback</span>}
          {item.results?.incident_created?.prediction === "1" && <span className="badge badge-high">Incident</span>}
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>
      {open && (
        <div className="history-detail fade-in">
          <div className="history-section">
            <div className="history-section-title">Inputs</div>
            <div className="detail-grid">
              {Object.entries(item.inputs || {}).map(([k, v]) => (
                <div key={k} className="detail-item">
                  <span className="detail-key">{k.replace(/_/g, " ")}</span>
                  <span className="detail-val">{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="history-section">
            <div className="history-section-title">Prediction Results</div>
            <div className="detail-grid">
              {Object.entries(item.results || {}).map(([task, r]) => (
                <div key={task} className="detail-item">
                  <span className="detail-key">{task.replace(/_/g, " ")}</span>
                  <span className="detail-val" style={{ color: RISK_COLORS[r.prediction] || "var(--accent)" }}>
                    {r.prediction} {r.confidence ? `(${(r.confidence * 100).toFixed(0)}%)` : ""}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    predictAPI.history()
      .then(r => setHistory(r.data))
      .catch(() => toast.error("Failed to load history"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="page-content fade-in">
      <div className="page-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 28 }}>
        <div>
          <h2 className="page-title">Prediction History</h2>
          <p className="page-desc">Your last 20 predictions, most recent first.</p>
        </div>
        <button className="btn btn-secondary" onClick={load}><RefreshCw size={14} /> Refresh</button>
      </div>

      {loading && <div style={{ display: "flex", justifyContent: "center", padding: 60 }}><div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} /></div>}
      {!loading && history.length === 0 && (
        <div className="card" style={{ textAlign: "center", padding: 60, color: "var(--text-secondary)" }}>
          <Clock size={40} style={{ margin: "0 auto 16px", display: "block", color: "var(--text-muted)" }} />
          <p>No predictions yet. Go to the <strong>Predict</strong> tab to run your first analysis.</p>
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {history.map(item => <HistoryItem key={item.id} item={item} />)}
      </div>

      <style>{`
        .history-item { padding: 16px 20px; cursor: pointer; }
        .history-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
        .history-meta { display: flex; align-items: center; gap: 14px; }
        .history-risk { font-size: 18px; font-weight: 800; min-width: 80px; }
        .history-info { display: flex; flex-direction: column; gap: 3px; }
        .history-info span:first-child { font-size: 14px; font-weight: 500; color: var(--text-primary); }
        .history-time { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--text-muted); }
        .history-badges { display: flex; align-items: center; gap: 8px; color: var(--text-muted); }
        .history-detail { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 16px; }
        .history-section-title { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
        .detail-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
        .detail-item { display: flex; flex-direction: column; gap: 2px; }
        .detail-key { font-size: 11px; color: var(--text-muted); text-transform: capitalize; }
        .detail-val { font-size: 13px; font-weight: 600; color: var(--text-primary); }
        .page-title { font-size: 24px; font-weight: 800; }
        .page-desc { color: var(--text-secondary); font-size: 14px; margin-top: 4px; }
      `}</style>
    </div>
  );
}
