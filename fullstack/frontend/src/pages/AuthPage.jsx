// src/pages/AuthPage.jsx — Login & Signup
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import toast from "react-hot-toast";
import { GitBranch, Lock, Mail, User, Eye, EyeOff, Zap } from "lucide-react";

export default function AuthPage() {
  const [mode, setMode]         = useState("login");
  const [name, setName]         = useState("");
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw]     = useState(false);
  const [loading, setLoading]   = useState(false);
  const { login, signup, bypassLogin } = useAuth();
  const navigate                = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (mode === "login") {
        await login(email, password);
        toast.success("Welcome back!");
      } else {
        await signup(name, email, password);
        toast.success("Account created!");
      }
      navigate("/dashboard");
    } catch (err) {
      toast.error(err.response?.data?.error || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  const fillDemo = () => { setEmail("demo@cicd.ai"); setPassword("demo1234"); };

  const handleBypass = () => {
    bypassLogin();
    navigate("/dashboard");
  };

  return (
    <div className="auth-page">
      <div className="auth-bg">
        <div className="auth-orb auth-orb-1" />
        <div className="auth-orb auth-orb-2" />
        <div className="auth-grid" />
      </div>

      <div className="auth-container fade-in">
        {/* Brand */}
        <div className="auth-brand">
          <div className="auth-logo"><GitBranch size={28} /></div>
          <div>
            <h1 className="auth-title gradient-text">CI/CD Predict</h1>
            <p className="auth-subtitle">ML-Powered Pipeline Intelligence</p>
          </div>
        </div>

        {/* Card */}
        <div className="auth-card glass">
          <div className="auth-tabs">
            <button className={`auth-tab ${mode === "login" ? "active" : ""}`} onClick={() => setMode("login")}>Sign In</button>
            <button className={`auth-tab ${mode === "signup" ? "active" : ""}`} onClick={() => setMode("signup")}>Sign Up</button>
          </div>

          <form onSubmit={handleSubmit} className="auth-form">
            {mode === "signup" && (
              <div className="input-group">
                <label className="input-label">Full Name</label>
                <div className="input-icon-wrap">
                  <User size={16} className="input-icon" />
                  <input id="auth-name" className="input input-with-icon" placeholder="Jane Doe" value={name} onChange={e => setName(e.target.value)} required />
                </div>
              </div>
            )}
            <div className="input-group">
              <label className="input-label">Email</label>
              <div className="input-icon-wrap">
                <Mail size={16} className="input-icon" />
                <input id="auth-email" type="email" className="input input-with-icon" placeholder="you@example.com" value={email} onChange={e => setEmail(e.target.value)} required />
              </div>
            </div>
            <div className="input-group">
              <label className="input-label">Password</label>
              <div className="input-icon-wrap">
                <Lock size={16} className="input-icon" />
                <input id="auth-password" type={showPw ? "text" : "password"} className="input input-with-icon" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} />
                <button type="button" className="input-eye" onClick={() => setShowPw(p => !p)}>{showPw ? <EyeOff size={15}/> : <Eye size={15}/>}</button>
              </div>
            </div>

            <button id="auth-submit" type="submit" className="btn btn-primary auth-submit" disabled={loading}>
              {loading ? <><span className="spinner" /> Processing…</> : mode === "login" ? "Sign In →" : "Create Account →"}
            </button>
          </form>

          {mode === "login" && (
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
              <button className="demo-btn" onClick={fillDemo}>
                <Zap size={13} /> Fill Demo Credentials
              </button>
              <button className="demo-btn" onClick={handleBypass} type="button">
                Bypass Login
              </button>
            </div>
          )}
        </div>

        <p className="auth-footer">Powered by XGBoost · LightGBM · scikit-learn · 45K pipeline logs</p>
      </div>

      <style>{`
        .auth-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 24px; position: relative; overflow: hidden; }
        .auth-bg { position: fixed; inset: 0; pointer-events: none; }
        .auth-orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.4; }
        .auth-orb-1 { width: 500px; height: 500px; background: radial-gradient(circle, #6366f1, transparent); top: -100px; right: -100px; animation: floatOrb 8s ease-in-out infinite; }
        .auth-orb-2 { width: 400px; height: 400px; background: radial-gradient(circle, #06b6d4, transparent); bottom: -80px; left: -80px; animation: floatOrb 10s ease-in-out infinite reverse; }
        .auth-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(99,102,241,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.05) 1px, transparent 1px); background-size: 60px 60px; }
        @keyframes floatOrb { 0%,100%{transform:translate(0,0)} 50%{transform:translate(20px,-20px)} }
        .auth-container { width: 100%; max-width: 420px; position: relative; z-index: 1; display: flex; flex-direction: column; gap: 24px; }
        .auth-brand { display: flex; align-items: center; gap: 14px; }
        .auth-logo { width: 52px; height: 52px; background: linear-gradient(135deg,#6366f1,#4f46e5); border-radius: 14px; display: flex; align-items: center; justify-content: center; color: #fff; box-shadow: 0 4px 20px rgba(99,102,241,0.5); }
        .auth-title { font-size: 26px; font-weight: 800; line-height: 1.1; }
        .auth-subtitle { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }
        .auth-card { border-radius: var(--radius); padding: 28px; display: flex; flex-direction: column; gap: 20px; }
        .auth-tabs { display: flex; background: rgba(0,0,0,0.3); border-radius: var(--radius-sm); padding: 4px; }
        .auth-tab { flex: 1; padding: 8px; border: none; background: transparent; color: var(--text-secondary); font-family: inherit; font-size: 14px; font-weight: 600; cursor: pointer; border-radius: 6px; transition: all 0.2s; }
        .auth-tab.active { background: var(--accent); color: #fff; box-shadow: 0 2px 10px rgba(99,102,241,0.4); }
        .auth-form { display: flex; flex-direction: column; gap: 16px; }
        .input-icon-wrap { position: relative; }
        .input-icon { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: var(--text-muted); pointer-events: none; }
        .input-with-icon { padding-left: 38px; }
        .input-eye { position: absolute; right: 12px; top: 50%; transform: translateY(-50%); background: none; border: none; color: var(--text-muted); cursor: pointer; padding: 2px; }
        .auth-submit { width: 100%; justify-content: center; padding: 13px; font-size: 15px; margin-top: 4px; }
        .demo-btn { background: none; border: 1px dashed var(--border-bright); color: var(--accent); font-family: inherit; font-size: 12px; font-weight: 600; cursor: pointer; border-radius: var(--radius-sm); padding: 8px 14px; display: flex; align-items: center; gap: 6px; margin: 0 auto; transition: all 0.2s; }
        .demo-btn:hover { background: rgba(99,102,241,0.1); }
        .auth-footer { text-align: center; font-size: 11px; color: var(--text-muted); }
      `}</style>
    </div>
  );
}
