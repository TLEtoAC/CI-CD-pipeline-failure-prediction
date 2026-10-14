// src/App.jsx — Root routing and layout
import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { useAuth } from "./AuthContext";
import Sidebar from "./components/Sidebar";
import Chatbot from "./components/Chatbot";
import AuthPage from "./pages/AuthPage";
import PredictPage from "./pages/PredictPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import HistoryPage from "./pages/HistoryPage";
import LandingPage from "./pages/LandingPage";

function ProtectedLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const sideW = collapsed ? 64 : 220;

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(p => !p)} />
      <main style={{ marginLeft: sideW, flex: 1, transition: "margin-left 0.3s", minHeight: "100vh", background: "var(--bg-primary)" }}>
        <Routes>
          <Route path="/dashboard"  element={<PredictPage />} />
          <Route path="/analytics"  element={<AnalyticsPage />} />
          <Route path="/history"    element={<HistoryPage />} />
          <Route path="*"           element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
      <Chatbot />
    </div>
  );
}

function RequireAuth({ children }) {
  const { user, loading } = useAuth();
  if (loading) return (
    <div style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
    </div>
  );
  return user ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <>
      <Toaster position="top-right" toastOptions={{
        style: { background: "#131c30", color: "#f1f5ff", border: "1px solid rgba(99,102,241,0.3)", borderRadius: 10, fontSize: 14 },
        success: { iconTheme: { primary: "#10b981", secondary: "#131c30" } },
        error:   { iconTheme: { primary: "#ef4444", secondary: "#131c30" } },
      }} />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login"  element={<AuthPage />} />
        <Route path="/*" element={
          <RequireAuth><ProtectedLayout /></RequireAuth>
        } />
      </Routes>
    </>
  );
}
