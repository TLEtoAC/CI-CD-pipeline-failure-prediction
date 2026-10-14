// src/components/Sidebar.jsx
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import { GitBranch, LayoutDashboard, BarChart2, MessageCircle, History, LogOut, ChevronRight, User } from "lucide-react";

const NAV = [
  { to: "/dashboard",  icon: LayoutDashboard, label: "Predict" },
  { to: "/analytics",  icon: BarChart2,        label: "Analytics" },
  { to: "/history",    icon: History,          label: "History" },
];

export default function Sidebar({ collapsed, onToggle }) {
  const { user, logout } = useAuth();
  const navigate         = useNavigate();

  const doLogout = () => { logout(); navigate("/login"); };

  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      {/* Logo */}
      <div className="sb-logo">
        <div className="sb-logo-icon"><GitBranch size={20} /></div>
        {!collapsed && <span className="sb-logo-text gradient-text">CI/CD Predict</span>}
      </div>

      {/* Toggle */}
      <button className="sb-toggle" onClick={onToggle} title={collapsed ? "Expand" : "Collapse"}>
        <ChevronRight size={16} style={{ transform: collapsed ? "none" : "rotate(180deg)", transition: "transform 0.3s" }} />
      </button>

      {/* Navigation */}
      <nav className="sb-nav">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) => `sb-link ${isActive ? "active" : ""}`}>
            <Icon size={18} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* User */}
      <div className="sb-footer">
        <div className="sb-user">
          <div className="sb-avatar"><User size={14} /></div>
          {!collapsed && (
            <div className="sb-user-info">
              <span className="sb-user-name">{user?.name || "User"}</span>
              <span className="sb-user-role">{user?.role || "user"}</span>
            </div>
          )}
        </div>
        <button className="sb-logout" onClick={doLogout} title="Logout">
          <LogOut size={16} />
          {!collapsed && <span>Logout</span>}
        </button>
      </div>

      <style>{`
        .sidebar {
          width: 220px; min-height: 100vh; background: var(--bg-secondary);
          border-right: 1px solid var(--border); display: flex; flex-direction: column;
          padding: 20px 12px; gap: 8px; position: fixed; left: 0; top: 0; bottom: 0;
          transition: width 0.3s ease; z-index: 100;
        }
        .sidebar.collapsed { width: 64px; }
        .sb-logo { display: flex; align-items: center; gap: 10px; padding: 8px 6px 16px; border-bottom: 1px solid var(--border); margin-bottom: 8px; overflow: hidden; }
        .sb-logo-icon { width: 36px; height: 36px; background: linear-gradient(135deg,#6366f1,#4f46e5); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #fff; flex-shrink: 0; }
        .sb-logo-text { font-size: 15px; font-weight: 800; white-space: nowrap; }
        .sb-toggle { position: absolute; top: 74px; right: -12px; width: 24px; height: 24px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; color: var(--text-secondary); transition: all 0.2s; }
        .sb-toggle:hover { color: var(--text-primary); border-color: var(--accent); }
        .sb-nav { display: flex; flex-direction: column; gap: 4px; flex: 1; }
        .sb-link { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: var(--radius-sm); color: var(--text-secondary); text-decoration: none; font-size: 14px; font-weight: 500; transition: all 0.2s; white-space: nowrap; overflow: hidden; }
        .sb-link:hover { color: var(--text-primary); background: rgba(99,102,241,0.1); }
        .sb-link.active { color: var(--text-primary); background: rgba(99,102,241,0.15); border-left: 2px solid var(--accent); padding-left: 10px; }
        .sb-footer { border-top: 1px solid var(--border); padding-top: 12px; display: flex; flex-direction: column; gap: 8px; }
        .sb-user { display: flex; align-items: center; gap: 8px; padding: 6px 8px; overflow: hidden; }
        .sb-avatar { width: 28px; height: 28px; background: linear-gradient(135deg,#6366f1,#06b6d4); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #fff; flex-shrink: 0; }
        .sb-user-info { display: flex; flex-direction: column; overflow: hidden; }
        .sb-user-name { font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .sb-user-role { font-size: 11px; color: var(--text-muted); text-transform: capitalize; }
        .sb-logout { display: flex; align-items: center; gap: 8px; padding: 9px 12px; border: none; background: transparent; color: var(--text-secondary); font-family: inherit; font-size: 13px; font-weight: 500; cursor: pointer; border-radius: var(--radius-sm); transition: all 0.2s; white-space: nowrap; }
        .sb-logout:hover { color: var(--danger); background: rgba(239,68,68,0.1); }
        @media (max-width: 768px) {
          .sidebar { width: 64px; }
          .sb-logo-text, .sb-user-info, .sb-link span, .sb-logout span { display: none; }
        }
      `}</style>
    </aside>
  );
}
