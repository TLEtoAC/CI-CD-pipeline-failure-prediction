// src/AuthContext.jsx
import { createContext, useContext, useState, useEffect } from "react";
import { authAPI } from "./api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => JSON.parse(localStorage.getItem("user") || "null"));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      if (token === "dummy-token") {
        setLoading(false);
        return;
      }
      authAPI.me()
        .then(r => { setUser(r.data); localStorage.setItem("user", JSON.stringify(r.data)); })
        .catch(() => logout())
        .finally(() => setLoading(false));
    } else { setLoading(false); }
  }, []);

  const login = async (email, password) => {
    const { data } = await authAPI.login({ email, password });
    localStorage.setItem("token", data.token);
    localStorage.setItem("user", JSON.stringify(data.user));
    setUser(data.user);
    return data;
  };

  const signup = async (name, email, password) => {
    const { data } = await authAPI.signup({ name, email, password });
    localStorage.setItem("token", data.token);
    localStorage.setItem("user", JSON.stringify(data.user));
    setUser(data.user);
    return data;
  };

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
  };

  const bypassLogin = () => {
    const dummyUser = { id: 'bypass', name: 'Guest', email: 'guest@cicd.ai', role: 'user', predictions: 0 };
    localStorage.setItem("token", "dummy-token");
    localStorage.setItem("user", JSON.stringify(dummyUser));
    setUser(dummyUser);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, signup, logout, bypassLogin }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
