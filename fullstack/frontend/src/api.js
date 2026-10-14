// src/api.js — Centralized API client for CI/CD Dashboard

import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:3001/api",
  timeout: 20000,
});

// Attach JWT token to every request
API.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Auto-logout on 401
API.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

// ── Auth ─────────────────────────────────────────────────────────────────────
export const authAPI = {
  login:  (data) => API.post("/auth/login",  data),
  signup: (data) => API.post("/auth/signup", data),
  me:     ()     => API.get("/auth/me"),
};

// ── Prediction ───────────────────────────────────────────────────────────────
export const predictAPI = {
  predict: (data)    => API.post("/predict",       data),
  history: ()        => API.get("/history"),
};

// ── Analytics ────────────────────────────────────────────────────────────────
export const analyticsAPI = {
  metrics:             ()       => API.get("/analytics/metrics"),
  featureImportances:  (task)   => API.get(`/analytics/feature-importances/${task}`),
};

// ── Chatbot ──────────────────────────────────────────────────────────────────
export const chatAPI = {
  send: (message, history) => API.post("/chat", { message, history }),
};

export default API;
