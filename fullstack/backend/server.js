/**
 * server.js — Express backend for CI/CD Failure Prediction Dashboard
 * Handles: Auth (JWT), Prediction proxy → ML Service, AI Chatbot
 */

require("dotenv").config();
const express    = require("express");
const cors       = require("cors");
const helmet     = require("helmet");
const rateLimit  = require("express-rate-limit");

const authRoutes = require("./routes/authRoutes");
const predictRoutes = require("./routes/predictRoutes");
const analyticsRoutes = require("./routes/analyticsRoutes");
const chatRoutes = require("./routes/chatRoutes");

const app  = express();
const PORT = process.env.PORT || 3001;
const ML   = process.env.ML_SERVICE_URL || "http://localhost:8001";

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*", credentials: true }));
app.use(express.json());

// Rate limiting
const authLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 20, message: "Too many requests" });
const apiLimiter  = rateLimit({ windowMs: 1  * 60 * 1000, max: 60, message: "Too many requests" });
app.use("/api/auth", authLimiter);
app.use("/api",      apiLimiter);

// ── Routes ────────────────────────────────────────────────────────────────────
app.use("/api/auth", authRoutes);
app.use("/api", predictRoutes);
app.use("/api/analytics", analyticsRoutes);
app.use("/api/chat", chatRoutes);

// ── Health ────────────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => res.json({ status: "ok", timestamp: new Date().toISOString() }));

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`✅ Backend running  → http://localhost:${PORT}`);
  console.log(`🤖 ML service proxy → ${ML}`);
});
