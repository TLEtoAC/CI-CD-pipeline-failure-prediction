const axios = require("axios");

const ML = process.env.ML_SERVICE_URL || "http://localhost:8001";

exports.getMetrics = async (req, res) => {
  try {
    const { data } = await axios.get(`${ML}/metrics`, { timeout: 10000 });
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: "ML service unavailable" });
  }
};

exports.getFeatureImportances = async (req, res) => {
  try {
    const { data } = await axios.get(`${ML}/feature-importances/${req.params.task}`, { timeout: 10000 });
    res.json(data);
  } catch (err) {
    const status = err.response?.status || 502;
    res.status(status).json({ error: err.response?.data?.detail || "ML service error" });
  }
};
