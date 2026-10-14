const axios = require("axios");
const supabase = require("../supabaseClient");

const ML = process.env.ML_SERVICE_URL || "http://localhost:8001";

const inMemoryHistory = [];

exports.predict = async (req, res) => {
  try {
    const { data } = await axios.post(`${ML}/predict`, req.body, { timeout: 15000 });

    // Authentication bypassed: Using in-memory history instead of Supabase
    const userId = req.user.id;
    inMemoryHistory.unshift({
      id: Date.now().toString(),
      user_id: userId,
      inputs: req.body,
      results: data,
      timestamp: new Date().toISOString()
    });
    if (inMemoryHistory.length > 50) inMemoryHistory.pop();

    res.json(data);
  } catch (err) {
    const status  = err.response?.status || 502;
    const message = err.response?.data?.detail || err.message || "ML service error";
    res.status(status).json({ error: message });
  }
};

exports.getHistory = async (req, res) => {
  try {
    // Authentication bypassed: Using in-memory history instead of Supabase
    const history = inMemoryHistory.filter(h => h.user_id === req.user.id);
    res.json(history);
  } catch (err) {
    console.error("GetHistory error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};
