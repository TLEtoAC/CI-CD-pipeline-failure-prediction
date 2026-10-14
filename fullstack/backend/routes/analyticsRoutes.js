const express = require("express");
const router = express.Router();
const analyticsController = require("../controllers/analyticsController");
const authMiddleware = require("../middleware/authMiddleware");

router.get("/metrics", authMiddleware, analyticsController.getMetrics);
router.get("/feature-importances/:task", authMiddleware, analyticsController.getFeatureImportances);

module.exports = router;
