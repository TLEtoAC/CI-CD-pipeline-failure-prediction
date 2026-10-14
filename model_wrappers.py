"""Joblib-safe helpers shared by training and inference."""

import numpy as np


class OrderedLabelEncoder:
    """Label encoder that keeps an explicit class order (not alphabetical)."""

    def __init__(self, classes):
        self.classes_ = np.asarray(list(classes))
        self._index = {c: i for i, c in enumerate(self.classes_)}

    def fit(self, y=None):
        return self

    def transform(self, y):
        return np.asarray([self._index[str(v)] for v in y], dtype=int)

    def inverse_transform(self, y):
        idx = np.asarray(y).astype(int)
        return self.classes_[idx]


class ThresholdScoreClassifier:
    """Map a continuous risk-index regressor onto ordered tiers.

    ``edges`` are the training-set cut points (length n_classes - 1).
    Class ``k`` is the bin whose right edge is ``edges[k]`` (the last class
    is everything above the final edge).
    """

    def __init__(self, regressor, edges, n_classes=4):
        self.regressor = regressor
        self.edges = np.asarray(edges, dtype=float)
        self.n_classes_ = int(n_classes)
        self.classes_ = np.arange(self.n_classes_)

    def _tier(self, scores):
        return np.digitize(np.asarray(scores, dtype=float), self.edges, right=True)

    def predict(self, X):
        return self._tier(self.regressor.predict(X))

    def predict_proba(self, X):
        scores = np.asarray(self.regressor.predict(X), dtype=float).reshape(-1)
        edges = self.edges
        width = float(np.median(np.diff(edges))) if len(edges) > 1 else 1.0
        width = max(width, 1e-6)
        centers = np.empty(self.n_classes_, dtype=float)
        centers[0] = edges[0] - width / 2.0
        for i in range(1, self.n_classes_ - 1):
            centers[i] = 0.5 * (edges[i - 1] + edges[i])
        centers[-1] = edges[-1] + width / 2.0
        dist = np.abs(scores.reshape(-1, 1) - centers.reshape(1, -1))
        logits = -dist / width
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)
