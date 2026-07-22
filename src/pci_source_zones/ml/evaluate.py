from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import MLData
from .metrics import classification_metrics, regression_metrics
from .predict import predict_probability


def evaluate_classifier(model: Any, data: MLData, threshold: float) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split_name, rows in data.splits.items():
        if len(rows) == 0:
            continue
        prob = predict_probability(model, data.X.iloc[rows])
        pred = (prob >= threshold).astype("uint8")
        metrics[split_name] = classification_metrics(data.y[rows], pred, prob)
    return metrics


def evaluate_regressor(model: Any, data: MLData) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split_name, rows in data.splits.items():
        if len(rows) == 0:
            continue
        pred = np.asarray(model.predict(data.X.iloc[rows]), dtype="float32")
        y_true = data.y[rows].astype("float32")
        metrics[split_name] = regression_metrics(y_true, pred)
    return metrics
