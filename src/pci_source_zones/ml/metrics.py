from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    if len(np.unique(y_true)) == 2:
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
    else:
        balanced_accuracy = accuracy
        mcc = 0.0

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive": int(np.sum(y_true == 1)),
        "negative": int(np.sum(y_true == 0)),
        "accuracy": accuracy,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out
