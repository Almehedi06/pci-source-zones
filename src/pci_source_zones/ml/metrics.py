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

    classes = sorted(np.unique(y_true).tolist())
    n_classes = len(classes)
    is_binary = n_classes == 2

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if is_binary else 0.0

    average = "binary" if is_binary else "macro"
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_classes": n_classes,
        "classes": classes,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }

    if is_binary:
        out["positive"] = int(np.sum(y_true == 1))
        out["negative"] = int(np.sum(y_true == 0))
        out["mcc"] = mcc
        if y_prob is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        # Per-class breakdown for multiclass
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
        out["per_class"] = {
            str(cls): {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i]),
                "n": int(np.sum(y_true == cls)),
            }
            for i, cls in enumerate(classes)
        }

    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    n = len(y_t)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}
