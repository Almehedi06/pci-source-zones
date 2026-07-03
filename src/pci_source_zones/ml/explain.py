from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def write_feature_scores(path: Path, model: Any, feature_names: list[str]) -> Path:
    names, scores, score_name = _feature_scores(model, feature_names)
    rows = sorted(zip(names, scores), key=lambda item: abs(float(item[1])), reverse=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", score_name])
        writer.writerows(rows)
    return path


def _feature_scores(model: Any, feature_names: list[str]) -> tuple[list[str], np.ndarray, str]:
    final_model = model
    names = list(feature_names)

    if hasattr(model, "named_steps"):
        final_model = model.named_steps.get("model", model)
        preprocessor = model.named_steps.get("preprocess")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            names = [str(name).replace("num__", "").replace("cat__", "") for name in preprocessor.get_feature_names_out()]

    if hasattr(final_model, "feature_importances_"):
        return names, np.asarray(final_model.feature_importances_, dtype="float64"), "importance"

    if hasattr(final_model, "coef_"):
        coef = np.asarray(final_model.coef_, dtype="float64")
        if coef.ndim == 2:
            coef = coef[0]
        return names, coef, "coefficient"

    return names, np.full(len(names), np.nan), "score"
