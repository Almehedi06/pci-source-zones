from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import MLData
from .outputs import write_float_raster, write_uint8_raster


def predict_probability(model: Any, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype("float32")
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return (1.0 / (1.0 + np.exp(-score))).astype("float32")
    return np.asarray(model.predict(X), dtype="float32")


def write_prediction_maps(
    model: Any,
    data: MLData,
    out_dir: Path,
    model_name: str,
    threshold: float,
    exclude_channels: bool = True,
) -> dict[str, Path]:
    prob_values = predict_probability(model, data.X)

    prob_map = np.full(data.target_data.target.shape, np.nan, dtype="float32")
    prob_map.ravel()[data.flat_indices] = prob_values

    nodata = 255
    class_map = np.full(data.target_data.target.shape, nodata, dtype="uint8")
    class_map.ravel()[data.flat_indices] = (prob_values >= threshold).astype("uint8")

    if exclude_channels:
        prob_map[data.target_data.channel_mask] = np.nan
        class_map[data.target_data.channel_mask] = nodata

    label = _threshold_label(threshold)
    return {
        "probability": write_float_raster(
            out_dir / f"{model_name}_source_probability.tif",
            prob_map,
            data.target_data.profile,
        ),
        "class": write_uint8_raster(
            out_dir / f"{model_name}_source_class_{label}.tif",
            class_map,
            data.target_data.profile,
            nodata=nodata,
        ),
    }


def _threshold_label(threshold: float) -> str:
    return f"p{threshold:g}".replace(".", "p")
