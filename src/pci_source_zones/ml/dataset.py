from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import FeatureStack, build_feature_stack
from .splits import make_splits
from .targets import TargetData, build_target


@dataclass
class MLData:
    X: pd.DataFrame
    y: np.ndarray
    flat_indices: np.ndarray
    splits: dict[str, np.ndarray]
    target_data: TargetData
    feature_stack: FeatureStack


def build_ml_dataset(cfg: dict[str, Any]) -> MLData:
    """Build X, y, and split indices from the shared ML config."""

    target_data = build_target(cfg)
    feature_stack = build_feature_stack(cfg, reference_shape=target_data.target.shape)

    valid = feature_stack.valid_mask & target_data.valid_mask
    flat_indices = np.flatnonzero(valid.ravel())
    if len(flat_indices) == 0:
        raise ValueError("No overlapping valid cells between target and ML features.")

    row_lookup = np.full(target_data.target.size, -1, dtype=int)
    row_lookup[flat_indices] = np.arange(flat_indices.size)

    keep_feature_rows = np.isin(feature_stack.flat_indices, flat_indices)
    X = feature_stack.frame.iloc[keep_feature_rows].reset_index(drop=True)
    y = target_data.target.ravel()[flat_indices].astype("uint8")

    split_flat = make_splits(cfg, target_data.target, valid, target_data.profile)
    splits = {name: row_lookup[idx][row_lookup[idx] >= 0] for name, idx in split_flat.items()}
    _validate_splits(splits, y)

    return MLData(
        X=X,
        y=y,
        flat_indices=flat_indices,
        splits=splits,
        target_data=target_data,
        feature_stack=feature_stack,
    )


def _validate_splits(splits: dict[str, np.ndarray], y: np.ndarray) -> None:
    if "train" not in splits or len(splits["train"]) == 0:
        raise ValueError("No training cells found. Check ml.split and target masks.")
    if len(np.unique(y[splits["train"]])) < 2:
        raise ValueError(
            "Training split has only one class. Use more polygons/cells or adjust target controls."
        )
