from __future__ import annotations

from typing import Any

from .dataset import MLData
from .features import configured_feature_names
from .models import build_model


def train_classifier(model_name: str, cfg: dict[str, Any], data: MLData):
    numeric, categorical = configured_feature_names(cfg)
    model = build_model(model_name, cfg, numeric, categorical)
    train_rows = data.splits["train"]
    model.fit(data.X.iloc[train_rows], data.y[train_rows])
    return model
