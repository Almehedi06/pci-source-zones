from __future__ import annotations

from typing import Any


def build_random_forest(model_cfg: dict[str, Any], seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=model_cfg.get("max_depth", None),
        min_samples_split=int(model_cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
        max_features=model_cfg.get("max_features", "sqrt"),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
