from __future__ import annotations

from typing import Any

from .logistic_regression import build_logistic_regression
from .random_forest import build_random_forest
from .xgboost import build_xgboost


def model_name_from_config(cfg: dict[str, Any], override: str | None = None) -> str:
    if override:
        return normalize_model_name(override)
    return normalize_model_name(cfg.get("ml", {}).get("model", {}).get("type", "random_forest"))


def normalize_model_name(name: str) -> str:
    clean = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest_classifier": "random_forest",
        "logistic": "logistic_regression",
        "logit": "logistic_regression",
        "lr": "logistic_regression",
        "xgb": "xgboost",
        "xgboost_classifier": "xgboost",
    }
    return aliases.get(clean, clean)


def build_model(model_name: str, cfg: dict[str, Any], numeric: list[str], categorical: list[str]):
    """Create one configured classifier by model name."""

    model_name = normalize_model_name(model_name)
    model_cfg = cfg.get("ml", {}).get("model", {})
    seed = int(model_cfg.get("random_state", cfg.get("ml", {}).get("split", {}).get("seed", 42)))

    if model_name == "random_forest":
        return build_random_forest(model_cfg, seed)
    if model_name == "logistic_regression":
        return build_logistic_regression(model_cfg, seed, numeric, categorical)
    if model_name == "xgboost":
        return build_xgboost(model_cfg, seed)

    raise ValueError(f"Unsupported ML model: {model_name!r}")
