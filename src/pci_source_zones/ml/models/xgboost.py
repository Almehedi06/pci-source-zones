from __future__ import annotations

from typing import Any


def build_xgboost(model_cfg: dict[str, Any], seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost to use ml.model.type: xgboost.") from exc

    return XGBClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=int(model_cfg.get("max_depth", 5)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )


def build_xgboost_regressor(model_cfg: dict[str, Any], seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost to use ml.model.type: xgboost_regressor.") from exc

    return XGBRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=int(model_cfg.get("max_depth", 6)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        min_child_weight=int(model_cfg.get("min_child_weight", 10)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(model_cfg.get("reg_lambda", 1.0)),
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
