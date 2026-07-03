from __future__ import annotations

from typing import Any


def build_logistic_regression(
    model_cfg: dict[str, Any],
    seed: int,
    numeric: list[str],
    categorical: list[str],
):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    logit_kwargs = {
        "C": float(model_cfg.get("C", 1.0)),
        "class_weight": model_cfg.get("class_weight", "balanced"),
        "max_iter": int(model_cfg.get("max_iter", 1000)),
        "solver": model_cfg.get("solver", "lbfgs"),
        "random_state": seed,
    }
    if "penalty" in model_cfg:
        logit_kwargs["penalty"] = model_cfg["penalty"]

    clf = LogisticRegression(**logit_kwargs)
    return Pipeline([("preprocess", preprocessor), ("model", clf)])
