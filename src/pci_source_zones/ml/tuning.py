from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump

from .cv import CVPlan, build_cv_plan
from .dataset import MLData, build_ml_dataset
from .evaluate import evaluate_classifier
from .explain import write_feature_scores
from .features import configured_feature_names
from .models import build_model, model_name_from_config
from .outputs import ml_output_dir, write_json, write_uint8_raster
from .predict import write_prediction_maps
from .workflow import write_split_summary


def run_tuning_workflow(cfg: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    """Tune one classifier on the training split, then evaluate final holdout splits."""

    model_name = model_name_from_config(cfg, model_name)
    artifact_name = f"{model_name}_tuned"
    out_dir = ml_output_dir(cfg, artifact_name)

    data = build_ml_dataset(cfg)
    train_rows = _tuning_rows(cfg, data)
    model = _base_model(cfg, model_name)
    cv_plan = build_cv_plan(cfg, data, train_rows)

    tuned_model, tuning_result = _run_search(cfg, model, data, train_rows, cv_plan)

    prediction_cfg = cfg.get("ml", {}).get("prediction", {})
    threshold = float(prediction_cfg.get("probability_threshold", 0.5))
    exclude_channels = bool(prediction_cfg.get("exclude_channels", True))

    metrics = evaluate_classifier(tuned_model, data, threshold)
    paths = write_prediction_maps(
        tuned_model,
        data,
        out_dir,
        artifact_name,
        threshold,
        exclude_channels,
    )

    model_path = out_dir / f"{artifact_name}_model.joblib"
    dump(
        {
            "model": tuned_model,
            "model_name": model_name,
            "feature_names": list(data.X.columns),
            "positive_rule": data.target_data.positive_rule,
            "best_params": tuning_result["best_params"],
            "cv_method": cv_plan.method,
            "search_method": tuning_result["search_method"],
        },
        model_path,
    )

    metrics_path = write_json(out_dir / f"{artifact_name}_metrics.json", metrics)
    best_path = write_json(out_dir / f"{artifact_name}_best_params.json", tuning_result["best_params"])
    summary_path = write_json(
        out_dir / f"{artifact_name}_tuning_summary.json",
        {
            "model_name": model_name,
            "search_method": tuning_result["search_method"],
            "scoring": tuning_result["scoring"],
            "best_score": tuning_result["best_score"],
            "cv": cv_plan.summary | {"n_splits": cv_plan.n_splits},
            "positive_rule": data.target_data.positive_rule,
        },
    )
    cv_results_path = _write_cv_results(out_dir / f"{artifact_name}_cv_results.csv", tuning_result["cv_results"])
    scores_path = write_feature_scores(
        out_dir / f"{artifact_name}_feature_scores.csv",
        tuned_model,
        list(data.X.columns),
    )
    split_path = write_split_summary(out_dir / "split_summary.csv", data)

    target_path: Path | None = None
    target_cfg = cfg.get("ml", {}).get("target", {})
    if bool(target_cfg.get("export", True)):
        target_path = write_uint8_raster(
            out_dir / "ml_target.tif",
            data.target_data.target,
            data.target_data.profile,
            nodata=int(target_cfg.get("nodata", 255)),
        )

    return {
        "model_name": model_name,
        "output_dir": out_dir,
        "target": target_path,
        "probability": paths["probability"],
        "class": paths["class"],
        "model": model_path,
        "metrics": metrics_path,
        "feature_scores": scores_path,
        "split_summary": split_path,
        "best_params": best_path,
        "tuning_summary": summary_path,
        "cv_results": cv_results_path,
        "positive_rule": data.target_data.positive_rule,
    }


def _base_model(cfg: dict[str, Any], model_name: str):
    numeric, categorical = configured_feature_names(cfg)
    return build_model(model_name, cfg, numeric, categorical)


def _tuning_rows(cfg: dict[str, Any], data: MLData) -> np.ndarray:
    rows = np.asarray(data.splits["train"], dtype=int)
    if bool(cfg.get("ml", {}).get("tuning", {}).get("include_validation_in_tuning", False)):
        if "val" in data.splits:
            rows = np.unique(np.concatenate([rows, data.splits["val"]]).astype(int))
    return rows


def _run_search(
    cfg: dict[str, Any],
    model: Any,
    data: MLData,
    train_rows: np.ndarray,
    cv_plan: CVPlan,
) -> tuple[Any, dict[str, Any]]:
    tuning_cfg = cfg.get("ml", {}).get("tuning", {})
    method = str(tuning_cfg.get("search_method", tuning_cfg.get("method", "random_search"))).lower()
    scoring = str(tuning_cfg.get("scoring", "mcc")).lower()
    search_space = _normalized_search_space(model, tuning_cfg.get("search_space", {}))
    if not search_space:
        raise ValueError("Set ml.tuning.search_space with at least one parameter.")

    if method == "grid_search":
        return _sklearn_search(cfg, model, data, train_rows, cv_plan, search_space, scoring, "grid_search")
    if method == "random_search":
        return _sklearn_search(cfg, model, data, train_rows, cv_plan, search_space, scoring, "random_search")
    if method == "optuna":
        return _optuna_search(cfg, model, data, train_rows, cv_plan, search_space, scoring)

    raise ValueError(f"Unsupported ml.tuning.search_method: {method!r}")


def _sklearn_search(
    cfg: dict[str, Any],
    model: Any,
    data: MLData,
    train_rows: np.ndarray,
    cv_plan: CVPlan,
    search_space: dict[str, Any],
    scoring: str,
    method: str,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    tuning_cfg = cfg.get("ml", {}).get("tuning", {})
    X_train = data.X.iloc[train_rows]
    y_train = data.y[train_rows]
    scorer = scorer_from_name(scoring)
    n_jobs = int(tuning_cfg.get("n_jobs", 1))

    if method == "grid_search":
        search = GridSearchCV(
            model,
            param_grid=search_space,
            scoring=scorer,
            cv=cv_plan.splitter,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
    else:
        n_iter = int(tuning_cfg.get("n_iter", 40))
        n_iter = min(n_iter, _finite_search_size(search_space))
        search = RandomizedSearchCV(
            model,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv_plan.splitter,
            n_jobs=n_jobs,
            random_state=int(tuning_cfg.get("random_state", 42)),
            refit=True,
            error_score="raise",
        )

    search.fit(X_train, y_train, groups=cv_plan.groups)
    return search.best_estimator_, {
        "search_method": method,
        "scoring": scoring,
        "best_score": float(search.best_score_),
        "best_params": _clean_params(search.best_params_),
        "cv_results": pd.DataFrame(search.cv_results_),
    }


def _optuna_search(
    cfg: dict[str, Any],
    model: Any,
    data: MLData,
    train_rows: np.ndarray,
    cv_plan: CVPlan,
    search_space: dict[str, Any],
    scoring: str,
) -> tuple[Any, dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna or set ml.tuning.search_method: random_search.") from exc

    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    tuning_cfg = cfg.get("ml", {}).get("tuning", {})
    X_train = data.X.iloc[train_rows]
    y_train = data.y[train_rows]
    scorer = scorer_from_name(scoring)
    n_trials = int(tuning_cfg.get("n_trials", tuning_cfg.get("n_iter", 40)))
    n_jobs = int(tuning_cfg.get("n_jobs", 1))

    def objective(trial: Any) -> float:
        params = {name: _suggest_param(trial, name, spec) for name, spec in search_space.items()}
        estimator = clone(model)
        estimator.set_params(**params)
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv_plan.splitter,
            groups=cv_plan.groups,
            scoring=scorer,
            n_jobs=n_jobs,
        )
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    tuned_model = clone(model)
    tuned_model.set_params(**study.best_params)
    tuned_model.fit(X_train, y_train)

    rows = []
    for trial in study.trials:
        row = {"number": trial.number, "value": trial.value, "state": str(trial.state)}
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)

    return tuned_model, {
        "search_method": "optuna",
        "scoring": scoring,
        "best_score": float(study.best_value),
        "best_params": _clean_params(study.best_params),
        "cv_results": pd.DataFrame(rows),
    }


def scorer_from_name(name: str):
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        jaccard_score,
        make_scorer,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    clean = name.lower()
    if clean in {"mcc", "matthews_corrcoef"}:
        return make_scorer(matthews_corrcoef)
    if clean in {"balanced_accuracy", "balanced_acc"}:
        return make_scorer(balanced_accuracy_score)
    if clean == "f1":
        return make_scorer(f1_score, zero_division=0)
    if clean == "precision":
        return make_scorer(precision_score, zero_division=0)
    if clean == "recall":
        return make_scorer(recall_score, zero_division=0)
    if clean in {"iou", "jaccard"}:
        return make_scorer(jaccard_score, zero_division=0)
    if clean in {"roc_auc", "average_precision"}:
        return clean
    raise ValueError(f"Unsupported tuning scoring metric: {name!r}")


def _normalized_search_space(model: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in search_space.items():
        out[_model_param_name(model, key)] = value
    return out


def _model_param_name(model: Any, key: str) -> str:
    if "__" in key:
        return key
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return f"model__{key}"
    return key


def _suggest_param(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, dict):
        kind = str(spec.get("type", "categorical")).lower()
        if kind == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        if kind == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step", None),
                log=bool(spec.get("log", False)),
            )
        values = spec.get("values", spec.get("choices", []))
        return trial.suggest_categorical(name, list(values))

    values = spec if isinstance(spec, list) else [spec]
    return trial.suggest_categorical(name, values)


def _finite_search_size(search_space: dict[str, Any]) -> int:
    size = 1
    for values in search_space.values():
        if isinstance(values, dict):
            return 10**9
        if isinstance(values, list):
            size *= max(1, len(values))
        else:
            size *= 1
    return max(1, size)


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        clean_key = key.replace("model__", "")
        if isinstance(value, np.generic):
            value = value.item()
        out[clean_key] = value
    return out


def _write_cv_results(path: Path, results: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if results.empty:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["empty"])
    else:
        results.to_csv(path, index=False)
    return path
