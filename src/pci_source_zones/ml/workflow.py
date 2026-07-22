from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from joblib import dump

from .dataset import MLData, build_ml_dataset
from .evaluate import evaluate_classifier, evaluate_regressor
from .explain import write_feature_scores
from .features import configured_feature_names
from .models import model_name_from_config
from .outputs import ml_output_dir, write_float_raster, write_json, write_uint8_raster
from .predict import write_prediction_maps
from .train import train_classifier


def run_ml_workflow(cfg: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    model_name = model_name_from_config(cfg, model_name)
    out_dir = ml_output_dir(cfg, model_name)

    target_cfg = cfg.get("ml", {}).get("target", {})
    is_regression = str(target_cfg.get("type", "")).lower() == "raster_continuous"

    data = build_ml_dataset(cfg)
    model = train_classifier(model_name, cfg, data)

    prediction_cfg = cfg.get("ml", {}).get("prediction", {})
    threshold = float(prediction_cfg.get("probability_threshold", 0.5))
    exclude_channels = bool(prediction_cfg.get("exclude_channels", True))

    metrics = evaluate_regressor(model, data) if is_regression else evaluate_classifier(model, data, threshold)
    paths = write_prediction_maps(model, data, out_dir, model_name, threshold, exclude_channels)

    model_path = out_dir / f"{model_name}_model.joblib"
    dump(
        {
            "model": model,
            "model_name": model_name,
            "feature_names": list(data.X.columns),
            "positive_rule": data.target_data.positive_rule,
        },
        model_path,
    )

    metrics_path = write_json(out_dir / f"{model_name}_metrics.json", metrics)
    scores_path = write_feature_scores(
        out_dir / f"{model_name}_feature_scores.csv",
        model,
        list(data.X.columns),
    )
    split_path = write_split_summary(out_dir / "split_summary.csv", data)

    target_path: Path | None = None
    if bool(target_cfg.get("export", True)):
        if is_regression:
            target_path = write_float_raster(
                out_dir / "ml_target.tif",
                data.target_data.target.astype("float32"),
                data.target_data.profile,
                nodata=-9999.0,
            )
        else:
            target_path = write_uint8_raster(
                out_dir / "ml_target.tif",
                data.target_data.target,
                data.target_data.profile,
                nodata=int(target_cfg.get("nodata", 255)),
            )

    numeric, categorical = configured_feature_names(cfg)
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
        "feature_names": list(data.X.columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "positive_rule": data.target_data.positive_rule,
    }


def write_split_summary(path: Path, data: MLData) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n", "positive", "negative"])
        for name, rows in data.splits.items():
            yy = data.y[rows]
            writer.writerow([name, len(rows), int((yy == 1).sum()), int((yy == 0).sum())])
    return path
