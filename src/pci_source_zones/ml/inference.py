"""Score a new fire with an already-trained model — no retraining.

Training workflows produce a prediction raster for the fire they trained on
as a byproduct. This module is the other direction: take a model trained at
one site and apply it to a different fire's aligned feature stack. That is
the operation the multi-fire rollout actually needs, since most fires have
no LiDAR-derived labels to train on.

Two safety properties matter more here than anywhere else in the pipeline,
because there is no target to sanity-check the output against:

1. Feature identity. The bundle records the exact feature names, in order,
   that the model was fit on. If the new config produces a different set or
   a different order, columns would silently map to the wrong variables and
   the output would look plausible but be meaningless. That is checked and
   raises.
2. Grid integrity. The new fire's rasters go through the same data-contract
   check as training (features only — a new fire has no target or splits).

UNet additionally reuses the training-time normalization statistics stored
in its bundle: normalizing a new fire by its *own* statistics would shift
the inputs out of the distribution the network was fit on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config_schema import validate_ml_config
from .data_contract import ContractReport, validate_data_contract
from .features import build_feature_stack, configured_feature_names
from .outputs import write_float_raster, write_uint8_raster
from .predict import predict_probability
from .provenance import new_run_id, write_run_manifest

SKLEARN_SUFFIXES = {".joblib", ".pkl"}
TORCH_SUFFIXES = {".pt", ".pth"}


@dataclass
class ModelBundle:
    """A trained model plus everything needed to reapply it elsewhere."""

    kind: str  # "sklearn" | "unet"
    model: Any
    feature_names: list[str]
    is_regression: bool
    model_name: str
    positive_rule: str | None = None
    norm_stats: dict[str, list[float]] | None = None  # UNet only
    tobit_mode: bool = False
    inference_cfg: dict[str, Any] | None = None  # UNet only: patch_size/overlap
    train_run_id: str | None = None
    path: str = ""


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a trained model saved by any of the training workflows."""
    p = Path(path)
    if p.is_dir():
        p = _find_bundle_in_dir(p)
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found: {p}")

    if p.suffix in SKLEARN_SUFFIXES:
        return _load_sklearn_bundle(p)
    if p.suffix in TORCH_SUFFIXES:
        return _load_unet_bundle(p)
    raise ValueError(
        f"Unrecognized model file type {p.suffix!r}: {p}. "
        f"Expected one of {sorted(SKLEARN_SUFFIXES | TORCH_SUFFIXES)}."
    )


def _find_bundle_in_dir(run_dir: Path) -> Path:
    """Locate the single model artifact inside a training run directory."""
    candidates = [
        c
        for suffix in sorted(SKLEARN_SUFFIXES | TORCH_SUFFIXES)
        for c in sorted(run_dir.glob(f"*{suffix}"))
        # unet_best_weights.pt is a mid-training checkpoint; unet_weights.pt is final
        if c.name != "unet_best_weights.pt"
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No model artifact (*.joblib / *.pt) found in {run_dir}. "
            f"Pass the model file directly if it lives elsewhere."
        )
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise ValueError(
            f"Multiple model artifacts found in {run_dir} ({names}). "
            f"Pass the one you want explicitly."
        )
    return candidates[0]


def _load_sklearn_bundle(path: Path) -> ModelBundle:
    from joblib import load as jload

    raw = jload(path)
    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError(
            f"{path} is not a training-workflow bundle "
            f"(expected a dict with a 'model' key)."
        )
    model = raw["model"]
    return ModelBundle(
        kind="sklearn",
        model=model,
        feature_names=list(raw["feature_names"]),
        is_regression=bool(raw.get("is_regression", _sklearn_is_regressor(model))),
        model_name=str(raw.get("model_name", path.stem)),
        positive_rule=raw.get("positive_rule"),
        train_run_id=raw.get("run_id"),
        path=str(path),
    )


def _sklearn_is_regressor(model: Any) -> bool:
    """Fallback for bundles saved before is_regression was recorded."""
    from sklearn.base import is_regressor

    try:
        return bool(is_regressor(model))
    except Exception:
        return not hasattr(model, "predict_proba")


def _load_unet_bundle(path: Path) -> ModelBundle:
    import torch

    from .models.unet import build_unet

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        raise ValueError(
            f"{path} is not a training-workflow UNet bundle "
            f"(expected a dict with a 'model_state_dict' key)."
        )

    positive_rule = raw.get("positive_rule", "")
    is_regression = bool(raw.get("is_regression", "regression" in str(positive_rule).lower()))

    model = build_unet(
        raw.get("model_cfg", {}),
        in_channels=int(raw["in_channels"]),
        regression=is_regression,
    )
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    return ModelBundle(
        kind="unet",
        model=model,
        feature_names=list(raw["feature_names"]),
        is_regression=is_regression,
        model_name="unet",
        positive_rule=positive_rule,
        norm_stats=raw.get("norm_stats"),
        tobit_mode=bool(raw.get("tobit_mode", False)),
        inference_cfg=raw.get("inference_cfg"),
        train_run_id=raw.get("run_id"),
        path=str(path),
    )


def _check_feature_match(bundle: ModelBundle, cfg_feature_names: list[str]) -> None:
    if list(cfg_feature_names) == list(bundle.feature_names):
        return

    trained, given = set(bundle.feature_names), set(cfg_feature_names)
    missing = sorted(trained - given)
    extra = sorted(given - trained)
    detail = []
    if missing:
        detail.append(f"missing from config: {missing}")
    if extra:
        detail.append(f"not seen during training: {extra}")
    if not detail:
        detail.append(
            f"same features but different order — trained {list(bundle.feature_names)}, "
            f"config gives {list(cfg_feature_names)}"
        )
    raise ValueError(
        "Feature mismatch between the trained model and this config; predictions "
        "would map columns to the wrong variables.\n  - " + "\n  - ".join(detail)
    )


def run_inference(
    cfg: dict[str, Any],
    model_path: str | Path,
    out_dir: str | Path,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Apply a trained model to the feature stack described by cfg.

    cfg only needs paths/base_rasters/feature_paths/features — the target
    and split sections are ignored, since a new fire has neither.
    """
    bundle = load_model_bundle(model_path)

    cfg = validate_ml_config(cfg)
    data_report = validate_data_contract(cfg, features_only=True)

    numeric, categorical = configured_feature_names(cfg)
    _check_feature_match(bundle, numeric + categorical)

    stack = build_feature_stack(cfg)
    profile = stack.profile
    shape = stack.valid_mask.shape

    if threshold is None:
        threshold = float(cfg.get("ml", {}).get("prediction", {}).get("probability_threshold", 0.5))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bundle.kind == "sklearn":
        outputs = _predict_sklearn(bundle, stack, shape, profile, out_dir, threshold)
    else:
        outputs = _predict_unet(bundle, cfg, stack, shape, profile, out_dir, threshold)

    run_id = new_run_id()
    manifest_path = write_run_manifest(cfg, out_dir, run_id, data_report)
    _annotate_inference_manifest(manifest_path, bundle, threshold)

    return {
        "run_id": run_id,
        "model_path": bundle.path,
        "model_name": bundle.model_name,
        "train_run_id": bundle.train_run_id,
        "is_regression": bundle.is_regression,
        "output_dir": out_dir,
        "manifest": manifest_path,
        "feature_names": bundle.feature_names,
        "n_valid_pixels": int(stack.valid_mask.sum()),
        **outputs,
    }


def _predict_sklearn(
    bundle: ModelBundle,
    stack: Any,
    shape: tuple[int, int],
    profile: dict[str, Any],
    out_dir: Path,
    threshold: float,
) -> dict[str, Path]:
    # Reindex defensively: build_feature_stack orders columns by config, and
    # _check_feature_match already proved the sets/order agree, but this makes
    # the column->model mapping explicit rather than positional.
    X = stack.frame[bundle.feature_names]

    if bundle.is_regression:
        values = np.asarray(bundle.model.predict(X), dtype="float32")
        pred_map = np.full(shape, np.nan, dtype="float32")
        pred_map.ravel()[stack.flat_indices] = values
        return {
            "prediction": write_float_raster(
                out_dir / f"{bundle.model_name}_prediction.tif", pred_map, profile
            )
        }

    probs = predict_probability(bundle.model, X)
    prob_map = np.full(shape, np.nan, dtype="float32")
    prob_map.ravel()[stack.flat_indices] = probs

    nodata = 255
    class_map = np.full(shape, nodata, dtype="uint8")
    class_map.ravel()[stack.flat_indices] = (probs >= threshold).astype("uint8")

    label = f"p{threshold:g}".replace(".", "p")
    return {
        "probability": write_float_raster(
            out_dir / f"{bundle.model_name}_source_probability.tif", prob_map, profile
        ),
        "class": write_uint8_raster(
            out_dir / f"{bundle.model_name}_source_class_{label}.tif",
            class_map,
            profile,
            nodata=nodata,
        ),
    }


def _sliding_window_cfg(bundle: ModelBundle, cfg: dict[str, Any]) -> dict[str, Any]:
    """Sliding-window geometry from the bundle, not the caller's config.

    The same weights produce different outputs under a different patch_size /
    overlap, so reusing training's values is what makes inference reproduce
    training. The caller's ml.unet is still consulted for runtime-only
    settings (device), and warns if it tries to override the geometry.
    """
    caller_unet = dict(cfg.get("ml", {}).get("unet", {}) or {})
    stored = bundle.inference_cfg

    if not stored:
        # Bundle predates inference_cfg — fall back to the caller's config and
        # say so, since the result then depends on values we cannot verify.
        print(
            "[predict] WARNING: this model bundle does not record its training "
            "patch_size/overlap. Falling back to the config's ml.unet values "
            f"(patch_size={caller_unet.get('patch_size', 128)}, "
            f"overlap={caller_unet.get('overlap', 0.5)}). These must match the values "
            "used at training or predictions will differ. Retrain to embed them."
        )
        merged = caller_unet
    else:
        for key, value in stored.items():
            if key in caller_unet and caller_unet[key] != value:
                print(
                    f"[predict] WARNING: config sets ml.unet.{key}={caller_unet[key]} but the "
                    f"model was trained with {key}={value}; using the trained value."
                )
        merged = {**caller_unet, **stored}

    return {**cfg, "ml": {**cfg.get("ml", {}), "unet": merged}}


def _predict_unet(
    bundle: ModelBundle,
    cfg: dict[str, Any],
    stack: Any,
    shape: tuple[int, int],
    profile: dict[str, Any],
    out_dir: Path,
    threshold: float,
) -> dict[str, Path]:
    from .patch_dataset import normalize_features, stack_feature_arrays
    from .unet_predict import predict_sliding_window

    if not bundle.norm_stats:
        raise ValueError(
            f"UNet bundle {bundle.path} has no stored norm_stats; cannot normalize "
            f"a new fire consistently with training."
        )

    features_2d = stack_feature_arrays(stack.arrays, bundle.feature_names)
    features_norm = normalize_features(features_2d, bundle.norm_stats)

    raw = predict_sliding_window(
        bundle.model, features_norm, _sliding_window_cfg(bundle, cfg), tobit_mode=bundle.tobit_mode
    )

    outputs: dict[str, Path] = {}
    if bundle.tobit_mode:
        pred_map = raw[0]
        sigma_map = np.exp(np.clip(raw[1], -3, 2))
        outputs["sigma"] = write_float_raster(out_dir / "unet_ceff_sigma.tif", sigma_map, profile)
    else:
        pred_map = raw

    # Only report where features were actually complete; the sliding window
    # runs over a nan_to_num'd array and would otherwise emit values in gaps.
    pred_map = np.where(stack.valid_mask, pred_map, np.nan).astype("float32")

    name = "unet_prediction.tif" if bundle.is_regression else "unet_source_probability.tif"
    outputs["prediction" if bundle.is_regression else "probability"] = write_float_raster(
        out_dir / name, pred_map, profile
    )

    if not bundle.is_regression:
        nodata = 255
        class_map = np.full(shape, nodata, dtype="uint8")
        finite = np.isfinite(pred_map)
        class_map[finite] = (pred_map[finite] >= threshold).astype("uint8")
        label = f"p{threshold:g}".replace(".", "p")
        outputs["class"] = write_uint8_raster(
            out_dir / f"unet_source_class_{label}.tif", class_map, profile, nodata=nodata
        )

    return outputs


def _annotate_inference_manifest(
    manifest_path: Path, bundle: ModelBundle, threshold: float
) -> None:
    """Record which trained model produced these predictions."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mode"] = "inference"
    manifest["source_model"] = {
        "path": bundle.path,
        "model_name": bundle.model_name,
        "kind": bundle.kind,
        "train_run_id": bundle.train_run_id,
        "is_regression": bundle.is_regression,
        "positive_rule": bundle.positive_rule,
        "feature_names": bundle.feature_names,
        "probability_threshold": None if bundle.is_regression else threshold,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
