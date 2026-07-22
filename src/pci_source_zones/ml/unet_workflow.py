from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import build_ml_dataset
from .evaluate import evaluate_classifier
from .explain import write_feature_scores
from .features import build_feature_stack, configured_feature_names
# from .metrics import classification_metrics  # classification mode
from .metrics import regression_metrics  # regression mode
from .models.unet import build_unet
from .outputs import ml_output_dir, write_float_raster, write_json, write_uint8_raster
from .patch_dataset import (
    PatchDataset,
    compute_norm_stats,
    load_norm_stats,
    normalize_features,
    save_norm_stats,
    stack_feature_arrays,
)
from .targets import build_target
from .unet_predict import extract_pixel_probs, predict_sliding_window
from .unet_train import train_unet, write_training_history


def run_unet_workflow(cfg: dict[str, Any]) -> dict[str, Any]:
    """End-to-end UNet source-zone pipeline.

    Mirrors run_ml_workflow() structure:
      build features → build target → split → normalize →
      create patches → train → predict → evaluate → write outputs
    """
    out_dir = ml_output_dir(cfg, "unet")
    unet_cfg = cfg.get("ml", {}).get("unet", {})
    target_type = str(cfg.get("ml", {}).get("target", {}).get("type", "physics_dod")).lower()
    is_regression = target_type == "raster_continuous"
    nodata_val = -9999 if is_regression else int(cfg.get("ml", {}).get("target", {}).get("nodata", 255))

    # 1. Build feature stack and target (same as RF)
    target_data = build_target(cfg)
    feature_stack = build_feature_stack(cfg, reference_shape=target_data.target.shape)

    numeric, categorical = configured_feature_names(cfg)
    feature_names = numeric + categorical

    # Stack 2D arrays → (C, H, W)
    features_2d = stack_feature_arrays(feature_stack.arrays, feature_names)
    shape = target_data.target.shape

    # 2. Build dataset for splits (reuse existing split infrastructure)
    data = build_ml_dataset(cfg)
    train_rows = data.splits["train"]
    test_rows = data.splits.get("test", np.array([], dtype=int))

    # 3. Compute normalization stats from train pixels only
    train_mask_2d = np.zeros(shape, dtype=bool)
    train_mask_2d.ravel()[data.flat_indices[train_rows]] = True

    norm_stats_path = out_dir / "unet_norm_stats.json"
    if bool(unet_cfg.get("load_norm_stats", False)) and norm_stats_path.exists():
        norm_stats = load_norm_stats(norm_stats_path)
        print(f"[unet] Loaded norm stats from {norm_stats_path}")
    else:
        norm_stats = compute_norm_stats(features_2d, train_mask_2d)
        save_norm_stats(norm_stats, norm_stats_path)

    features_norm = normalize_features(features_2d, norm_stats)

    # 4. Build train / val patch datasets
    patch_size = int(unet_cfg.get("patch_size", 128))
    overlap = float(unet_cfg.get("overlap", 0.5))
    stride = max(1, int(patch_size * (1.0 - overlap)))
    val_frac = float(unet_cfg.get("val_fraction", 0.15))

    # Split train locations into train/val by shuffling patch indices
    full_train_ds = PatchDataset.from_data(
        features_norm,
        target_data.target,
        data.flat_indices[train_rows],
        shape,
        patch_size=patch_size,
        stride=stride,
        nodata=nodata_val,
        augment=True,
    )

    n_total = len(full_train_ds)
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val
    rng = np.random.default_rng(int(cfg.get("ml", {}).get("split", {}).get("seed", 42)))
    idx = rng.permutation(n_total)

    from torch.utils.data import Subset

    train_ds = Subset(full_train_ds, idx[:n_train].tolist())
    val_ds_raw = PatchDataset.from_data(
        features_norm,
        target_data.target,
        data.flat_indices[train_rows],
        shape,
        patch_size=patch_size,
        stride=stride,
        nodata=nodata_val,
        augment=False,
    )
    val_ds = Subset(val_ds_raw, idx[n_train:].tolist())

    print(f"[unet] Patches — train: {n_train}, val: {n_val}, patch_size: {patch_size}")

    # 5. Build and train UNet
    model_cfg = cfg.get("ml", {}).get("model", {})
    in_channels = features_2d.shape[0]
    model = build_unet(model_cfg, in_channels=in_channels)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unet] Model: {n_params:,} trainable parameters, {in_channels} input channels")

    model, history = train_unet(model, train_ds, val_ds, cfg, out_dir)

    history_path = write_training_history(history, out_dir / "unet_training_history.csv")

    # 6. Save model weights
    import torch

    weights_path = out_dir / "unet_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "in_channels": in_channels,
            "feature_names": feature_names,
            "positive_rule": target_data.positive_rule,
            "norm_stats": norm_stats,
        },
        weights_path,
    )

    # 7. Sliding window prediction → probability raster
    prob_map = predict_sliding_window(model, features_norm, cfg)

    prob_path = write_float_raster(
        out_dir / "unet_source_probability.tif",
        prob_map,
        target_data.profile,
        nodata=-9999.0,
    )

    # 8. Binary class raster at configured threshold (classification mode)
    # threshold = float(cfg.get("ml", {}).get("prediction", {}).get("probability_threshold", 0.5))
    # class_map = np.full(shape, nodata_val, dtype="uint8")
    # valid_prob = np.isfinite(prob_map)
    # class_map[valid_prob] = (prob_map[valid_prob] >= threshold).astype("uint8")
    # if bool(cfg.get("ml", {}).get("prediction", {}).get("exclude_channels", True)):
    #     class_map[target_data.channel_mask] = nodata_val
    # label = f"p{threshold:g}".replace(".", "p")
    # class_path = write_uint8_raster(
    #     out_dir / f"unet_source_class_{label}.tif",
    #     class_map,
    #     target_data.profile,
    #     nodata=nodata_val,
    # )
    class_path = None  # regression mode: continuous prediction only, no binary class map

    # 9. Evaluate on test/train pixels
    metrics: dict[str, Any] = {}
    if len(test_rows) > 0:
        test_flat = data.flat_indices[test_rows]
        y_test = data.y[test_rows]
        prob_test = extract_pixel_probs(prob_map, test_flat, shape)
        valid_test = np.isfinite(prob_test) & np.isfinite(y_test.astype("float32"))
        if valid_test.any():
            # classification mode:
            # pred_test = (prob_test[valid_test] >= threshold).astype("uint8")
            # metrics["test"] = classification_metrics(y_test[valid_test], pred_test, prob_test[valid_test])
            metrics["test"] = regression_metrics(y_test[valid_test].astype("float32"), prob_test[valid_test])

    if len(train_rows) > 0:
        train_flat = data.flat_indices[train_rows]
        y_train = data.y[train_rows]
        prob_train = extract_pixel_probs(prob_map, train_flat, shape)
        valid_train = np.isfinite(prob_train) & np.isfinite(y_train.astype("float32"))
        if valid_train.any():
            # classification mode:
            # pred_train = (prob_train[valid_train] >= threshold).astype("uint8")
            # metrics["train"] = classification_metrics(y_train[valid_train], pred_train, prob_train[valid_train])
            metrics["train"] = regression_metrics(y_train[valid_train].astype("float32"), prob_train[valid_train])

    metrics_path = write_json(out_dir / "unet_metrics.json", metrics)

    # 10. Export target raster
    target_path: Path | None = None
    target_cfg = cfg.get("ml", {}).get("target", {})
    if bool(target_cfg.get("export", True)):
        if is_regression:
            target_path = write_float_raster(
                out_dir / "ml_target.tif",
                target_data.target.astype("float32"),
                target_data.profile,
                nodata=-9999.0,
            )
        else:
            target_path = write_uint8_raster(
                out_dir / "ml_target.tif",
                target_data.target,
                target_data.profile,
                nodata=nodata_val,
            )

    return {
        "model_name": "unet",
        "output_dir": out_dir,
        "target": target_path,
        "probability": prob_path,
        "class": class_path,
        "weights": weights_path,
        "norm_stats": norm_stats_path,
        "metrics": metrics_path,
        "training_history": history_path,
        "positive_rule": target_data.positive_rule,
        "feature_names": feature_names,
        "in_channels": in_channels,
        "n_train_patches": n_train,
        "n_val_patches": n_val,
    }
