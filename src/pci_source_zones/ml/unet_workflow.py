from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Subset

from pci_source_zones.config import resolve_data_path

from .dataset import build_ml_dataset
from .evaluate import evaluate_classifier
from .explain import write_feature_scores
from .features import build_feature_stack, configured_feature_names
from .metrics import classification_metrics, regression_metrics
from .models.unet import build_unet
from .outputs import ml_output_dir, write_float_raster, write_json, write_uint8_raster
from .patch_dataset import (
    PatchDataset,
    TobitPatchDataset,
    assign_patch_groups,
    compute_norm_stats,
    leave_one_polygon_out_indices,
    load_norm_stats,
    normalize_features,
    pad_like_patches,
    save_norm_stats,
    stack_feature_arrays,
)
from .preflight import run_preflight
from .provenance import new_run_id, write_run_manifest
from .splits import polygon_group_raster, polygon_train_region_mask
from .targets import build_target
from .unet_predict import extract_pixel_probs, predict_sliding_window
from .unet_train import train_unet, write_training_history


def run_unet_workflow(cfg: dict[str, Any]) -> dict[str, Any]:
    """End-to-end UNet source-zone pipeline.

    Mirrors run_ml_workflow() structure:
      build features → build target → split → normalize →
      create patches → train → predict → evaluate → write outputs

    Handles both classification (BCE+Dice) and regression (MSE / Tobit)
    targets in one place — there is deliberately no separate
    "classify workflow": it branches on is_regression/tobit_mode below.
    """
    cfg, data_report = run_preflight(cfg)
    run_id = new_run_id()
    out_dir = ml_output_dir(cfg, "unet", run_id=run_id)
    write_run_manifest(cfg, out_dir, run_id, data_report)
    unet_cfg = cfg.get("ml", {}).get("unet", {})
    model_cfg = cfg.get("ml", {}).get("model", {})
    target_type = str(cfg.get("ml", {}).get("target", {}).get("type", "physics_dod")).lower()
    is_regression = target_type == "raster_continuous"
    tobit_mode = bool(model_cfg.get("tobit", False))
    nodata_val = -9999 if is_regression else int(cfg.get("ml", {}).get("target", {}).get("nodata", 255))

    # 1. Build feature stack and target (same as RF)
    target_data = build_target(cfg)
    feature_stack = build_feature_stack(cfg, reference_shape=target_data.target.shape)

    numeric, categorical = configured_feature_names(cfg)
    feature_names = numeric + categorical

    # Stack 2D arrays → (C, H, W)
    features_2d = stack_feature_arrays(feature_stack.arrays, feature_names)
    shape = target_data.target.shape

    # 1b. Load Tobit auxiliary layers if requested
    censored_arr: np.ndarray | None = None
    logC_bound_arr: np.ndarray | None = None
    if tobit_mode:
        tobit_paths = unet_cfg.get("tobit", {})
        cen_path   = tobit_paths.get("censored_path")
        bound_path = tobit_paths.get("logC_bound_path")
        if not cen_path or not bound_path:
            raise ValueError("tobit mode requires unet.tobit.censored_path and unet.tobit.logC_bound_path in config")
        cen_path   = resolve_data_path(cfg, cen_path)
        bound_path = resolve_data_path(cfg, bound_path)
        with rasterio.open(cen_path) as src:
            censored_arr = src.read(1).astype("float32")
            if src.nodata is not None:
                censored_arr[censored_arr == src.nodata] = np.nan
        with rasterio.open(bound_path) as src:
            logC_bound_arr = src.read(1).astype("float32")
            if src.nodata is not None:
                logC_bound_arr[logC_bound_arr == src.nodata] = np.nan
        n_cen = int(np.nansum(censored_arr > 0.5))
        print(f"[unet/tobit] censored pixels: {n_cen:,}")
        print(f"[unet/tobit] logC_bound: {np.nanmin(logC_bound_arr):.2f} – {np.nanmax(logC_bound_arr):.2f}")

    # 2. Build dataset for splits (reuse existing split infrastructure)
    data = build_ml_dataset(cfg)
    train_rows = data.splits["train"]
    test_rows = data.splits.get("test", np.array([], dtype=int))

    # 3. Compute normalization stats from train pixels only
    #
    # use_all_pixels / tobit both want to sample patches from every
    # target-valid pixel, not just flat_indices[train_rows] (useful for a
    # sparsely-valid target like C_eff, where requiring full feature
    # completeness discards a lot of otherwise-usable pixels). But
    # target_data.valid_mask alone has no idea which pixels sit inside the
    # train polygons vs the held-out test/val polygons — using it directly
    # let patches (and the norm-stats computed from them) draw straight from
    # test geography, leaking test labels into training gradients. Restrict
    # to the train-polygon footprint first; only fall back to the strictly
    # valid train indices when the split method has no spatial train region
    # to restrict to (e.g. "random", which has no spatial locality at all).
    use_all_pixels = bool(unet_cfg.get("use_all_pixels", False))
    train_region_2d: np.ndarray | None = None
    if tobit_mode or use_all_pixels:
        train_region_2d = polygon_train_region_mask(
            cfg, target_data.profile, shape, cfg.get("ml", {}).get("split", {})
        )

    if tobit_mode and censored_arr is not None:
        # Tobit: sample patches from all hillslope pixels — observed (C_eff finite) + censored
        obs_mask      = target_data.valid_mask  # True where C_eff is observed
        censored_bool = np.nan_to_num(censored_arr, nan=0.0) > 0.5
        train_mask_2d = obs_mask | censored_bool
        if train_region_2d is not None:
            train_mask_2d &= train_region_2d
        else:
            train_mask_2d &= _pixels_in(data.flat_indices[train_rows], shape)
            print("[unet/tobit] no spatial train region for this split method — "
                  "restricting to train-split pixels only to avoid leakage.")
        patch_indices = np.flatnonzero(train_mask_2d.ravel())
        print(f"[unet/tobit] {len(patch_indices):,} pixels for patches "
              f"(obs: {(obs_mask & train_mask_2d).sum():,}, "
              f"censored: {(censored_bool & train_mask_2d).sum():,})")
    elif use_all_pixels:
        if train_region_2d is not None:
            train_mask_2d = target_data.valid_mask & train_region_2d
            print(f"[unet] use_all_pixels=True — using {train_mask_2d.sum():,} valid pixels "
                  f"within the train-polygon region for patches")
        else:
            train_mask_2d = _pixels_in(data.flat_indices[train_rows], shape)
            print(f"[unet] use_all_pixels=True requested but ml.split.method="
                  f"{cfg.get('ml', {}).get('split', {}).get('method')!r} has no spatial train "
                  f"region — falling back to train-split pixels only ({train_mask_2d.sum():,}) "
                  f"to avoid test/val leakage.")
        patch_indices = np.flatnonzero(train_mask_2d.ravel())
    else:
        train_mask_2d = _pixels_in(data.flat_indices[train_rows], shape)
        patch_indices = data.flat_indices[train_rows]

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
    if tobit_mode and censored_arr is not None:
        cen_clean   = np.nan_to_num(censored_arr,   nan=0.0)
        bound_clean = np.nan_to_num(logC_bound_arr, nan=0.0)
        full_train_ds = TobitPatchDataset.from_data_tobit(
            features_norm, target_data.target, cen_clean, bound_clean,
            patch_indices, shape, patch_size=patch_size, stride=stride,
            nodata=float(nodata_val), augment=True,
        )
    else:
        full_train_ds = PatchDataset.from_data(
            features_norm, target_data.target, patch_indices, shape,
            patch_size=patch_size, stride=stride, nodata=nodata_val, augment=True,
        )

    train_idx, val_idx, val_desc = _train_val_patch_indices(
        cfg, full_train_ds.locations, patch_size, shape, target_data.profile, val_frac
    )
    n_train, n_val = len(train_idx), len(val_idx)
    if n_train == 0 or n_val == 0:
        raise ValueError(
            f"Patch train/val split produced train={n_train}, val={n_val}. "
            f"Lower ml.unet.patch_size or raise overlap so patches fit inside the polygons."
        )

    train_ds = Subset(full_train_ds, train_idx)

    if tobit_mode and censored_arr is not None:
        val_ds_raw = TobitPatchDataset.from_data_tobit(
            features_norm, target_data.target, cen_clean, bound_clean,
            patch_indices, shape, patch_size=patch_size, stride=stride,
            nodata=float(nodata_val), augment=False,
        )
    else:
        val_ds_raw = PatchDataset.from_data(
            features_norm, target_data.target, patch_indices, shape,
            patch_size=patch_size, stride=stride, nodata=nodata_val, augment=False,
        )
    val_ds = Subset(val_ds_raw, val_idx)

    print(f"[unet] Patches — train: {n_train}, val: {n_val}, patch_size: {patch_size}")
    print(f"[unet] Validation: {val_desc}")

    # 5. Build and train UNet
    in_channels = features_2d.shape[0]
    model = build_unet(model_cfg, in_channels=in_channels, regression=is_regression)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unet] Model: {n_params:,} trainable parameters, {in_channels} input channels"
          + (" [tobit 2-head]" if tobit_mode else ""))

    model, history = train_unet(model, train_ds, val_ds, cfg, out_dir,
                                is_regression=is_regression, tobit_mode=tobit_mode)

    history_path = write_training_history(history, out_dir / "unet_training_history.csv")

    # 6. Save model weights
    weights_path = out_dir / "unet_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "in_channels": in_channels,
            "feature_names": feature_names,
            "positive_rule": target_data.positive_rule,
            "norm_stats": norm_stats,
            "tobit_mode": tobit_mode,
            "is_regression": is_regression,
            "run_id": run_id,
            # Sliding-window geometry is part of the model's identity: the same
            # weights produce different outputs under a different patch_size /
            # overlap, so inference must reuse these rather than whatever config
            # it happens to be handed.
            "inference_cfg": {
                "patch_size": int(unet_cfg.get("patch_size", 128)),
                "overlap": float(unet_cfg.get("overlap", 0.5)),
            },
        },
        weights_path,
    )

    # 7. Sliding window prediction → mu (and sigma for Tobit)
    raw_pred = predict_sliding_window(model, features_norm, cfg, tobit_mode=tobit_mode)

    if tobit_mode:
        mu_map    = raw_pred[0]
        lsig_map  = raw_pred[1]
        sigma_map = np.exp(np.clip(lsig_map, -3, 2))
        pred_map  = mu_map
        write_float_raster(out_dir / "unet_ceff_sigma.tif", sigma_map, target_data.profile, nodata=-9999.0)
    else:
        pred_map = raw_pred

    out_fname = "unet_prediction.tif" if is_regression else "unet_source_probability.tif"
    pred_path = write_float_raster(
        out_dir / out_fname,
        pred_map,
        target_data.profile,
        nodata=-9999.0,
    )

    # 8. Binary class raster at configured threshold (classification mode only)
    class_path = None
    if not is_regression:
        threshold = float(cfg.get("ml", {}).get("prediction", {}).get("probability_threshold", 0.5))
        class_map = np.full(shape, nodata_val, dtype="uint8")
        valid_prob = np.isfinite(pred_map)
        class_map[valid_prob] = (pred_map[valid_prob] >= threshold).astype("uint8")
        if bool(cfg.get("ml", {}).get("prediction", {}).get("exclude_channels", True)):
            class_map[target_data.channel_mask] = nodata_val
        label = f"p{threshold:g}".replace(".", "p")
        class_path = write_uint8_raster(
            out_dir / f"unet_source_class_{label}.tif",
            class_map,
            target_data.profile,
            nodata=nodata_val,
        )

    # 9. Evaluate on test/train pixels
    metrics: dict[str, Any] = {}
    if len(test_rows) > 0:
        test_flat = data.flat_indices[test_rows]
        y_test = data.y[test_rows]
        preds_test = extract_pixel_probs(pred_map, test_flat, shape)
        valid_test = np.isfinite(preds_test) & np.isfinite(y_test.astype("float32"))
        if valid_test.any():
            if is_regression:
                metrics["test"] = regression_metrics(y_test[valid_test].astype("float32"), preds_test[valid_test])
            else:
                pred_cls = (preds_test[valid_test] >= threshold).astype("uint8")
                metrics["test"] = classification_metrics(y_test[valid_test], pred_cls, preds_test[valid_test])

    if len(train_rows) > 0:
        train_flat = data.flat_indices[train_rows]
        y_train = data.y[train_rows]
        preds_train = extract_pixel_probs(pred_map, train_flat, shape)
        valid_train = np.isfinite(preds_train) & np.isfinite(y_train.astype("float32"))
        if valid_train.any():
            if is_regression:
                metrics["train"] = regression_metrics(y_train[valid_train].astype("float32"), preds_train[valid_train])
            else:
                pred_cls = (preds_train[valid_train] >= threshold).astype("uint8")
                metrics["train"] = classification_metrics(y_train[valid_train], pred_cls, preds_train[valid_train])

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
        "run_id": run_id,
        "output_dir": out_dir,
        "target": target_path,
        "prediction": pred_path,
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
        "validation": val_desc,
    }


def _pixels_in(flat_indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[flat_indices] = True
    return mask


def _train_val_patch_indices(
    cfg: dict[str, Any],
    locations: list[tuple[int, int]],
    patch_size: int,
    shape: tuple[int, int],
    profile: dict[str, Any],
    val_frac: float,
) -> tuple[list[int], list[int], str]:
    """Choose which patches train and which validate.

    Prefers leave-one-polygon-out: validation patches come from a whole
    held-out train polygon, and any patch touching that polygon is excluded
    from training. A random patch split (the alternative) puts neighbouring,
    strongly autocorrelated patches on both sides, so val loss reads
    optimistically and early stopping / HP choices are made on an inflated
    signal.

    Controlled by ml.unet.val_method: auto (default) | polygon_groups | random,
    and ml.unet.val_polygon_id to pick which polygon is held out.
    """
    unet_cfg = cfg.get("ml", {}).get("unet", {}) or {}
    split_cfg = cfg.get("ml", {}).get("split", {})
    method = str(unet_cfg.get("val_method", "auto")).lower()
    seed = int(split_cfg.get("seed", 42))

    if method in {"auto", "polygon_groups"}:
        grouped = polygon_group_raster(cfg, profile, shape, split_cfg)
        if grouped is None:
            if method == "polygon_groups":
                raise ValueError(
                    "ml.unet.val_method: polygon_groups requires ml.split.method: polygons "
                    "with a polygons.path and train_ids."
                )
        else:
            group_raster, ids = grouped
            padded = pad_like_patches(group_raster, shape, patch_size, fill=0)
            dominant, touches = assign_patch_groups(locations, patch_size, padded)

            held_out = _choose_held_out_group(unet_cfg, ids, dominant)
            train_idx, val_idx = leave_one_polygon_out_indices(dominant, touches, held_out)
            dropped = len(locations) - len(train_idx) - len(val_idx)
            desc = (
                f"leave-one-polygon-out, holding out polygon id={ids[held_out - 1]} "
                f"({len(val_idx)} val patches; {dropped} boundary-straddling patches dropped "
                f"from both sides)"
            )
            return train_idx, val_idx, desc

    # Fallback: random patch split (spatially leaky — only for non-polygon splits)
    n_total = len(locations)
    n_val = max(1, int(n_total * val_frac))
    idx = np.random.default_rng(seed).permutation(n_total)
    desc = (
        f"random {val_frac:.0%} patch split — NOTE: spatially autocorrelated with train, "
        f"so val loss is optimistic (no polygon split available for this config)"
    )
    return idx[n_val:].tolist(), idx[:n_val].tolist(), desc


def _choose_held_out_group(
    unet_cfg: dict[str, Any],
    ids: list[Any],
    dominant: np.ndarray,
) -> int:
    """Pick which polygon (1-based group label) to hold out for validation."""
    requested = unet_cfg.get("val_polygon_id")
    if requested is not None:
        if requested not in ids:
            raise ValueError(
                f"ml.unet.val_polygon_id={requested!r} is not one of the train polygon ids {ids}."
            )
        return ids.index(requested) + 1

    # Default: the polygon holding the most patches, so the held-out fold is
    # large enough to give a usable validation signal.
    counts = np.bincount(dominant, minlength=len(ids) + 1)
    counts[0] = 0
    return int(counts.argmax())
