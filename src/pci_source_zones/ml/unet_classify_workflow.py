"""UNet classification pipeline for postfire debris-flow source zones.

Standalone — no shared logic with the regression workflow.
Target: binary 0/1 raster (physics_dod or raster type).
Loss:   BCE + Dice with positive-class upweighting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import build_ml_dataset
from .features import build_feature_stack, configured_feature_names
from .metrics import classification_metrics
from .models.unet import build_unet
from .outputs import ml_output_dir, write_json, write_uint8_raster, write_float_raster
from .patch_dataset import (
    PatchDataset,
    compute_norm_stats,
    save_norm_stats,
    normalize_features,
    stack_feature_arrays,
)
from .targets import build_target
from .unet_predict import extract_pixel_probs, predict_sliding_window
from .unet_train import write_training_history


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _bce_dice_loss(pred, target, valid_mask, pos_weight: float = 3.0):
    import torch
    import torch.nn.functional as F

    p = pred.squeeze(1)          # (B, H, W)
    eps = 1e-6

    weight = torch.where(
        target == 1.0,
        torch.tensor(pos_weight, device=p.device, dtype=torch.float32),
        torch.ones_like(p),
    )
    bce_all = F.binary_cross_entropy(p, target, weight=weight, reduction="none")
    bce = (bce_all * valid_mask.float()).sum() / (valid_mask.float().sum() + eps)

    p_v = p[valid_mask]
    t_v = target[valid_mask]
    intersection = (p_v * t_v).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (p_v.sum() + t_v.sum() + eps)

    return 0.5 * bce + 0.5 * dice


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(model, train_loader, val_loader, cfg, out_dir: Path, pos_weight: float):
    import torch
    import torch.optim as optim

    unet_cfg = cfg.get("ml", {}).get("unet", {})
    epochs    = int(unet_cfg.get("epochs", 50))
    lr        = float(unet_cfg.get("learning_rate", 0.001))
    patience  = int(unet_cfg.get("early_stopping_patience", 10))
    device    = _get_device(unet_cfg)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    best_loss = float("inf")
    patience_counter = 0
    best_path = out_dir / "unet_best_weights.pt"
    history = []

    n_train = sum(len(b[0]) for b in train_loader)
    n_val   = sum(len(b[0]) for b in val_loader)
    print(f"[classify] Training on {device} | loss: bce+dice | pos_weight: {pos_weight}")
    print(f"[classify] {len(train_loader)} train batches, {len(val_loader)} val batches")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, pos_weight, optimizer, training=True)
        val_loss   = _run_epoch(model, val_loader,   device, pos_weight, training=False)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6), "lr": lr_now})
        print(f"  Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} val={val_loss:.4f} lr={lr_now:.2e}")

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_loss:.4f})")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"  Loaded best weights (val_loss={best_loss:.4f})")

    return model, history


def _run_epoch(model, loader, device, pos_weight, optimizer=None, training=True):
    import torch

    model.train(training)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y, valid in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            pred = model(x)
            loss = _bce_dice_loss(pred, y, valid, pos_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def _get_device(unet_cfg):
    import torch
    req = str(unet_cfg.get("device", "auto")).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_unet_classify_workflow(cfg: dict[str, Any]) -> dict[str, Any]:
    """UNet binary classification: feature stack → patches → train → predict → evaluate."""

    out_dir  = ml_output_dir(cfg, "unet_classify")
    unet_cfg = cfg.get("ml", {}).get("unet", {})
    nodata   = int(cfg.get("ml", {}).get("target", {}).get("nodata", 255))

    # 1. Target + features
    target_data   = build_target(cfg)
    feature_stack = build_feature_stack(cfg, reference_shape=target_data.target.shape)

    numeric, categorical = configured_feature_names(cfg)
    feature_names = numeric + categorical
    features_2d   = stack_feature_arrays(feature_stack.arrays, feature_names)
    shape         = target_data.target.shape

    print(f"[classify] Target rule : {target_data.positive_rule}")
    n_pos = int((target_data.target == 1).sum())
    n_val = int((target_data.target != nodata).sum())
    print(f"[classify] Positives   : {n_pos:,} / {n_val:,} valid pixels  ({100*n_pos/max(n_val,1):.1f}%)")
    print(f"[classify] Features    : {feature_names}")

    # 2. Split
    data       = build_ml_dataset(cfg)
    train_rows = data.splits["train"]
    test_rows  = data.splits.get("test", np.array([], dtype=int))

    # 3. Normalise on train pixels only
    train_mask = np.zeros(shape, dtype=bool)
    train_mask.ravel()[data.flat_indices[train_rows]] = True
    norm_stats = compute_norm_stats(features_2d, train_mask)
    save_norm_stats(norm_stats, out_dir / "unet_norm_stats.json")
    features_norm = normalize_features(features_2d, norm_stats)

    # 4. Patch datasets
    patch_size = int(unet_cfg.get("patch_size", 128))
    stride     = max(1, int(patch_size * (1.0 - float(unet_cfg.get("overlap", 0.5)))))
    val_frac   = float(unet_cfg.get("val_fraction", 0.15))
    seed       = int(cfg.get("ml", {}).get("split", {}).get("seed", 42))

    full_ds = PatchDataset.from_data(
        features_norm, target_data.target, data.flat_indices[train_rows],
        shape, patch_size=patch_size, stride=stride, nodata=nodata, augment=True,
    )
    n_total = len(full_ds)
    n_val_p = max(1, int(n_total * val_frac))
    n_train_p = n_total - n_val_p
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)

    from torch.utils.data import DataLoader, Subset

    train_ds = Subset(full_ds, idx[:n_train_p].tolist())
    val_ds_base = PatchDataset.from_data(
        features_norm, target_data.target, data.flat_indices[train_rows],
        shape, patch_size=patch_size, stride=stride, nodata=nodata, augment=False,
    )
    val_ds = Subset(val_ds_base, idx[n_train_p:].tolist())

    num_workers = int(unet_cfg.get("num_workers", 0))
    batch_size  = int(unet_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 5. Build + train model
    in_channels = features_2d.shape[0]
    model_cfg   = cfg.get("ml", {}).get("model", {})
    model       = build_unet(model_cfg, in_channels=in_channels)
    pos_weight  = float(unet_cfg.get("pos_weight", 3.0))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[classify] UNet: {n_params:,} params, {in_channels} input channels")

    model, history = _train(model, train_loader, val_loader, cfg, out_dir, pos_weight)
    write_training_history(history, out_dir / "unet_training_history.csv")

    # 6. Save weights
    import torch
    weights_path = out_dir / "unet_weights.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": model_cfg,
        "in_channels": in_channels,
        "feature_names": feature_names,
        "norm_stats": norm_stats,
        "positive_rule": target_data.positive_rule,
    }, weights_path)

    # 7. Probability map
    prob_map  = predict_sliding_window(model, features_norm, cfg)
    prob_path = write_float_raster(
        out_dir / "unet_source_probability.tif",
        prob_map, target_data.profile, nodata=-9999.0,
    )

    # 8. Binary class map
    threshold = float(cfg.get("ml", {}).get("prediction", {}).get("probability_threshold", 0.5))
    class_map = np.full(shape, nodata, dtype="uint8")
    valid_prob = np.isfinite(prob_map)
    class_map[valid_prob] = (prob_map[valid_prob] >= threshold).astype("uint8")
    if bool(cfg.get("ml", {}).get("prediction", {}).get("exclude_channels", True)):
        class_map[target_data.channel_mask] = nodata
    thr_tag = f"{threshold:g}".replace(".", "p")   # 0.5 → "0p5", never touches .tif
    class_path = write_uint8_raster(
        out_dir / f"unet_source_class_p{thr_tag}.tif",
        class_map, target_data.profile, nodata=nodata,
    )

    # 9. Classification metrics
    metrics: dict[str, Any] = {}
    for split_name, rows in [("train", train_rows), ("test", test_rows)]:
        if len(rows) == 0:
            continue
        flat   = data.flat_indices[rows]
        y_true = data.y[rows].astype("uint8")
        probs  = extract_pixel_probs(prob_map, flat, shape)
        valid  = np.isfinite(probs)
        if valid.any():
            y_pred = (probs[valid] >= threshold).astype("uint8")
            metrics[split_name] = classification_metrics(y_true[valid], y_pred, probs[valid])
    metrics_path = write_json(out_dir / "unet_metrics.json", metrics)

    # 10. Export target
    target_path = write_uint8_raster(
        out_dir / "ml_target.tif",
        target_data.target, target_data.profile, nodata=nodata,
    )

    result = {
        "model_name":       "unet_classify",
        "output_dir":       out_dir,
        "positive_rule":    target_data.positive_rule,
        "in_channels":      in_channels,
        "n_train_patches":  n_train_p,
        "n_val_patches":    n_val_p,
        "target":           target_path,
        "probability":      prob_path,
        "class":            class_path,
        "weights":          weights_path,
        "metrics":          metrics_path,
        "training_history": out_dir / "unet_training_history.csv",
    }

    # Print summary
    print("\n" + "─" * 50)
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
    if metrics:
        for split, m in metrics.items():
            print(f"\n  [{split}] mcc={m.get('mcc', float('nan')):.3f}  "
                  f"f1={m.get('f1', float('nan')):.3f}  "
                  f"precision={m.get('precision', float('nan')):.3f}  "
                  f"recall={m.get('recall', float('nan')):.3f}")

    return result
