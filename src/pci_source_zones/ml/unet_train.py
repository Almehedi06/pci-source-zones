from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


# def bce_dice_loss(pred, target, valid_mask, pos_weight: float = 3.0):
#     """Combined BCE + Dice loss, ignoring nodata pixels via valid_mask.

#     pred       : (B, 1, H, W) sigmoid output
#     target     : (B, H, W) float32  {0.0, 1.0}
#     valid_mask : (B, H, W) bool
#     """
#     import torch
#     import torch.nn.functional as F

#     p = pred.squeeze(1)  # (B, H, W)
#     eps = 1e-6

#     # Weighted BCE — upweight positives
#     weight = torch.where(target == 1.0, torch.tensor(pos_weight, device=p.device), torch.ones_like(p))
#     bce_all = F.binary_cross_entropy(p, target, weight=weight, reduction="none")
#     bce = (bce_all * valid_mask.float()).sum() / (valid_mask.float().sum() + eps)

#     # Dice over valid pixels
#     p_v = p[valid_mask]
#     t_v = target[valid_mask]
#     intersection = (p_v * t_v).sum()
#     dice = 1.0 - (2.0 * intersection + eps) / (p_v.sum() + t_v.sum() + eps)

#     return 0.5 * bce + 0.5 * dice

def weighted_mse_loss(pred, target, valid_mask, high_weight: float = 10.0, threshold: float = 0.05):
    import torch
    p = pred.squeeze(1)
    p_valid = p[valid_mask]
    t_valid = target[valid_mask]
    weight = torch.where(t_valid > threshold,
                         torch.tensor(high_weight, device=p.device, dtype=torch.float32),
                         torch.ones_like(t_valid))
    return (weight * (p_valid - t_valid) ** 2).mean()


def train_unet(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    cfg: dict[str, Any],
    out_dir: Path,
) -> tuple[Any, list[dict[str, Any]]]:
    """Train UNet with BCE+Dice loss and early stopping.

    Returns the best model (by val loss) and training history.
    """
    try:
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Install torch to use UNet: pip install torch") from exc

    unet_cfg = cfg.get("ml", {}).get("unet", {})
    epochs = int(unet_cfg.get("epochs", 50))
    lr = float(unet_cfg.get("learning_rate", 0.001))
    batch_size = int(unet_cfg.get("batch_size", 16))
    patience = int(unet_cfg.get("early_stopping_patience", 10))
    pos_weight = float(unet_cfg.get("pos_weight", 3.0))
    num_workers = int(unet_cfg.get("num_workers", 0))
    device = _get_device(unet_cfg)

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = out_dir / "unet_best_weights.pt"
    history: list[dict[str, Any]] = []

    print(f"Training UNet on {device} | {len(train_dataset)} train patches, {len(val_dataset)} val patches")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, pos_weight, optimizer, training=True)
        val_loss = _run_epoch(model, val_loader, device, pos_weight, training=False)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr_now,
        }
        history.append(row)
        print(f"  Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} val={val_loss:.4f} lr={lr_now:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_val_loss:.4f})")
                break

    # Restore best weights
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"  Loaded best weights (val_loss={best_val_loss:.4f})")

    return model, history


def _run_epoch(
    model: Any,
    loader: Any,
    device: Any,
    pos_weight: float,
    optimizer: Any = None,
    training: bool = True,
) -> float:
    import torch

    model.train(training)
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y, valid in loader:
            x = x.to(device)
            y = y.to(device)
            valid = valid.to(device)

            pred = model(x)
            # bce_dice_loss(pred, y, valid, pos_weight)  # classification
            loss = weighted_mse_loss(pred, y, valid)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def write_training_history(history: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return path


def _get_device(unet_cfg: dict[str, Any]):
    import torch

    requested = str(unet_cfg.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
