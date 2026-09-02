from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def bce_dice_loss(pred, target, valid_mask, pos_weight: float = 3.0):
    """Combined BCE + Dice loss, ignoring nodata pixels via valid_mask.

    pred       : (B, 1, H, W) sigmoid output
    target     : (B, H, W) float32  {0.0, 1.0}
    valid_mask : (B, H, W) bool
    """
    import torch
    import torch.nn.functional as F

    p = pred.squeeze(1)  # (B, H, W)
    eps = 1e-6

    # Weighted BCE — upweight positives
    weight = torch.where(target == 1.0, torch.tensor(pos_weight, device=p.device), torch.ones_like(p))
    bce_all = F.binary_cross_entropy(p, target, weight=weight, reduction="none")
    bce = (bce_all * valid_mask.float()).sum() / (valid_mask.float().sum() + eps)

    # Dice over valid pixels
    p_v = p[valid_mask]
    t_v = target[valid_mask]
    intersection = (p_v * t_v).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (p_v.sum() + t_v.sum() + eps)

    return 0.5 * bce + 0.5 * dice

def mse_loss(pred, target, valid_mask):
    import torch
    p = pred.squeeze(1)
    p_valid = p[valid_mask]
    t_valid = target[valid_mask]
    return ((p_valid - t_valid) ** 2).mean()


def tobit_loss(pred, target, valid_obs, censored, logC_bound, sigma_frozen: bool = False, censored_weight: float = 1.0):
    """Heteroscedastic Tobit loss for left-censored regression.

    pred      : (B, 2, H, W) — channel 0 = mu, channel 1 = log_sigma
    target    : (B, H, W)    — logC_eff; zeroed where nodata (use valid_obs mask)
    valid_obs : (B, H, W) bool — True where p_obs > 0 (C_eff observed)
    censored  : (B, H, W) float32 — 1 where p_obs = 0 AND hillslope (censored)
    logC_bound: (B, H, W) float32 — lower bound for censored cells (0 elsewhere)
    sigma_frozen: if True, treat sigma = 1 (warm-start; sigma head gets no gradient)
    """
    import torch

    mu      = pred[:, 0]        # (B, H, W)
    log_sig = pred[:, 1]        # (B, H, W)

    if sigma_frozen:
        sigma = torch.ones_like(mu)
    else:
        sigma = torch.exp(log_sig.clamp(-3, 2))

    eps = 1e-8

    # Observed NLL — Gaussian negative log-likelihood
    z_obs = (target - mu) / (sigma + eps)
    nll_obs = 0.5 * z_obs ** 2 + torch.log(sigma + eps)
    n_obs = valid_obs.float().sum() + eps
    loss_obs = (nll_obs * valid_obs.float()).sum() / n_obs

    # Censored NLL — survival function: -log P(C > C_bound | mu, sigma)
    # = -log(1 - Phi(z_cen)) = -log(Phi(-z_cen))
    cen_mask = (censored > 0.5) & (logC_bound > 0.0)
    loss_cen = torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    if cen_mask.any():
        z_cen = (logC_bound - mu) / (sigma + eps)
        try:
            log_surv = torch.special.log_ndtr(-z_cen)   # log(Phi(-z_cen)), numerically stable
        except AttributeError:
            import math
            log_surv = torch.log(0.5 * torch.erfc(z_cen / math.sqrt(2)) + eps)
        nll_cen = -log_surv
        n_cen = cen_mask.float().sum() + eps
        loss_cen = (nll_cen * cen_mask.float()).sum() / n_cen

    return loss_obs + censored_weight * loss_cen


def train_unet(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    cfg: dict[str, Any],
    out_dir: Path,
    is_regression: bool = False,
    tobit_mode: bool = False,
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
    warm_start_epochs = int(unet_cfg.get("warm_start_epochs", 5)) if tobit_mode else 0
    fixed_sigma = bool(unet_cfg.get("fixed_sigma", False))
    censored_weight = float(unet_cfg.get("censored_weight", 1.0))
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

    loss_name = "tobit" if tobit_mode else ("mse" if is_regression else "bce+dice")
    print(f"Training UNet on {device} | {len(train_dataset)} train patches, {len(val_dataset)} val patches")
    print(f"  loss: {loss_name}" + (f"  warm_start: {warm_start_epochs} epochs" if tobit_mode else ""))

    for epoch in range(1, epochs + 1):
        sigma_frozen = fixed_sigma or (tobit_mode and epoch <= warm_start_epochs)
        train_loss = _run_epoch(model, train_loader, device, pos_weight, optimizer, training=True,
                                is_regression=is_regression, tobit_mode=tobit_mode,
                                sigma_frozen=sigma_frozen, censored_weight=censored_weight)
        val_loss = _run_epoch(model, val_loader, device, pos_weight, training=False,
                              is_regression=is_regression, tobit_mode=tobit_mode,
                              sigma_frozen=sigma_frozen, censored_weight=censored_weight)
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
    is_regression: bool = False,
    tobit_mode: bool = False,
    sigma_frozen: bool = False,
    censored_weight: float = 1.0,
) -> float:
    import torch

    model.train(training)
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            if tobit_mode:
                x, y, valid, censored, logC_bound = batch
                x          = x.to(device)
                y          = y.to(device)
                valid      = valid.to(device)
                censored   = censored.to(device)
                logC_bound = logC_bound.to(device)
                pred = model(x)
                loss = tobit_loss(pred, y, valid, censored, logC_bound,
                                  sigma_frozen=sigma_frozen, censored_weight=censored_weight)
            else:
                x, y, valid = batch
                x     = x.to(device)
                y     = y.to(device)
                valid = valid.to(device)
                pred = model(x)
                if is_regression:
                    loss = mse_loss(pred, y, valid)
                else:
                    loss = bce_dice_loss(pred, y, valid, pos_weight)

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
