from __future__ import annotations

from typing import Any

import numpy as np


def predict_sliding_window(
    model: Any,
    features: np.ndarray,
    cfg: dict[str, Any],
    batch_size: int = 8,
) -> np.ndarray:
    """Sliding window inference over the full raster.

    features : (C, H, W) float32, already normalized
    Returns  : (H, W) float32 probability map, NaN where no patch covered
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install torch to use UNet: pip install torch") from exc

    unet_cfg = cfg.get("ml", {}).get("unet", {})
    patch_size = int(unet_cfg.get("patch_size", 128))
    overlap = float(unet_cfg.get("overlap", 0.5))
    device = _get_device(unet_cfg)
    stride = max(1, int(patch_size * (1.0 - overlap)))

    features = np.nan_to_num(features.astype("float32"), nan=0.0)
    C, H, W = features.shape

    # Pad so the last stride still produces a full patch
    pad_h = _pad_needed(H, patch_size, stride)
    pad_w = _pad_needed(W, patch_size, stride)
    feat_pad = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    prob_sum = np.zeros((H + pad_h, W + pad_w), dtype="float64")
    count = np.zeros((H + pad_h, W + pad_w), dtype="float64")

    # Collect all patch top-left corners
    corners: list[tuple[int, int]] = []
    r = 0
    while r + patch_size <= feat_pad.shape[1]:
        c = 0
        while c + patch_size <= feat_pad.shape[2]:
            corners.append((r, c))
            c += stride
        r += stride

    model.eval()
    model.to(device)

    # Process in batches for GPU efficiency
    with torch.no_grad():
        for i in range(0, len(corners), batch_size):
            batch_corners = corners[i : i + batch_size]
            patches = np.stack(
                [feat_pad[:, r : r + patch_size, c : c + patch_size] for r, c in batch_corners],
                axis=0,
            )
            x = torch.tensor(patches).to(device)
            preds = model(x).squeeze(1).cpu().numpy()  # (B, ps, ps)

            for pred, (r, c) in zip(preds, batch_corners):
                prob_sum[r : r + patch_size, c : c + patch_size] += pred
                count[r : r + patch_size, c : c + patch_size] += 1.0

    # Crop back to original size and average
    prob_sum = prob_sum[:H, :W]
    count = count[:H, :W]

    prob = np.full((H, W), np.nan, dtype="float32")
    covered = count > 0
    prob[covered] = (prob_sum[covered] / count[covered]).astype("float32")
    return prob


def extract_pixel_probs(
    prob_map: np.ndarray,
    flat_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Pull per-pixel probabilities for a set of flat indices (for metric computation)."""
    return prob_map.ravel()[flat_indices].astype("float32")


def _pad_needed(size: int, patch_size: int, stride: int) -> int:
    if size <= patch_size:
        return patch_size - size
    remainder = (size - patch_size) % stride
    return 0 if remainder == 0 else stride - remainder


def _get_device(unet_cfg: dict[str, Any]):
    import torch

    requested = str(unet_cfg.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
