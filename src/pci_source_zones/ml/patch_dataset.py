from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def stack_feature_arrays(arrays: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Stack named 2D feature arrays into (C, H, W) float32 tensor."""
    return np.stack([arrays[name].astype("float32") for name in names], axis=0)


def compute_norm_stats(features: np.ndarray, train_mask: np.ndarray) -> dict[str, list[float]]:
    """Compute per-channel mean and std from train pixels only.

    features   : (C, H, W) float32
    train_mask : (H, W) bool
    Returns dict with 'mean' and 'std' lists of length C.
    """
    means, stds = [], []
    for c in range(features.shape[0]):
        vals = features[c][train_mask & np.isfinite(features[c])]
        means.append(float(vals.mean()) if vals.size > 0 else 0.0)
        stds.append(float(vals.std()) if vals.size > 0 and vals.std() > 0 else 1.0)
    return {"mean": means, "std": stds}


def normalize_features(features: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    """Apply per-channel z-score normalization."""
    out = features.copy()
    means = np.array(stats["mean"], dtype="float32")[:, None, None]
    stds = np.array(stats["std"], dtype="float32")[:, None, None]
    out = (out - means) / stds
    return out


def save_norm_stats(stats: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_norm_stats(path: Path) -> dict[str, list[float]]:
    return json.loads(path.read_text(encoding="utf-8"))


class PatchDataset:
    """PyTorch Dataset that yields (feature_patch, label_patch) pairs.

    Patches are extracted on a regular strided grid over the valid mask.
    Each sample: x = (C, patch_size, patch_size) float32
                 y = (patch_size, patch_size) float32  [0.0, 1.0, or nan for nodata]
    """

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray,
        patch_size: int = 128,
        stride: int | None = None,
        nodata: int = 255,
        augment: bool = False,
        min_valid_frac: float = 0.1,
    ) -> None:
        try:
            from torch.utils.data import Dataset  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install torch to use UNet: pip install torch") from exc

        self.features = np.nan_to_num(features.astype("float32"), nan=0.0)
        self.target = target
        self.valid_mask = valid_mask
        self.patch_size = patch_size
        self.stride = stride if stride is not None else max(1, patch_size // 2)
        self.nodata = nodata
        self.augment = augment
        self.min_valid_frac = min_valid_frac
        self.locations = self._sample_locations()

    def _sample_locations(self) -> list[tuple[int, int]]:
        H, W = self.valid_mask.shape
        ps = self.patch_size
        locations = []
        r = 0
        while r + ps <= H:
            c = 0
            while c + ps <= W:
                patch_valid = self.valid_mask[r : r + ps, c : c + ps]
                if patch_valid.mean() >= self.min_valid_frac:
                    locations.append((r, c))
                c += self.stride
            r += self.stride
        return locations

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, idx: int):
        import torch

        r, c = self.locations[idx]
        ps = self.patch_size

        x = self.features[:, r : r + ps, c : c + ps].copy()

        y_raw = self.target[r : r + ps, c : c + ps].copy().astype("float32")
        valid = y_raw != float(self.nodata)
        y = y_raw.copy()
        y[~valid] = 0.0

        if self.augment:
            x, y, valid = _augment(x, y, valid)

        return torch.tensor(x), torch.tensor(y), torch.tensor(valid)

    @classmethod
    def from_data(
        cls,
        features: np.ndarray,
        target: np.ndarray,
        pixel_rows: np.ndarray,
        shape: tuple[int, int],
        patch_size: int = 128,
        stride: int | None = None,
        nodata: int = 255,
        augment: bool = False,
    ) -> "PatchDataset":
        """Build dataset from flat pixel row indices (e.g. data.splits['train'])."""
        valid_mask = np.zeros(shape, dtype=bool)
        flat_indices = pixel_rows
        valid_mask.ravel()[flat_indices] = True

        # Pad features/target so patches fit exactly
        H, W = shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            features = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            target = np.pad(target, ((0, pad_h), (0, pad_w)), constant_values=nodata)
            valid_mask = np.pad(valid_mask, ((0, pad_h), (0, pad_w)), constant_values=False)

        return cls(features, target, valid_mask, patch_size, stride, nodata, augment)


def pad_like_patches(arr: np.ndarray, shape: tuple[int, int], patch_size: int, fill: int = 0) -> np.ndarray:
    """Pad a 2D array exactly as PatchDataset.from_data pads features/target.

    Patch locations are expressed in padded coordinates, so any array indexed
    by those locations (e.g. a polygon-group raster) must use the same padding.
    """
    H, W = shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=fill)


def assign_patch_groups(
    locations: list[tuple[int, int]],
    patch_size: int,
    group_raster: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each patch to the polygon groups it covers.

    group_raster holds 1..N inside the Nth polygon, 0 outside (already padded
    to match patch coordinates).

    Returns:
      dominant: (n_patches,) most-covered nonzero group per patch, 0 if none
      touches:  (n_patches, N+1) bool, whether the patch contains any pixel
                of group g — this is what makes leave-one-polygon-out honest:
                a patch is ~1.28 km wide and can straddle a polygon boundary,
                so "held out" has to mean "touches zero held-out pixels", not
                just "is mostly elsewhere".
    """
    n_groups = int(group_raster.max())
    dominant = np.zeros(len(locations), dtype=int)
    touches = np.zeros((len(locations), n_groups + 1), dtype=bool)

    for i, (r, c) in enumerate(locations):
        block = group_raster[r : r + patch_size, c : c + patch_size]
        counts = np.bincount(block.ravel(), minlength=n_groups + 1)
        touches[i] = counts > 0
        counts[0] = 0  # "outside any train polygon" is not a group
        if counts.sum() > 0:
            dominant[i] = int(counts.argmax())

    return dominant, touches


def leave_one_polygon_out_indices(
    dominant: np.ndarray,
    touches: np.ndarray,
    held_out_group: int,
) -> tuple[list[int], list[int]]:
    """Split patch indices into (train, val) for one leave-one-polygon-out fold.

    Val patches are those centred on the held-out polygon; train patches are
    those touching *none* of it. Patches that merely straddle the boundary are
    dropped from both — they would otherwise leak val pixels into training.
    """
    val_idx = [i for i in range(len(dominant)) if dominant[i] == held_out_group]
    train_idx = [i for i in range(len(dominant)) if not touches[i, held_out_group]]
    return train_idx, val_idx


def _augment(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.random.random() > 0.5:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=1).copy()
        valid = np.flip(valid, axis=1).copy()
    if np.random.random() > 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=0).copy()
        valid = np.flip(valid, axis=0).copy()
    return x, y, valid


def _augment_tobit(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    cen: np.ndarray,
    bound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.random.random() > 0.5:
        x     = np.flip(x,     axis=2).copy()
        y     = np.flip(y,     axis=1).copy()
        valid = np.flip(valid, axis=1).copy()
        cen   = np.flip(cen,   axis=1).copy()
        bound = np.flip(bound, axis=1).copy()
    if np.random.random() > 0.5:
        x     = np.flip(x,     axis=1).copy()
        y     = np.flip(y,     axis=0).copy()
        valid = np.flip(valid, axis=0).copy()
        cen   = np.flip(cen,   axis=0).copy()
        bound = np.flip(bound, axis=0).copy()
    return x, y, valid, cen, bound


class TobitPatchDataset(PatchDataset):
    """PatchDataset extended for Tobit regression.

    Yields (x, y, valid_obs, censored, logC_bound) per patch where:
      valid_obs : bool mask — True where p_obs > 0 (C_eff observed)
      censored  : float32  — 1 where p_obs = 0 AND hillslope
      logC_bound: float32  — lower bound on logC_eff for censored cells
    """

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray,
        censored: np.ndarray,
        logC_bound: np.ndarray,
        patch_size: int = 128,
        stride: int | None = None,
        nodata: float = -9999.0,
        augment: bool = False,
        min_valid_frac: float = 0.05,
    ) -> None:
        # Store aux arrays before super().__init__ (which calls _sample_locations)
        self._censored_full   = np.nan_to_num(censored.astype("float32"),   nan=0.0)
        self._logC_bound_full = np.nan_to_num(logC_bound.astype("float32"), nan=0.0)
        super().__init__(features, target, valid_mask, patch_size, stride,
                         int(nodata), augment, min_valid_frac)

    def __getitem__(self, idx: int):
        import torch

        r, c = self.locations[idx]
        ps = self.patch_size

        x     = self.features[:, r:r+ps, c:c+ps].copy()
        y_raw = self.target[r:r+ps, c:c+ps].copy().astype("float32")
        valid = y_raw != float(self.nodata)
        y     = y_raw.copy()
        y[~valid] = 0.0

        cen   = self._censored_full[r:r+ps, c:c+ps].copy()
        bound = self._logC_bound_full[r:r+ps, c:c+ps].copy()

        if self.augment:
            x, y, valid, cen, bound = _augment_tobit(x, y, valid, cen, bound)

        return (
            torch.tensor(x),
            torch.tensor(y),
            torch.tensor(valid),
            torch.tensor(cen),
            torch.tensor(bound),
        )

    @classmethod
    def from_data_tobit(
        cls,
        features: np.ndarray,
        target: np.ndarray,
        censored: np.ndarray,
        logC_bound: np.ndarray,
        pixel_rows: np.ndarray,
        shape: tuple[int, int],
        patch_size: int = 128,
        stride: int | None = None,
        nodata: float = -9999.0,
        augment: bool = False,
    ) -> "TobitPatchDataset":
        valid_mask = np.zeros(shape, dtype=bool)
        valid_mask.ravel()[pixel_rows] = True

        H, W = shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            features   = np.pad(features,   ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            target     = np.pad(target,     ((0, pad_h), (0, pad_w)), constant_values=nodata)
            valid_mask = np.pad(valid_mask, ((0, pad_h), (0, pad_w)), constant_values=False)
            censored   = np.pad(censored,   ((0, pad_h), (0, pad_w)), constant_values=0.0)
            logC_bound = np.pad(logC_bound, ((0, pad_h), (0, pad_w)), constant_values=0.0)

        return cls(features, target, valid_mask, censored, logC_bound,
                   patch_size, stride, nodata, augment)
