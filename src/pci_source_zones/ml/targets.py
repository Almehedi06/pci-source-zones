from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pci_source_zones.config import output_path, resolve_data_path, resolve_path
from pci_source_zones.inputs import read_raster


@dataclass
class TargetData:
    target: np.ndarray
    valid_mask: np.ndarray
    channel_mask: np.ndarray
    profile: dict[str, Any]
    positive_rule: str


def build_target(cfg: dict[str, Any]) -> TargetData:
    """Build a binary or multiclass ML target from config."""

    target_cfg = cfg.get("ml", {}).get("target", {})
    target_type = str(target_cfg.get("type", "physics_dod")).lower()
    nodata = int(target_cfg.get("nodata", 255))

    if target_type == "raster":
        path = resolve_data_path(cfg, target_cfg["path"])
        arr, profile = read_raster(path)
        positive = int(target_cfg.get("positive_value", 1))
        valid = np.isfinite(arr)
        target = np.full(arr.shape, nodata, dtype="uint8")
        target[valid] = (arr[valid] == positive).astype("uint8")
        channel = np.zeros(arr.shape, dtype=bool)
        return TargetData(target, valid, channel, profile, f"raster == {positive}")

    if target_type == "multiclass_raster":
        return _build_multiclass_target(cfg, target_cfg, nodata)

    if target_type == "raster_continuous":
        return _build_continuous_target(cfg, target_cfg)

    g, profile = _read_g(cfg)
    dem_diff, _ = _read_dem_diff(cfg)
    if dem_diff.shape != g.shape:
        raise ValueError(f"DEM-difference shape {dem_diff.shape} does not match G shape {g.shape}.")

    erosion_threshold = float(target_cfg.get("erosion_threshold_m", -0.55))
    c_min = target_cfg.get("c_min", target_cfg.get("g_min", None))
    c_channel = target_cfg.get("c_channel", target_cfg.get("g_channel", 120))
    c_channel = None if c_channel is None else float(c_channel)

    valid = np.isfinite(g) & np.isfinite(dem_diff)
    erosion = dem_diff < erosion_threshold
    channel = np.zeros(g.shape, dtype=bool)
    if c_channel is not None:
        channel = valid & (g >= c_channel)

    if target_type in {"dod_only", "dem_diff", "ddem"}:
        source = valid & erosion
        rule = f"DoD < {erosion_threshold:g}"
    elif target_type in {"physics_dod", "c_band_dod", "g_band_dod"}:
        if c_min is None:
            raise ValueError("ml.target.c_min is required for physics_dod targets.")
        source = valid & erosion & (g >= float(c_min))
        if c_channel is not None:
            source &= g < c_channel
        rule = f"{float(c_min):g} <= G < {c_channel:g} and DoD < {erosion_threshold:g}"
    else:
        raise ValueError(f"Unsupported ml.target.type: {target_type!r}")

    if bool(target_cfg.get("exclude_channels_from_training", True)):
        valid = valid & ~channel

    target = np.full(g.shape, nodata, dtype="uint8")
    target[valid] = 0
    target[source & valid] = 1
    return TargetData(target, valid, channel, profile, rule)


def _read_g(cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    ml_cfg = cfg.get("ml", {})
    base = ml_cfg.get("base_rasters", {})
    if "g" in base:
        return read_raster(resolve_data_path(cfg, base["g"]))
    return read_raster(output_path(cfg, "topographic_driving_index", "topographic_driving_index.tif"))


def _build_continuous_target(
    cfg: dict[str, Any],
    target_cfg: dict[str, Any],
) -> TargetData:
    """Load a continuous float raster as regression target (e.g. src_hi 0–1).

    Nodata sentinel: -9999.0 (compatible with PatchDataset).
    """
    if "path" not in target_cfg:
        raise ValueError("raster_continuous target requires ml.target.path.")
    path = resolve_data_path(cfg, target_cfg["path"])
    arr, profile = read_raster(path)
    valid = np.isfinite(arr)
    target = np.full(arr.shape, -9999.0, dtype="float32")
    target[valid] = arr[valid].astype("float32")
    channel = np.zeros(arr.shape, dtype=bool)
    return TargetData(target, valid, channel, profile, "continuous raster (regression)")


def _build_multiclass_target(
    cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    nodata: int,
) -> TargetData:
    """Load a multiclass target raster with an optional integer class map.

    Config example::

        ml:
          target:
            type: multiclass_raster
            path: /path/to/class_labels.tif
            class_map:            # optional: remap raw pixel values to class ids
              0: 0                # background / no-erosion
              1: 1                # rill initiation
              2: 2                # channel scour
              3: 3                # deposition
            nodata: 255
    """
    if "path" not in target_cfg:
        raise ValueError("multiclass_raster target requires ml.target.path.")

    path = resolve_data_path(cfg, target_cfg["path"])
    arr, profile = read_raster(path)
    class_map: dict[int, int] | None = target_cfg.get("class_map", None)

    valid = np.isfinite(arr) & (arr != nodata)
    target = np.full(arr.shape, nodata, dtype="uint8")

    if class_map:
        for src_val, dst_val in class_map.items():
            mask = valid & (arr == int(src_val))
            target[mask] = int(dst_val)
    else:
        target[valid] = arr[valid].astype("uint8")

    classes = sorted(set(int(v) for v in np.unique(target) if int(v) != nodata))
    rule = f"multiclass ({classes})"
    channel = np.zeros(arr.shape, dtype=bool)
    return TargetData(target, valid, channel, profile, rule)


def _read_dem_diff(cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    ml_cfg = cfg.get("ml", {})
    base = ml_cfg.get("base_rasters", {})
    if "dem_diff" in base:
        return read_raster(resolve_data_path(cfg, base["dem_diff"]))
    if "dem_diff" in cfg.get("paths", {}):
        return read_raster(resolve_data_path(cfg, cfg["paths"]["dem_diff"]))
    raise ValueError("Set ml.base_rasters.dem_diff or paths.dem_diff in the config.")

