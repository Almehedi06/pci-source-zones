from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from pci_source_zones.config import resolve_path


def ml_output_dir(cfg: dict[str, Any], model_name: str, run_id: str | None = None) -> Path:
    """Resolve (and create) the output directory for a training run.

    With run_id, each run gets its own immutable subdirectory
    (.../ml/{model_name}/{run_id}/) so a rerun can never silently overwrite
    a previous run's model/metrics — every current workflow function passes
    one. run_id=None keeps the old flat (.../ml/{model_name}/) behavior for
    any other/future direct caller.
    """
    ml_cfg = cfg.get("ml", {})
    subdir = ml_cfg.get("output_subdir", f"source_area_workflow/ml/{model_name}")
    subdir = str(subdir).format(model=model_name)
    out_root = resolve_path(cfg, cfg.get("paths", {}).get("output_dir", "data/outputs"))
    out_dir = out_root / subdir
    if run_id is not None:
        out_dir = out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ml_cache_dir(cfg: dict[str, Any], model_name: str) -> Path:
    """Stable (non-timestamped) cache directory, independent of run_id.

    Used for the tuning dataset cache: it must survive across separate
    tuning invocations to be useful, so it cannot live inside a per-run
    ml_output_dir() — every run would get a fresh, empty cache otherwise.
    """
    out_root = resolve_path(cfg, cfg.get("paths", {}).get("output_dir", "data/outputs"))
    cache_dir = out_root / ".cache" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def write_float_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
    nodata: float = -9999.0,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype="float32")
    meta = _clean_profile(profile)
    meta.update(driver="GTiff", count=1, dtype="float32", nodata=nodata, compress="deflate")
    write_path = _available_path(out)
    with rasterio.open(write_path, "w", **meta) as dst:
        dst.write(np.where(np.isfinite(data), data, nodata).astype("float32"), 1)
    return write_path


def write_uint8_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
    nodata: int = 255,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = _clean_profile(profile)
    meta.update(driver="GTiff", count=1, dtype="uint8", nodata=nodata, compress="deflate")
    write_path = _available_path(out)
    with rasterio.open(write_path, "w", **meta) as dst:
        dst.write(np.asarray(array, dtype="uint8"), 1)
    return write_path


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _clean_profile(profile: dict[str, Any]) -> dict[str, Any]:
    meta = profile.copy()
    if not meta.get("tiled", False):
        meta.pop("blockxsize", None)
        meta.pop("blockysize", None)
    return meta


def _available_path(path: Path) -> Path:
    """Return path, or a suffixed path if the original is locked by QGIS."""
    if not path.exists():
        return path
    try:
        path.unlink()
        return path
    except PermissionError:
        pass

    stem = path.stem
    for i in range(1, 100):
        candidate = path.with_name(f"{stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise PermissionError(f"Could not find available output path near {path}")
