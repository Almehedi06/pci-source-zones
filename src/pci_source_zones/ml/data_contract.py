"""Executable version of docs/data_contract.md.

Resolves every raster a *given* ML config will actually load (not a fixed
universal list — feature sets differ per config) and checks, before any
training starts, that each one exists and shares the same CRS/resolution/
shape/pixel grid as the reference raster. Generalizes the warn-only
`features._check_transform_alignment` into a hard-failing preflight pass
that reports every problem at once, instead of failing one raster at a time
deep inside training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rasterio

from pci_source_zones.config import resolve_data_path
from .features import configured_feature_names


@dataclass
class RasterFingerprint:
    name: str
    path: str
    size_bytes: int
    mtime: float


@dataclass
class ContractReport:
    reference: str
    rasters: list[RasterFingerprint] = field(default_factory=list)


def _referenced_raster_paths(cfg: dict[str, Any]) -> dict[str, str]:
    """Every raster path a run of this config would load, keyed by a label."""
    ml_cfg = cfg.get("ml", {})
    paths: dict[str, str] = {}

    for name, p in ml_cfg.get("base_rasters", {}).items():
        paths[f"base_rasters.{name}"] = p

    # Only the features actually selected in ml.features.numeric/categorical
    # get loaded at runtime (features.py::_load_file_features) — feature_paths
    # commonly holds extra entries kept around for reference/leakage-avoidance
    # (commented out of the feature list), which must not be validated.
    numeric, categorical = configured_feature_names(cfg)
    selected = set(numeric) | set(categorical)
    for name, p in ml_cfg.get("feature_paths", {}).items():
        if name in selected:
            paths[f"feature_paths.{name}"] = p

    target_cfg = ml_cfg.get("target", {})
    if target_cfg.get("path"):
        paths["target.path"] = target_cfg["path"]

    unet_cfg = ml_cfg.get("unet", {}) or {}
    tobit_cfg = unet_cfg.get("tobit") or {}
    if tobit_cfg.get("censored_path"):
        paths["unet.tobit.censored_path"] = tobit_cfg["censored_path"]
    if tobit_cfg.get("logC_bound_path"):
        paths["unet.tobit.logC_bound_path"] = tobit_cfg["logC_bound_path"]

    split_cfg = ml_cfg.get("split", {})
    if split_cfg.get("method") == "raster":
        raster_split = split_cfg.get("raster", {})
        if raster_split.get("path"):
            paths["split.raster.path"] = raster_split["path"]
    if split_cfg.get("method") == "aoi_rasters":
        for name in ("train", "val", "test"):
            if split_cfg.get(name):
                paths[f"split.{name}"] = split_cfg[name]

    return paths


def validate_data_contract(cfg: dict[str, Any], tolerance: float = 1e-3) -> ContractReport:
    """Validate every raster this config references shares one common grid.

    Raises ValueError listing *every* problem found (missing file, CRS
    mismatch, resolution mismatch, shape mismatch, or same-shape-different-
    transform) in a single consolidated message. Returns a ContractReport
    (file fingerprints) on success, reused by provenance.write_run_manifest
    so files aren't stat'd twice.
    """
    referenced = _referenced_raster_paths(cfg)
    if not referenced:
        raise ValueError("No rasters referenced by ml.base_rasters/feature_paths — nothing to validate.")

    problems: list[str] = []
    fingerprints: list[RasterFingerprint] = []
    reference_meta: dict[str, Any] | None = None
    reference_label = ""

    for label, raw_path in sorted(referenced.items()):
        resolved = resolve_data_path(cfg, raw_path)
        if not resolved.exists():
            problems.append(f"{label}: file not found: {resolved}")
            continue

        try:
            with rasterio.open(resolved) as src:
                meta = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "shape": (src.height, src.width),
                    "res": src.res,
                }
        except Exception as exc:  # rasterio's own open-time errors
            problems.append(f"{label}: could not open {resolved}: {exc}")
            continue

        stat = resolved.stat()
        fingerprints.append(
            RasterFingerprint(name=label, path=str(resolved), size_bytes=stat.st_size, mtime=stat.st_mtime)
        )

        if reference_meta is None:
            reference_meta = meta
            reference_label = label
            continue

        if meta["crs"] != reference_meta["crs"]:
            problems.append(
                f"{label}: CRS {meta['crs']} does not match reference "
                f"({reference_label}: {reference_meta['crs']})"
            )
        if meta["shape"] != reference_meta["shape"]:
            problems.append(
                f"{label}: shape {meta['shape']} does not match reference "
                f"({reference_label}: {reference_meta['shape']})"
            )
        elif not _transform_matches(meta["transform"], reference_meta["transform"], tolerance):
            problems.append(
                f"{label}: same shape as reference ({reference_label}) but a different pixel "
                f"grid (transform mismatch) — pixels would be spatially misaligned"
            )

    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(
            f"Data contract check failed for {len(problems)} raster(s) "
            f"(reference grid: {reference_label}):\n  - {joined}\n\n"
            f"Re-run scripts/10_align_features.py, or fix the offending path(s) in the config."
        )

    return ContractReport(reference=reference_label, rasters=fingerprints)


def _transform_matches(t: Any, r: Any, tolerance: float) -> bool:
    ref_vals = [r.a, r.b, r.c, r.d, r.e, r.f]
    feat_vals = [t.a, t.b, t.c, t.d, t.e, t.f]
    return all(abs(rv - fv) < tolerance for rv, fv in zip(ref_vals, feat_vals))
