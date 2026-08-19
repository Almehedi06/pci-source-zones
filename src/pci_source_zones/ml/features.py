from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pci_source_zones.config import output_path, resolve_data_path, resolve_path
from pci_source_zones.inputs import read_raster


@dataclass
class FeatureStack:
    arrays: dict[str, np.ndarray]
    frame: pd.DataFrame
    valid_mask: np.ndarray
    flat_indices: np.ndarray
    profile: dict[str, Any]


def configured_feature_names(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    feature_cfg = cfg.get("ml", {}).get("features", {})
    numeric = list(feature_cfg.get("numeric", []))
    categorical = list(feature_cfg.get("categorical", []))
    exclude = set(feature_cfg.get("exclude", []))
    numeric = [n for n in numeric if n not in exclude]
    categorical = [n for n in categorical if n not in exclude]
    return numeric, categorical


def build_feature_stack(
    cfg: dict[str, Any],
    reference_shape: tuple[int, int] | None = None,
) -> FeatureStack:
    """Load configured feature rasters and return a flat DataFrame."""

    arrays, profile = _load_base_arrays(cfg)
    numeric, categorical = configured_feature_names(cfg)
    names = numeric + categorical
    if not names:
        raise ValueError("Set ml.features.numeric and/or ml.features.categorical.")

    ref_shape = reference_shape or _first_shape(arrays)
    arrays.update(_load_file_features(cfg, ref_shape, names, profile))
    arrays = _add_derived_features(arrays, profile)

    missing = [name for name in names if name not in arrays]
    if missing:
        raise ValueError(f"Configured ML features are missing: {missing}")

    selected = {name: np.asarray(arrays[name], dtype="float64") for name in names}
    shape = reference_shape or _first_shape(selected)

    valid = np.ones(shape, dtype=bool)
    for name, arr in selected.items():
        if arr.shape != shape:
            raise ValueError(f"Feature {name!r} shape {arr.shape} does not match {shape}.")
        valid &= np.isfinite(arr)

    flat_indices = np.flatnonzero(valid.ravel())
    frame = pd.DataFrame({name: arr.ravel()[flat_indices] for name, arr in selected.items()})
    return FeatureStack(selected, frame, valid, flat_indices, profile)


def _load_base_arrays(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base = cfg.get("ml", {}).get("base_rasters", {})
    arrays: dict[str, np.ndarray] = {}
    profile: dict[str, Any] | None = None

    paths = {
        "slope": base.get("slope", output_path(cfg, "slope", "slope.tif")),
        "specific_area": base.get(
            "specific_area",
            base.get(
                "a",
                output_path(cfg, "specific_catchment_area", "specific_catchment_area.tif"),
            ),
        ),
        "G": base.get("g", output_path(cfg, "topographic_driving_index", "topographic_driving_index.tif")),
    }

    for name, path in paths.items():
        try:
            arr, prof = read_raster(resolve_data_path(cfg, path))
        except Exception:
            continue
        arrays[name] = arr
        if profile is None:
            profile = prof

    if profile is None:
        raise ValueError("No base ML rasters found. Set ml.base_rasters in the config.")
    return arrays, profile


def _load_file_features(
    cfg: dict[str, Any],
    shape: tuple[int, int],
    requested_names: list[str],
    reference_profile: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    paths = cfg.get("ml", {}).get("feature_paths", {})
    out: dict[str, np.ndarray] = {}
    for name in requested_names:
        if name not in paths:
            continue
        path = resolve_data_path(cfg, paths[name])
        arr, prof = read_raster(path)
        if arr.shape != shape:
            raise ValueError(f"Feature {name!r} shape {arr.shape} does not match {shape}: {path}")
        if reference_profile is not None:
            _check_transform_alignment(name, prof, reference_profile)
        out[name] = arr
    return out


def _check_transform_alignment(
    name: str,
    profile: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    """Raise if a feature raster has the same shape but a misaligned pixel grid.

    Same shape with a different transform silently produces spatially wrong
    training data — pixel [i, j] would mean a different real-world location
    in this raster than in every other feature. That must fail loudly: a
    warning is easy to miss in a long training log, and training would
    otherwise proceed on misaligned data without any error.
    """
    t = profile.get("transform")
    r = reference.get("transform")
    if t is None or r is None:
        return
    # Compare origin and pixel size (affine coefficients 0–5)
    ref_vals = [r.a, r.b, r.c, r.d, r.e, r.f]
    feat_vals = [t.a, t.b, t.c, t.d, t.e, t.f]
    if not all(abs(rv - fv) < 1e-3 for rv, fv in zip(ref_vals, feat_vals)):
        raise ValueError(
            f"Feature {name!r} has the same shape as the reference raster but a different "
            f"pixel grid (transform mismatch: {feat_vals} vs reference {ref_vals}). "
            f"Pixels would be spatially misaligned. Re-snap the raster to the reference grid "
            f"(scripts/10_align_features.py) before training."
        )


def _add_derived_features(arrays: dict[str, np.ndarray], profile: dict[str, Any]) -> dict[str, np.ndarray]:
    out = dict(arrays)
    if "specific_area" in out:
        cell_size = abs(float(profile["transform"].a))
        out["drainage_area"] = out["specific_area"] * cell_size
        out["log_drainage_area"] = _safe_log10(out["drainage_area"])
        out["log_DA"] = out["log_drainage_area"]
    if "G" in out:
        out["log_G"] = _safe_log10(out["G"])
        out["log_g"] = out["log_G"]
    return out


def _safe_log10(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype="float64")
    mask = np.isfinite(arr) & (arr > 0)
    out[mask] = np.log10(arr[mask])
    return out


def _first_shape(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    for arr in arrays.values():
        return arr.shape
    raise ValueError("No arrays available.")
