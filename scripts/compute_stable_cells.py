"""Compute a stable-cell mask from one or more change layers (dDEM, dG, dC, ...),
using a simple constant-threshold rule per layer: abs(value) < threshold -> stable.

Deliberately simple starting point, not a final geomorphic stable-terrain
classification -- neither this repo nor DemDifferencing currently has a script
that PRODUCES a stable-cells raster (DemDifferencing's stable-mask logic lives
inline in xDEM_pipeline_Thomas_production.py::build_stable_mask and is only
ever plotted, never written to disk). Thresholds are placeholders to be tuned
per site/layer; more conditions (slope, G-range, elevation, aspect, ...) can
be appended to the `layers` list later without changing the combine logic.

Config:
    stable_cells:
      layers:
        - name: ddem
          path: /path/to/dDEM.tif
          threshold: 1.0
        - name: dg
          path: /path/to/DeltaG_1m_dinf.tif
          threshold: 3.0
        - name: dc
          path: /path/to/DeltaC_1m.tif
          threshold: 0.1
      combine: all      # "all" (AND, default) or "any" (OR)
      output: /path/to/stable_cells_1m.tif

A pixel is stable only where every listed layer is valid and below its
threshold (combine: all) -- or where at least one is (combine: any).
Missing files or shape-mismatched rasters are skipped with a warning, not
treated as fatal, so this still works when only some layers exist for a
given chunk.

Also writes a per-layer mask (<output>_<name>.tif) alongside the combined
one, so a concentrated/lopsided mask is visible before it gets used to
calibrate a LoD (see the "avoid masks concentrated in one terrain type"
caveat in the DemDifferencing README).

Run:
    conda run -n ml_debris python scripts/compute_stable_cells.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import yaml


def load(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr, profile


def write(path: Path, mask: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    out = np.where(np.isfinite(mask), mask, -9999.0).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def threshold_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    """1.0 where |arr| < threshold, 0.0 where valid but not, NaN where arr is NaN."""
    mask = np.full(arr.shape, np.nan, dtype="float64")
    valid = np.isfinite(arr)
    mask[valid] = (np.abs(arr[valid]) < threshold).astype("float64")
    return mask


def combine_masks(stacked: np.ndarray, mode: str) -> np.ndarray:
    """stacked: (n_layers, H, W) of {0, 1, NaN}. Returns combined {0, 1, NaN}."""
    valid = np.isfinite(stacked)
    if mode == "all":
        all_valid = np.all(valid, axis=0)
        filled = np.where(valid, stacked, 1.0)  # non-stable elsewhere is moot; all_valid gates it
        result = np.all(filled == 1.0, axis=0)
        any_or_all_valid = all_valid
    elif mode == "any":
        any_or_all_valid = np.any(valid, axis=0)
        filled = np.where(valid, stacked, 0.0)
        result = np.any(filled == 1.0, axis=0)
    else:
        raise ValueError(f"combine must be 'all' or 'any', got {mode!r}")
    return np.where(any_or_all_valid, result.astype("float64"), np.nan)


def report(label: str, mask: np.ndarray, source: str = "") -> None:
    n_valid = int(np.isfinite(mask).sum())
    n_stable = int(np.nansum(mask == 1.0))
    pct = 100 * n_stable / n_valid if n_valid else float("nan")
    suffix = f"  -> {source}" if source else ""
    print(f"  {label}: stable={n_stable:,}/{n_valid:,} ({pct:.1f}%){suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/source_zones.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sc_cfg = cfg["stable_cells"]
    combine = sc_cfg.get("combine", "all")
    out_path = Path(sc_cfg["output"])

    masks: dict[str, np.ndarray] = {}
    ref_profile = None
    ref_shape = None

    for layer in sc_cfg["layers"]:
        name = layer["name"]
        path = layer["path"]
        threshold = float(layer["threshold"])

        if not Path(path).exists():
            print(f"  SKIP  {name}  (not found: {path})")
            continue

        arr, profile = load(path)

        if ref_shape is None:
            ref_profile, ref_shape = profile, arr.shape
        elif arr.shape != ref_shape:
            print(f"  SKIP  {name}  (shape {arr.shape} != reference {ref_shape} -- "
                  f"align it first, e.g. via 10_align_features.py)")
            continue

        mask = threshold_mask(arr, threshold)
        masks[name] = mask
        report(f"{name}  (|.|<{threshold})", mask, path)

        per_layer_path = out_path.parent / f"{out_path.stem}_{name}{out_path.suffix}"
        write(per_layer_path, mask, profile)

    if not masks:
        raise SystemExit("No layers found -- nothing to combine. Check paths in config.")

    stacked = np.stack(list(masks.values()), axis=0)
    combined = combine_masks(stacked, combine)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(out_path, combined, ref_profile)
    print()
    report(f"Combined ({combine}, {len(masks)} layer(s): {list(masks)})", combined, str(out_path))


if __name__ == "__main__":
    main()
