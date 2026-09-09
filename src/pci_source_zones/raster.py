"""Shared raster I/O and array utilities.

Single source of truth for reading, writing, aggregating and validating rasters.
Previously reimplemented across six scripts with drift between the copies.

Design rules:
    * Functions take arrays and typed parameters, not config dicts.
    * Nodata is always represented internally as NaN in float arrays.
    * Nothing here knows about sites, fires, or pipeline stages.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

FLOAT_NODATA = -9999.0
AggMethod = Literal["mean", "max", "min", "sum"]


# ──────────────────────────────────────────────────────────── read

def read_float(path: str | Path, nodata: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Read band 1 as float32 with nodata as NaN.

    Args:
        path: raster path.
        nodata: override the file's declared nodata value.

    Returns:
        (array, profile)
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = nodata if nodata is not None else src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        profile = src.profile.copy()
    return arr, profile


def read_binary(path: str | Path, nodata: int = 255) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a 0/1 mask raster as float32 with nodata as NaN.

    Values equal to 1 become 1.0; all other valid values become 0.0.
    """
    with rasterio.open(path) as src:
        raw = src.read(1, masked=True).filled(nodata)
        profile = src.profile.copy()
    valid = raw != nodata
    arr = np.full(raw.shape, np.nan, dtype="float32")
    arr[valid] = (raw[valid] == 1).astype("float32")
    return arr, profile


def reproject_to_grid(path: str | Path, dst_transform: Affine, dst_crs: Any,
                      height: int, width: int,
                      resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Reproject a raster onto an explicit destination grid, nodata as NaN."""
    dst = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst


# ──────────────────────────────────────────────────────────── write

def clean_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Drop block/tiling keys that break profile reuse across differing shapes."""
    out = dict(profile)
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        out.pop(key, None)
    return out


def write_float(path: str | Path, arr: np.ndarray, profile: dict[str, Any],
                transform: Affine | None = None, nodata: float = FLOAT_NODATA) -> Path:
    """Write a float32 raster, converting NaN to `nodata`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    prof = clean_profile(profile)
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=nodata,
                height=out.shape[0], width=out.shape[1], compress="lzw")
    if transform is not None:
        prof["transform"] = transform
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
    return path


def write_uint8(path: str | Path, arr: np.ndarray, profile: dict[str, Any],
                nodata: int = 255, transform: Affine | None = None) -> Path:
    """Write a uint8 raster (masks, class rasters)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = clean_profile(profile)
    prof.update(driver="GTiff", dtype="uint8", count=1, nodata=nodata,
                height=arr.shape[0], width=arr.shape[1], compress="lzw")
    if transform is not None:
        prof["transform"] = transform
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("uint8"), 1)
    return path


# ──────────────────────────────────────────────────────────── aggregate

def block_aggregate(arr: np.ndarray, block: int, method: AggMethod = "mean") -> np.ndarray:
    """Aggregate a fine grid to coarse cells of `block` x `block` pixels.

    NaN-aware: a coarse cell containing only NaN returns NaN, not zero.
    The array is zero-padded with NaN to a whole number of blocks.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    height, width = arr.shape
    pad_h = (block - height % block) % block
    pad_w = (block - width % block) % block
    padded = np.pad(arr.astype("float32"), ((0, pad_h), (0, pad_w)), constant_values=np.nan)
    h2, w2 = padded.shape
    blocks = padded.reshape(h2 // block, block, w2 // block, block)
    # All-NaN blocks are expected (padding, masked regions) and correctly return
    # NaN; suppress the accompanying RuntimeWarning rather than the result.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(Mean|All-NaN)", RuntimeWarning)
        if method == "mean":
            return np.nanmean(blocks, axis=(1, 3)).astype("float32")
        if method == "max":
            return np.nanmax(blocks, axis=(1, 3)).astype("float32")
        if method == "min":
            return np.nanmin(blocks, axis=(1, 3)).astype("float32")
        if method == "sum":
            # nansum returns 0 for all-NaN slices; restore NaN explicitly
            out = np.nansum(blocks, axis=(1, 3)).astype("float32")
            allnan = np.all(~np.isfinite(blocks), axis=(1, 3))
            out[allnan] = np.nan
            return out
    raise ValueError(f"Unsupported aggregation method: {method!r}")


def transform_for_block(transform: Affine, block: int) -> Affine:
    """Scale an affine transform for a `block`-times coarser grid."""
    return Affine(transform.a * block, transform.b, transform.c,
                  transform.d, transform.e * block, transform.f)


def nansum_masked(cube: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Sum a stack along axis 0, keeping NaN where no layer had data.

    `np.nansum` returns 0.0 for an all-NaN slice, which silently converts
    "no data" into "zero". This restores NaN for those cells.
    """
    if valid is None:
        valid = np.isfinite(cube).any(axis=0)
    total = np.nansum(cube, axis=0)
    return np.where(valid, total, np.nan)


# ──────────────────────────────────────────────────────────── validate

class GridMismatch(ValueError):
    """Raised when rasters that must share a grid do not."""


# A COG rewrite or a reprojection round-trip can shift an origin by a fraction
# of a micron. That is float noise, not a misalignment, and resampling to "fix"
# it would degrade the data. Anything up to a hundredth of a pixel is accepted;
# a genuine half-pixel offset is still caught.
GRID_ATOL_PIXELS = 0.01


def transforms_match(a, b, *, atol_pixels: float = GRID_ATOL_PIXELS) -> bool:
    """True if two affine transforms describe the same grid within float noise."""
    if abs(a.a - b.a) > abs(a.a) * 1e-9 or abs(a.e - b.e) > abs(a.e) * 1e-9:
        return False                                  # different pixel size
    if abs(a.b - b.b) > 1e-12 or abs(a.d - b.d) > 1e-12:
        return False                                  # different rotation/skew
    return (abs(a.c - b.c) <= abs(a.a) * atol_pixels
            and abs(a.f - b.f) <= abs(a.e) * atol_pixels)


def grids_match(a: dict[str, Any], b: dict[str, Any], *, check_crs: bool = True) -> bool:
    """True if two profiles describe the same grid, tolerating float noise."""
    same = (a["height"] == b["height"] and a["width"] == b["width"]
            and transforms_match(a["transform"], b["transform"]))
    if check_crs:
        same = same and (a.get("crs") == b.get("crs"))
    return same


def require_same_grid(reference: dict[str, Any], others: dict[str, dict[str, Any]],
                      *, check_crs: bool = True) -> None:
    """Raise GridMismatch if any profile in `others` differs from `reference`.

    Args:
        reference: the profile everything must match.
        others: {label: profile} so the error names the offender.
    """
    bad = []
    for label, prof in others.items():
        if not grids_match(reference, prof, check_crs=check_crs):
            bad.append(f"{label} ({prof['height']}x{prof['width']})")
    if bad:
        raise GridMismatch(
            f"grid mismatch against reference ({reference['height']}x{reference['width']}): "
            + ", ".join(bad)
        )


def raster_stats(arr: np.ndarray) -> dict[str, float]:
    """Summary statistics over finite values. Empty-safe."""
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"n_valid": 0, "min": np.nan, "max": np.nan,
                "mean": np.nan, "median": np.nan, "std": np.nan}
    return {"n_valid": int(v.size), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std())}
