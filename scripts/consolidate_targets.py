"""Consolidate and rename all ML targets to final naming convention.

Label files produced in source_zone_labels/:
    p_obs_inter.tif         - P(C2 AND C3) per 10m cell
    p_obs_union.tif         - P(C2) + P(C3) - P(C2∩C3) per 10m cell
    C_p_obs_union.tif       - C_eff inverted from p_obs_union   (Pa)
    C_p_obs_inter.tif       - C_eff inverted from p_obs_inter   (Pa)
    C_p_obs_union_log.tif   - log1p(C_p_obs_union)
    C_p_obs_inter_log.tif   - log1p(C_p_obs_inter)

Aligned files produced in thomas/aligned/:
    p_obs_inter_aligned.tif
    p_obs_union_aligned.tif
    C_p_obs_union_aligned.tif
    C_p_obs_inter_aligned.tif
    C_p_obs_union_log_aligned.tif
    C_p_obs_inter_log_aligned.tif

Also deletes:
    frac_c2c3_dg1_gc120.tif
    frac_c2_dg1_gc120.tif
    frac_c3_gc120.tif
    mean_score_c2c3_dg1_gc120.tif

Run:
    conda run -n ml_debris python scripts/consolidate_targets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import fiona
import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.stats import gamma


# ── helpers ──────────────────────────────────────────────────────────────────

def load(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        prof = src.profile.copy()
    return arr, prof


def save(path: Path, arr: np.ndarray, prof: dict) -> None:
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    p = prof.copy()
    p.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)
    valid = out[out != -9999.0]
    n = len(valid)
    print(f"  -> {path.name}  n={n:,}  min={valid.min():.4f}  "
          f"mean={valid.mean():.4f}  max={valid.max():.4f}")


def build_aoi_mask(aoi_path: Path, shape: tuple, transform) -> np.ndarray:
    with fiona.open(aoi_path) as src:
        geoms = [feat["geometry"] for feat in src]
    mask = rasterize(geoms, out_shape=shape, transform=transform,
                     fill=0, default_value=1, dtype="uint8")
    return mask.astype(bool)


def align_raster(src_path: Path, dst_path: Path,
                 ref_profile: dict, ref_shape: tuple,
                 ref_crs, ref_transform,
                 aoi_mask: np.ndarray) -> None:
    nodata_out = -9999.0
    with rasterio.open(src_path) as src:
        data = np.full((1, ref_shape[0], ref_shape[1]), nodata_out, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=nodata_out,
        )
    data = data.astype("float32")
    data[~np.isfinite(data)] = nodata_out
    data[(data < -9000) & (data != nodata_out)] = nodata_out
    data[0, ~aoi_mask] = nodata_out

    prof = ref_profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata_out, compress="lzw")
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(data)
    arr = data[0]
    valid = arr[arr != nodata_out]
    print(f"  -> {dst_path.name}  n={len(valid):,}  min={valid.min():.4f}  "
          f"mean={valid.mean():.4f}  max={valid.max():.4f}")


def invert_ceff(p_obs: np.ndarray, g_mean: np.ndarray,
                k: float, ch_mask: np.ndarray) -> np.ndarray:
    """Invert p_obs + G_mean → C_eff_mean using Gamma quantile method."""
    valid = np.isfinite(g_mean) & np.isfinite(p_obs) & ch_mask & (p_obs > 0)
    ratio = np.full_like(p_obs, np.nan)
    ratio[valid] = np.clip(p_obs[valid], 1e-6, 1 - 1e-6)
    inv_std = np.full_like(g_mean, np.nan)
    inv_std[valid] = gamma.ppf(ratio[valid], a=k, scale=1)
    scale_map = np.full_like(g_mean, np.nan)
    mask = valid & (inv_std > 0)
    scale_map[mask] = g_mean[mask] / inv_std[mask]
    return k * scale_map  # C_eff_mean = k * scale_local


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("config/source_zones.yaml") as f:
        cfg = yaml.safe_load(f)

    inp      = cfg["inputs"]
    labels   = Path(cfg["output_dir"]["labels"])
    aligned  = Path(cfg["output_dir"]["aligned"])
    aligned.mkdir(exist_ok=True)

    # ── Reference grid (for alignment) ───────────────────────────────────
    ref_path = Path(inp["dem_pre"])
    with rasterio.open(ref_path) as ref:
        ref_profile   = ref.profile.copy()
        ref_shape     = (ref.height, ref.width)
        ref_crs       = ref.crs
        ref_transform = ref.transform

    aoi_mask = build_aoi_mask(Path(inp["aoi"]), ref_shape, ref_transform)
    print(f"Reference grid: {ref_shape[1]}x{ref_shape[0]}, CRS={ref_crs}")
    print(f"AOI mask: {aoi_mask.sum():,} valid pixels\n")

    # ── Load shared inputs ────────────────────────────────────────────────
    g_mean, prof_10m = load(labels / "G_mean_10m.tif")
    ch_mask_arr, _   = load(labels / "channel_mask_10m.tif")
    ch_mask = ch_mask_arr > 0.5  # True = hillslope

    p_inter, _ = load(labels / "p_obs_intersection.tif")
    p_union, _ = load(labels / "p_obs_union.tif")

    # ── Load Gamma fit (already computed, same for both p_obs methods) ────
    with open(labels / "ceff_fit.json") as f:
        fit = json.load(f)
    k = float(fit["k"])
    print(f"Gamma fit: k={k:.4f}, scale={fit['scale']:.4f} "
          f"(n_hillslope={fit['n_hillslope']:,})\n")

    # ── Step 1: p_obs files (rename/copy) ────────────────────────────────
    print("=== Step 1: p_obs label files ===")
    shutil.copy2(labels / "p_obs_intersection.tif", labels / "p_obs_inter.tif")
    print(f"  -> p_obs_inter.tif  (copy of p_obs_intersection.tif)")
    # p_obs_union.tif already correctly named — no copy needed

    # ── Step 2: Invert C_eff for both p_obs methods ───────────────────────
    print("\n=== Step 2: C_eff inversion ===")
    c_union = invert_ceff(p_union, g_mean, k, ch_mask)
    save(labels / "C_p_obs_union.tif", c_union, prof_10m)

    c_inter = invert_ceff(p_inter, g_mean, k, ch_mask)
    save(labels / "C_p_obs_inter.tif", c_inter, prof_10m)

    # ── Step 3: log1p transforms ──────────────────────────────────────────
    print("\n=== Step 3: log1p transforms ===")
    save(labels / "C_p_obs_union_log.tif", np.log1p(c_union), prof_10m)
    save(labels / "C_p_obs_inter_log.tif", np.log1p(c_inter), prof_10m)

    # ── Step 4: Align all 6 to ML reference grid ─────────────────────────
    print("\n=== Step 4: Aligning to ML reference grid ===")
    targets = [
        ("p_obs_inter.tif",       "p_obs_inter_aligned.tif"),
        ("p_obs_union.tif",       "p_obs_union_aligned.tif"),
        ("C_p_obs_union.tif",     "C_p_obs_union_aligned.tif"),
        ("C_p_obs_inter.tif",     "C_p_obs_inter_aligned.tif"),
        ("C_p_obs_union_log.tif", "C_p_obs_union_log_aligned.tif"),
        ("C_p_obs_inter_log.tif", "C_p_obs_inter_log_aligned.tif"),
    ]
    for src_name, dst_name in targets:
        align_raster(
            labels  / src_name,
            aligned / dst_name,
            ref_profile, ref_shape, ref_crs, ref_transform, aoi_mask,
        )

    # ── Step 5: Delete retired files ──────────────────────────────────────
    print("\n=== Step 5: Deleting retired files ===")
    to_delete = [
        "frac_c2c3_dg1_gc120.tif",
        "frac_c2_dg1_gc120.tif",
        "frac_c3_gc120.tif",
        "mean_score_c2c3_dg1_gc120.tif",
    ]
    for name in to_delete:
        p = labels / name
        if p.exists():
            p.unlink()
            print(f"  deleted {name}")
        else:
            print(f"  not found (skip): {name}")

    print("\nDone.")
    print("\nSummary — label files:")
    for src_name, _ in targets:
        p = labels / src_name
        print(f"  {'OK' if p.exists() else 'MISSING':6s} {p.name}")
    print("\nSummary — aligned files:")
    for _, dst_name in targets:
        p = aligned / dst_name
        print(f"  {'OK' if p.exists() else 'MISSING':6s} {p.name}")


if __name__ == "__main__":
    main()
