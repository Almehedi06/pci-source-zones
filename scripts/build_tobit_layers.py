"""Build censored.tif and logC_bound.tif for Tobit UNet training.

censored_aligned.tif  : 1 where p_obs_union == 0 AND valid hillslope, 0 elsewhere
logC_bound_aligned.tif: log1p(C_lower_bound) for censored cells only

C_lower_bound is the minimum C_eff consistent with observing p_obs = 0:
    C_bound = k * G_mean / gamma.ppf(P_CRIT, a=k, scale=1)
where P_CRIT = 3/100 means "at most 3 of 100 pixels could have initiated".

Run:
    conda run -n ml_debris python scripts/build_tobit_layers.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import gamma as gamma_dist

P_CRIT = 3 / 100  # upper bound on p_obs for "effectively zero" initiation


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        return arr, src.profile.copy()


def align_to_ml_grid(
    arr: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
    ref_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    arr_in = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    out = np.full((1, ref_shape[0], ref_shape[1]), -9999.0, dtype="float32")
    reproject(
        source=arr_in[np.newaxis],
        destination=out,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=-9999.0,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        dst_nodata=-9999.0,
        resampling=resampling,
    )
    result = out[0]
    result[result == -9999.0] = np.nan
    return result


def save_aligned(path: Path, arr: np.ndarray, ref_profile: dict) -> None:
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    prof = ref_profile.copy()
    prof.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
    valid = out[out != -9999.0]
    print(f"  {path.name}: n={len(valid):,}  min={valid.min():.3f}  "
          f"median={np.median(valid):.3f}  max={valid.max():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build censored.tif and logC_bound.tif for Tobit UNet.")
    parser.add_argument("--config", default="config/source_zones.yaml",
                        help="Path to source_zones.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    LABELS    = Path(cfg["output_dir"]["labels"])
    ALIGNED   = Path(cfg["output_dir"]["aligned"])
    REFERENCE = ALIGNED / "slope_aligned.tif"

    # Load Gamma fit
    fit = json.loads((LABELS / "ceff_fit.json").read_text())
    k = float(fit["k"])
    print(f"Gamma fit: k={k:.4f}  scale={fit['scale']:.4f}")

    q_crit = float(gamma_dist.ppf(P_CRIT, a=k, scale=1))
    print(f"q_crit = gamma.ppf({P_CRIT}, k={k:.3f}) = {q_crit:.6f}")
    print(f"C_bound multiplier = k / q_crit = {k / q_crit:.2f}× G_mean")

    # Load native 10m rasters
    p_obs, p_prof = load_raster(LABELS / "p_obs_union.tif")
    g_mean, _     = load_raster(LABELS / "G_mean_10m.tif")

    mask_path = LABELS / "channel_mask_10m.tif"
    if mask_path.exists():
        ch_raw, _ = load_raster(mask_path)
        ch_mask = ch_raw > 0.5
        print(f"Channel mask: {ch_mask.sum():,} hillslope pixels")
    else:
        ch_mask = np.isfinite(g_mean)
        print("Channel mask not found — using all finite G_mean pixels")

    # Censored: p_obs == 0 AND hillslope AND valid G_mean
    domain = np.isfinite(p_obs) & np.isfinite(g_mean) & ch_mask
    cen_mask = domain & (p_obs == 0)
    obs_mask = domain & (p_obs >  0)

    print(f"\nDomain: {domain.sum():,} pixels")
    print(f"  Observed  (p_obs > 0): {obs_mask.sum():,}")
    print(f"  Censored  (p_obs = 0): {cen_mask.sum():,}")

    # censored array: 1 inside domain where p=0, 0 elsewhere in domain, NaN outside domain
    censored = np.full_like(g_mean, np.nan)
    censored[domain] = 0.0
    censored[cen_mask] = 1.0

    # logC_bound: only filled for censored cells
    logC_bound = np.full_like(g_mean, np.nan)
    if q_crit > 0 and cen_mask.any():
        C_bound = k * g_mean[cen_mask] / q_crit
        logC_bound[cen_mask] = np.log1p(C_bound)

    valid_bound = logC_bound[np.isfinite(logC_bound)]
    print(f"\nlogC_bound (censored cells): min={valid_bound.min():.2f}  "
          f"median={np.median(valid_bound):.2f}  max={valid_bound.max():.2f}")

    # Load reference profile
    with rasterio.open(REFERENCE) as ref:
        ref_profile = ref.profile.copy()
        ref_shape = (ref_profile["height"], ref_profile["width"])

    print(f"\nAligning to ML grid {ref_shape}...")
    cen_aligned   = align_to_ml_grid(censored,   p_prof, ref_profile, ref_shape, Resampling.nearest)
    bound_aligned = align_to_ml_grid(logC_bound, p_prof, ref_profile, ref_shape, Resampling.bilinear)

    # Snap censored to binary after bilinear-free reproject
    finite_cen = np.isfinite(cen_aligned)
    cen_aligned[finite_cen] = (cen_aligned[finite_cen] > 0.5).astype("float32")

    print("\nSaving...")
    save_aligned(ALIGNED / "censored_aligned.tif",    cen_aligned,   ref_profile)
    save_aligned(ALIGNED / "logC_bound_aligned.tif",  bound_aligned, ref_profile)
    print("Done.")


if __name__ == "__main__":
    main()
