"""Align all ML input feature rasters to the reference grid.

Reference and AOI are read from --config (source_zones.yaml):
  inputs.dem_pre  →  reference grid (835×716, EPSG:32611, 10m)
  inputs.aoi      →  AOI shapefile  (pixels outside → nodata)

Run:
    conda run -n ml_debris python scripts/10_align_features.py
    conda run -n ml_debris python scripts/10_align_features.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fiona
import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_aoi_mask(aoi_path: Path, ref_profile: dict, ref_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize AOI polygon to reference grid. Returns bool array (True = inside AOI)."""
    with fiona.open(aoi_path) as src:
        geoms = [feat["geometry"] for feat in src]
    mask = rasterize(
        geoms,
        out_shape=ref_shape,
        transform=ref_profile["transform"],
        fill=0,
        default_value=1,
        dtype="uint8",
    )
    return mask.astype(bool)


def align_to_reference(
    src_path: Path,
    dst_path: Path,
    resampling: Resampling,
    aoi_mask: np.ndarray,
    ref_profile: dict,
    ref_shape: tuple[int, int],
    ref_crs,
    ref_transform,
    nodata_out: float = -9999.0,
) -> None:
    with rasterio.open(src_path) as src:
        data = np.full((1, ref_shape[0], ref_shape[1]), nodata_out, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=nodata_out,
        )

    data = data.astype("float32")
    data[~np.isfinite(data)] = nodata_out
    data[(data < -9000) & (data != nodata_out)] = nodata_out
    data[0, ~aoi_mask] = nodata_out

    prof = ref_profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata_out)
    with rasterio.open(dst_path, "w", **prof) as dst:
        dst.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/source_zones.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    inp      = cfg["inputs"]
    out_dirs = cfg["output_dir"]

    REFERENCE = Path(inp["dem_pre"])
    AOI       = Path(inp["aoi"])
    BASE      = REFERENCE.parent
    GEE       = BASE / "gee_lai_ndvi_et"
    PRISM     = BASE / "prism_forcing_montecito/aligned_tif"
    LABELS    = Path(out_dirs["labels"])
    G_DINF    = Path(out_dirs.get("g_dinf", str(BASE)))
    OUT       = Path(out_dirs.get("aligned", str(BASE / "aligned")))
    OUT.mkdir(exist_ok=True)

    NODATA_OUT = -9999.0

    # (source_path, output_name, resampling_method)
    TO_ALIGN = [
        # --- terrain ---
        (BASE / "S_from_topographic_elevation.tif",                "slope_aligned.tif",            Resampling.bilinear),
        (BASE / "topographic__specific_contributing_area.tif",     "drainage_area_aligned.tif",    Resampling.bilinear),
        (BASE / "topographic__curvature.tif",                      "curvature_aligned.tif",        Resampling.bilinear),
        (BASE / "topographic__aspect.tif",                         "aspect_aligned.tif",           Resampling.bilinear),
        (BASE / "topographic__elevation.tif",                      "elevation_aligned.tif",        Resampling.bilinear),
        # --- fire effect ---
        (BASE / "burn_severity_thomas.tif",                        "burn_severity_aligned.tif",    Resampling.nearest),
        (BASE / "dnbr_thomas.tif",                                 "dnbr_aligned.tif",              Resampling.bilinear),
        (
            GEE / "ET_median_pre_2017-04-01_to_2017-08-30_post_2018-04-01_to_2018-08-30_diff_post_minus_pre_aligned.tif",
            "et_diff_aligned.tif",
            Resampling.bilinear,
        ),
        (
            GEE / "NDVI_median_pre_2017-04-01_to_2017-08-30_post_2018-04-01_to_2018-08-30_diff_post_minus_pre_aligned.tif",
            "ndvi_diff_aligned.tif",
            Resampling.bilinear,
        ),
        # --- soil properties ---
        (BASE / "soil__thickness.tif",                             "soil_thickness_aligned.tif",   Resampling.bilinear),
        (BASE / "kffact_thomas.tif",                               "kf_factor_aligned.tif",         Resampling.bilinear),
        (BASE / "thick_thomas.tif",                                "soil_thickness_statsgo_aligned.tif", Resampling.bilinear),
        (BASE / "sand__total.tif",                                 "sand_total_aligned.tif",       Resampling.bilinear),
        (BASE / "silt__total.tif",                                 "silt_total_aligned.tif",       Resampling.bilinear),
        (BASE / "clay__total.tif",                                 "clay_total_aligned.tif",       Resampling.bilinear),
        (BASE / "porosity.tif",                                    "porosity_aligned.tif",         Resampling.bilinear),
        (BASE / "soil__saturated_hydraulic_conductivity.tif",      "ksat_aligned.tif",             Resampling.bilinear),
        (BASE / "pH.tif",                                          "ph_aligned.tif",               Resampling.bilinear),
        (BASE / "field__capacity.tif",                             "field_capacity_aligned.tif",   Resampling.bilinear),
        (BASE / "landcover.tif",                                   "landcover_aligned.tif",        Resampling.nearest),
        # --- precipitation (PRISM daily, Jan 2018 storm) ---
        (PRISM / "ppt/precip_20180109.tif",                        "precip_jan9_aligned.tif",      Resampling.bilinear),
        (PRISM / "ppt/precip_20180110.tif",                        "precip_jan10_aligned.tif",     Resampling.bilinear),
        # --- labels / targets ---
        (LABELS / "p_obs_intersection.tif",                        "p_obs_inter_aligned.tif",      Resampling.bilinear),
        (LABELS / "p_obs_union.tif",                               "p_obs_union_aligned.tif",      Resampling.bilinear),
        (LABELS / "G_star_10m.tif",                                "G_star_aligned.tif",           Resampling.bilinear),
        (LABELS / "G_mean_10m.tif",                                "G_mean_aligned.tif",           Resampling.bilinear),
        (LABELS / "C_p_obs_inter.tif",                             "C_p_obs_inter_aligned.tif",    Resampling.bilinear),
        (LABELS / "C_p_obs_inter_log.tif",                         "C_p_obs_inter_log_aligned.tif", Resampling.bilinear),
        (LABELS / "C_p_obs_union.tif",                             "C_p_obs_union_aligned.tif",    Resampling.bilinear),
        (LABELS / "C_p_obs_union_log.tif",                         "C_p_obs_union_log_aligned.tif",  Resampling.bilinear),
        (LABELS / "C_p005.tif",                                    "C_p005_aligned.tif",             Resampling.bilinear),
        (LABELS / "C_p005_log.tif",                                "C_p005_log_aligned.tif",         Resampling.bilinear),
        (LABELS / "C_eff_mean_10m.tif",                            "C_eff_mean_aligned.tif",       Resampling.bilinear),
        (LABELS / "channel_mask_10m.tif",                          "channel_mask_aligned.tif",     Resampling.nearest),
        (LABELS / "src_hi_g10_dg1_gc120_s2_bt5.tif",              "target_src_hi_aligned.tif",    Resampling.bilinear),
        # --- G pre/post (10m, unmasked) ---
        (BASE / "G_before.tif",                                    "G_before_aligned.tif",         Resampling.bilinear),
        (BASE / "G_after.tif",                                     "G_after_aligned.tif",          Resampling.bilinear),
        # --- delta curvature (1m, ML feature only -- scripts/compute_terrain_deltas_1m.py) ---
        (G_DINF / "curvature_pre_1m.tif",                          "curv_pre_aligned.tif",         Resampling.bilinear),
        (G_DINF / "curvature_post_1m.tif",                         "curv_post_aligned.tif",        Resampling.bilinear),
        (G_DINF / "DeltaC_1m.tif",                                 "curv_delta_aligned.tif",       Resampling.bilinear),
    ]

    print(f"Reference grid : {REFERENCE.name}")
    with rasterio.open(REFERENCE) as ref:
        ref_profile   = ref.profile.copy()
        ref_crs       = ref.crs
        ref_transform = ref.transform
        ref_shape     = (ref.height, ref.width)
        print(f"  {ref.height} rows × {ref.width} cols  {ref_crs}  {ref.res[0]:.0f} m")

    print(f"AOI mask       : {AOI.name}")
    aoi_mask = build_aoi_mask(AOI, ref_profile, ref_shape)
    print(f"  {aoi_mask.sum():,} pixels inside AOI  ({100*aoi_mask.mean():.1f}% of grid)\n")

    for src_path, out_name, resampling in TO_ALIGN:
        dst_path = OUT / out_name
        if not src_path.exists():
            print(f"  SKIP  {out_name}  (source not found: {src_path.name})")
            continue
        print(f"  aligning  {out_name} ...", end=" ", flush=True)
        align_to_reference(
            src_path, dst_path, resampling,
            aoi_mask, ref_profile, ref_shape, ref_crs, ref_transform, NODATA_OUT,
        )
        print("done")

    print(f"\nAll done. Aligned rasters saved to: {OUT}")


if __name__ == "__main__":
    main()
