"""Align all ML input feature rasters to the reference grid.

Reference: dem_diff.tif  →  height=835, width=716, EPSG:32611, 10 m
(same grid as the physics_dod classification target)

Run:
    conda run -n ml_debris python scripts/10_align_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

BASE = Path("/mnt/c/Users/amehedi/Downloads/thomas")
GEE = BASE / "gee_lai_ndvi_et"
LABELS = Path("/mnt/c/Users/amehedi/Downloads/source_zone_labels")
OUT = BASE / "aligned"
OUT.mkdir(exist_ok=True)

# Single reference for ALL rasters — same grid as physics_dod classification target
REFERENCE = BASE / "dem_diff.tif"

# (source_path, output_name, resampling_method)
TO_ALIGN = [
    # --- terrain (from original 10m topographic outputs) ---
    (BASE / "S_from_topographic_elevation.tif",                "slope_aligned.tif",         Resampling.bilinear),
    (BASE / "topographic__specific_contributing_area.tif",     "drainage_area_aligned.tif",  Resampling.bilinear),
    (BASE / "topographic__curvature.tif",                      "curvature_aligned.tif",      Resampling.bilinear),
    (BASE / "topographic__aspect.tif",                         "aspect_aligned.tif",         Resampling.bilinear),
    (BASE / "topographic__elevation.tif",                      "elevation_aligned.tif",      Resampling.bilinear),
    # --- fire effect ---
    (BASE / "burn__severity.tif",                              "burn_severity_aligned.tif",  Resampling.bilinear),
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
    (BASE / "soil__thickness.tif",                             "soil_thickness_aligned.tif", Resampling.bilinear),
    (BASE / "sand__total.tif",                                 "sand_total_aligned.tif",     Resampling.bilinear),
    (BASE / "silt__total.tif",                                 "silt_total_aligned.tif",     Resampling.bilinear),
    (BASE / "clay__total.tif",                                 "clay_total_aligned.tif",     Resampling.bilinear),
    (BASE / "porosity.tif",                                    "porosity_aligned.tif",       Resampling.bilinear),
    (BASE / "soil__saturated_hydraulic_conductivity.tif",      "ksat_aligned.tif",           Resampling.bilinear),
    (BASE / "pH.tif",                                          "ph_aligned.tif",             Resampling.bilinear),
    (BASE / "field__capacity.tif",                             "field_capacity_aligned.tif", Resampling.bilinear),
    (BASE / "landcover.tif",                                   "landcover_aligned.tif",      Resampling.nearest),
    # --- regression target (align to same grid for consistency) ---
    (LABELS / "src_hi_g10_dg1_gc120_s2_bt5.tif",              "target_src_hi_aligned.tif",  Resampling.bilinear),
    # --- p_obs: observed rill initiation probability (C2 AND C3 at 1m, aggregated to 10m) ---
    (LABELS / "frac_c2c3_dg1_gc120.tif",                      "p_obs_c2c3_aligned.tif",     Resampling.bilinear),
    # --- G*: max 1m G_pre per 10m cell (terrain forcing for probability theory) ---
    (LABELS / "G_star_10m.tif",                               "G_star_aligned.tif",         Resampling.bilinear),
]


NODATA_OUT = -9999.0  # written to all output rasters; read_raster converts this to NaN


def align_to_reference(src_path: Path, dst_path: Path, resampling: Resampling) -> None:
    with rasterio.open(REFERENCE) as ref:
        dst_profile = ref.profile.copy()
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata  # read source nodata from metadata
        data = np.full((1, dst_shape[0], dst_shape[1]), NODATA_OUT, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=NODATA_OUT,
        )

    # Mask any remaining sentinel values (e.g. -999999, -32768) that weren't caught
    data = data.astype("float32")
    data[~np.isfinite(data)] = NODATA_OUT
    data[(data < -9000) & (data != NODATA_OUT)] = NODATA_OUT

    dst_profile.update(dtype="float32", count=1, nodata=NODATA_OUT)
    with rasterio.open(dst_path, "w", **dst_profile) as dst:
        dst.write(data)


def main() -> None:
    print(f"Reference grid: {REFERENCE.name}")
    with rasterio.open(REFERENCE) as ref:
        print(f"  {ref.height} rows × {ref.width} cols  {ref.crs}  {ref.res[0]:.0f} m\n")

    for src_path, out_name, resampling in TO_ALIGN:
        dst_path = OUT / out_name
        if not src_path.exists():
            print(f"  SKIP  {out_name}  (source not found: {src_path.name})")
            continue
        print(f"  aligning  {out_name} ...", end=" ", flush=True)
        align_to_reference(src_path, dst_path, resampling)
        print("done")

    print("\nAll done. Aligned rasters saved to:", OUT)


if __name__ == "__main__":
    main()
