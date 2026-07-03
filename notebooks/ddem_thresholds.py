from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject


DEFAULT_DDEM = Path(
    "/mnt/c/Users/amehedi/Downloads/"
    "Thomas_xDEM-20260619T021336Z-3-001/"
    "Thomas_xDEM/post-pre_COREG_BIAS_ASPECT_o1_a90_dDEM.tif"
)

DEFAULT_OUT_DIR = Path("/mnt/c/Users/amehedi/Downloads/thomas")
DEFAULT_AOI = DEFAULT_OUT_DIR / "montecito_aoi.shp"
DEFAULT_TEMPLATE = DEFAULT_OUT_DIR / "topographic__elevation.tif"


def _read_ddem_on_template(ddem_path, template_path):
    """Reproject/resample dDEM to the template raster grid."""
    with rasterio.open(template_path) as template, rasterio.open(ddem_path) as src:
        ddem = np.full((template.height, template.width), np.nan, dtype="float32")

        reproject(
            source=src.read(1),
            destination=ddem,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=template.transform,
            dst_crs=template.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

        profile = template.profile.copy()
        pixel_area_m2 = abs(template.transform.a * template.transform.e)

    return ddem, profile, pixel_area_m2


def _valid_template_mask(template_path, aoi_path=None):
    """Use template valid pixels, optionally clipped to AOI polygon."""
    with rasterio.open(template_path) as template:
        template_arr = template.read(1, masked=True).filled(np.nan)
        valid = np.isfinite(template_arr)

        if aoi_path is not None and Path(aoi_path).exists():
            aoi = gpd.read_file(aoi_path).to_crs(template.crs)
            geoms = [geom for geom in aoi.geometry if geom is not None and not geom.is_empty]

            if geoms:
                aoi_mask = geometry_mask(
                    geoms,
                    out_shape=(template.height, template.width),
                    transform=template.transform,
                    invert=True,
                )
                valid = valid & aoi_mask

    return valid


def run_threshold_analysis(
    ddem_path=DEFAULT_DDEM,
    out_dir=DEFAULT_OUT_DIR,
    template_path=DEFAULT_TEMPLATE,
    aoi_path=DEFAULT_AOI,
    thresholds=(0.01, 0.03, 0.05, 0.08, 0.10, 0.25, 0.03, 0.50, 1.00),
):
    """Create binary erosion maps where dDEM < -threshold on the template grid."""
    ddem_path = Path(ddem_path)
    out_dir = Path(out_dir)
    template_path = Path(template_path)
    aoi_path = Path(aoi_path) if aoi_path is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)

    ddem, profile, pixel_area_m2 = _read_ddem_on_template(ddem_path, template_path)
    template_valid = _valid_template_mask(template_path, aoi_path)

    valid = np.isfinite(ddem) & template_valid
    aoi_area_m2 = valid.sum() * pixel_area_m2

    if aoi_area_m2 == 0:
        raise ValueError("No valid dDEM pixels overlap the template/AOI.")

    ddem_for_output = ddem.copy()
    ddem_for_output[~valid] = np.nan

    aligned_profile = profile.copy()
    aligned_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    aligned_ddem = out_dir / "corrected_dDEM_on_topographic_template.tif"
    with rasterio.open(aligned_ddem, "w", **aligned_profile) as dst:
        dst.write(ddem_for_output.astype("float32"), 1)

    profile.update(dtype="uint8", count=1, nodata=255, compress="lzw")

    rows = []

    for threshold in thresholds:
        erosion = valid & (ddem < -threshold)

        out_arr = np.full(ddem.shape, 255, dtype="uint8")
        out_arr[valid] = 0
        out_arr[erosion] = 1

        tag = str(threshold).replace(".", "p")
        out_tif = out_dir / f"erosion_threshold_{tag}m.tif"
        out_png = out_dir / f"erosion_threshold_{tag}m.png"

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(out_arr, 1)

        plt.figure(figsize=(8, 6))
        plt.imshow(out_arr == 1, cmap="Reds")
        plt.title(f"Erosion where dDEM < -{threshold} m")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

        erosion_area_m2 = erosion.sum() * pixel_area_m2

        rows.append(
            {
                "threshold_m": -threshold,
                "erosion_area_m2": erosion_area_m2,
                "erosion_area_km2": erosion_area_m2 / 1_000_000,
                "percent_of_AOI": 100 * erosion_area_m2 / aoi_area_m2,
                "output_tif": str(out_tif),
                "output_png": str(out_png),
            }
        )

    summary = pd.DataFrame(rows)
    summary_csv = out_dir / "erosion_threshold_area_summary.csv"
    summary.to_csv(summary_csv, index=False)

    vals = np.sort(ddem[valid].ravel())
    cdf = np.arange(1, len(vals) + 1) / len(vals) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(vals, cdf, color="black")
    for threshold in thresholds:
        plt.axvline(-threshold, linestyle="--", label=f"-{threshold} m")
    plt.xlabel("dDEM value, post - pre (m)")
    plt.ylabel("Cumulative percent of pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "corrected_dDEM_CDF.png", dpi=300)
    plt.close()

    return summary


if __name__ == "__main__":
    print(run_threshold_analysis())
