"""Generate a binary hillslope/channel mask at 10m from G_pre (10m).

Reprojects G_pre_10m to the pipeline reference grid (same as G_mean_10m.tif).
Hillslope domain: g_min <= G < g_channel  → mask = 1 (keep)
Channel / flat  : G >= g_channel or G < g_min → mask = 0 (exclude)

Output: channel_mask_10m.tif  saved to output_dir.labels

Run:
    conda run -n ml_debris python scripts/compute_channel_mask.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/source_zones.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    g_pre_10m_path = Path(cfg["inputs"]["g_pre_10m"])
    out_dir        = Path(cfg["output_dir"]["labels"])
    g_min          = float(cfg["masking"]["g_min"])
    g_channel      = float(cfg["masking"]["g_channel"])
    out_path       = out_dir / "channel_mask_10m.tif"

    # ── Use G_mean_10m as reference grid ───────────────────────────
    ref_path = out_dir / "G_mean_10m.tif"
    with rasterio.open(ref_path) as ref:
        ref_profile   = ref.profile.copy()
        ref_crs       = ref.crs
        ref_transform = ref.transform
        ref_shape     = (ref.height, ref.width)

    print(f"G_pre 10m  : {g_pre_10m_path.name}")
    print(f"Reference  : {ref_path.name}  shape={ref_shape}  crs={ref_crs}")
    print(f"g_min      : {g_min}")
    print(f"g_channel  : {g_channel}")

    # ── Reproject G_before to reference grid ───────────────────────
    with rasterio.open(g_pre_10m_path) as src:
        g_reproj = np.full(ref_shape, np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=g_reproj,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    print(f"G reprojected: min={np.nanmin(g_reproj):.2f}  max={np.nanmax(g_reproj):.2f}")

    # ── Apply threshold ────────────────────────────────────────────
    mask = np.where(
        np.isfinite(g_reproj) & (g_reproj >= g_min) & (g_reproj < g_channel),
        1, 0
    ).astype("uint8")

    # ── Write ──────────────────────────────────────────────────────
    prof = ref_profile.copy()
    prof.update(dtype="uint8", count=1, nodata=255, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(mask, 1)

    print(f"\nWritten: {out_path.name}  shape={mask.shape}")
    print(f"  hillslope pixels : {int((mask == 1).sum()):,}")
    print(f"  channel/flat     : {int((mask == 0).sum()):,}")


if __name__ == "__main__":
    main()
