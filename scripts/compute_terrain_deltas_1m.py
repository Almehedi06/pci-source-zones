"""Compute 1m D-infinity G/DeltaG and plan-curvature/DeltaC from pre/post bare-earth DEMs.

Reproduces the method previously run ad-hoc in a notebook cell (FIS_v1.ipynb)
to build the currently-trusted thomas/G_1m_dinf_delta/{G_pre_dinf,G_post_dinf,
DeltaG_1m_dinf}.tif files, as a repeatable script for any fire/AOI. Also adds
DeltaC (plan curvature post - pre), previously ad-hoc in notebooks/variables.ipynb
and never wired into the aligned feature stack.

G (evidence for C2 in compute_p_obs.py):
    fill sinks:  WhiteboxTools fill_depressions_wang_and_liu(fix_flats=True)
    slope:       WhiteboxTools slope(units="percent"), converted to m/m
    a:           WhiteboxTools d_inf_flow_accumulation(out_type="sca")
    G = a * S^alpha
    DeltaG = G_post - G_pre

Curvature (ML feature only -- not folded into p_obs evidence):
    curvature:   WhiteboxTools plan_curvature on the raw (unfilled) DEM --
                 pure surface derivative, not flow-routed, so no fill needed.
    DeltaC = curvature_post - curvature_pre

Run:
    conda run -n ml_debris python scripts/compute_terrain_deltas_1m.py --config config/source_zones.yaml

Validated 2026-09-01: output is an exact, zero-diff match against the
existing hand-run thomas/G_1m_dinf_delta/{G_pre_dinf,G_post_dinf,
DeltaG_1m_dinf}.tif files that current p_obs/C_eff results were built on.
Safe to point compute_dg_lod.py / compute_p_obs.py at this script's
output_dir.g_dinf whenever inputs.g_pre_1m / delta_g_1m are repointed there.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import whitebox
import yaml


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr, profile


def write_raster(path: Path, arr: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def compute_g_dinf(
    wbt: "whitebox.WhiteboxTools", dem_path: Path, out_dir: Path, tag: str, alpha: float
) -> tuple[np.ndarray, dict]:
    filled    = out_dir / f"{tag}_filled.tif"
    slope_pct = out_dir / f"S_{tag}_percent.tif"
    sca       = out_dir / f"a_{tag}_dinf_sca.tif"
    G_path    = out_dir / f"G_{tag}_dinf.tif"

    wbt.fill_depressions_wang_and_liu(str(dem_path), str(filled), fix_flats=True)
    wbt.slope(str(filled), str(slope_pct), units="percent")
    wbt.d_inf_flow_accumulation(str(filled), str(sca), out_type="sca")

    S_pct, profile = read_raster(slope_pct)
    a, _ = read_raster(sca)

    S = S_pct / 100.0
    G = a * (S**alpha)
    G[~np.isfinite(G)] = np.nan
    G[G < 0] = np.nan

    write_raster(G_path, G, profile)
    print(f"  G_{tag}: mean={np.nanmean(G):.4f}  max={np.nanmax(G):.4f}  -> {G_path.name}")
    return G, profile


def compute_curvature(
    wbt: "whitebox.WhiteboxTools", dem_path: Path, out_dir: Path, tag: str
) -> tuple[np.ndarray, dict]:
    curv_path = out_dir / f"curvature_{tag}_1m.tif"
    wbt.plan_curvature(str(dem_path), str(curv_path))
    curv, profile = read_raster(curv_path)
    print(f"  curvature_{tag}: mean={np.nanmean(curv):.4f}  -> {curv_path.name}")
    return curv, profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/source_zones.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    inp     = cfg["inputs"]
    out_dir = Path(cfg["output_dir"]["g_dinf"])
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha   = float(cfg.get("terrain_dinf", {}).get("alpha", 1.167))

    pre_dem  = Path(inp["dem_pre_1m"])
    post_dem = Path(inp["dem_post_1m"])

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False

    print(f"Pre  DEM: {pre_dem}")
    print(f"Post DEM: {post_dem}")
    print(f"Output  : {out_dir}")
    print(f"alpha   : {alpha}\n")

    print("G (D-infinity)...")
    G_pre,  profile = compute_g_dinf(wbt, pre_dem, out_dir, "pre", alpha)
    G_post, _        = compute_g_dinf(wbt, post_dem, out_dir, "post", alpha)

    DeltaG = G_post - G_pre
    write_raster(out_dir / "DeltaG_1m_dinf.tif", DeltaG, profile)
    print(f"  DeltaG: mean={np.nanmean(DeltaG):.4f}  -> DeltaG_1m_dinf.tif\n")

    print("Curvature (plan)...")
    curv_pre,  cprofile = compute_curvature(wbt, pre_dem, out_dir, "pre")
    curv_post, _         = compute_curvature(wbt, post_dem, out_dir, "post")

    DeltaC = curv_post - curv_pre
    write_raster(out_dir / "DeltaC_1m.tif", DeltaC, cprofile)
    print(f"  DeltaC: mean={np.nanmean(DeltaC):.4f}  -> DeltaC_1m.tif")


if __name__ == "__main__":
    main()
