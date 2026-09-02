"""Fit USGS M1 (Staley et al. 2017) on Montecito stream segments and compare
against our own p_obs/C prediction, aggregated to the same segments.

Uses BARC4-from-dNBR as a soil-burn-severity proxy (real BAER SBS not yet
obtained for Thomas) and NOAA Atlas 14 design-storm i15 (not the actual
Jan 9, 2018 event). See m1_comparison_experiment.md for caveats.

Run (needs the pfdf311 env, not ml_debris):
    conda run -n pfdf311 python scripts/15_compare_m1.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pfdf import severity, watershed
from pfdf.models import staley2017
from pfdf.models.staley2017 import M1
from pfdf.raster import Raster
from pfdf.segments import Segments

BASE = Path("/mnt/c/Users/amehedi/Downloads/thomas")
ALIGNED = BASE / "aligned"
OUT_DIR = Path.home() / "ml_output" / "m1_comparison"
QGIS_DIR = BASE / "m1_comparison"

MIN_CONTRIBUTING_AREA_KM2 = 0.02

# Atlas 14, Montecito centroid, mean/intensity/pds/metric (mm/hr) by ARI (yrs)
ARI_YEARS = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
I15_MMHR = np.array([33, 43, 54, 64, 77, 86, 96, 106, 119, 130])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QGIS_DIR.mkdir(parents=True, exist_ok=True)

    dem = Raster.from_file(BASE / "dem_pre.tif")
    dnbr = Raster.from_file(ALIGNED / "dnbr_aligned.tif")  # already dNBR*1000 scale
    kf = Raster.from_file(ALIGNED / "kf_factor_aligned.tif")

    print("Conditioning DEM and computing flow...")
    conditioned = watershed.condition(dem)
    flow = watershed.flow(conditioned)
    slopes = watershed.slopes(conditioned, flow)  # gradients, TauDEM style

    pixel_area_km2 = dem.pixel_area(units="kilometers")
    area = watershed.accumulation(flow, times=pixel_area_km2)
    mask = area.values >= MIN_CONTRIBUTING_AREA_KM2

    print(f"Building stream segment network (contributing area >= {MIN_CONTRIBUTING_AREA_KM2} km^2)...")
    segments = Segments(flow, mask)
    print(f"  {segments.size} segments")

    # Flag segments whose upslope catchment touches the domain boundary. Their
    # contributing area is truncated by the DEM extent, so every M1 variable
    # (all of which are upslope accumulations) is computed over an incomplete
    # catchment. At Montecito this mostly catches the back side of the Santa
    # Ynez ridge, which drains away from the study area entirely.
    valid = np.isfinite(dem.values) & (dem.values != dem.nodata)
    boundary = valid & ~(
        np.pad(valid[1:, :], ((0, 1), (0, 0)))
        & np.pad(valid[:-1, :], ((1, 0), (0, 0)))
        & np.pad(valid[:, 1:], ((0, 0), (0, 1)))
        & np.pad(valid[:, :-1], ((0, 0), (1, 0)))
    )
    boundary_acc = watershed.accumulation(flow, mask=boundary)
    truncated = segments.catchment_summary("outlet", boundary_acc) > 0
    print(f"  {truncated.sum()} segments have catchments truncated by the domain edge")

    print("Estimating BARC4-from-dNBR (SBS proxy) and M1 variables...")
    barc4 = severity.estimate(dnbr)
    moderate_high = severity.mask(barc4, ["moderate", "high"])
    T, F, S = M1.variables(segments, moderate_high, slopes, dnbr, kf, omitnan=True)

    R = I15_MMHR * 0.25  # mm/hr -> mm accumulated in 15 min
    B, Ct, Cf, Cs = M1.parameters([15])
    P = staley2017.likelihood(R, B, Ct, T, Cf, F, Cs, S)  # shape (segments, len(R))
    P = np.atleast_2d(P)
    if P.shape[0] != segments.size:
        P = P.T

    print("Aggregating our own p_obs/C to the same segments...")
    p_obs = Raster.from_file(ALIGNED / "p_obs_union_aligned.tif")
    c_eff = Raster.from_file(ALIGNED / "C_p_obs_union_aligned.tif")
    our_p_obs = segments.catchment_summary("nanmean", p_obs)
    our_c = segments.catchment_summary("nanmean", c_eff)

    df = pd.DataFrame(
        {
            "segment_id": segments.ids,
            "T": T,
            "F": F,
            "S": S,
            "our_p_obs_mean": our_p_obs,
            "our_c_mean": our_c,
            "truncated": truncated.astype(int),
        }
    )
    for i, ari in enumerate(ARI_YEARS):
        df[f"M1_P_ari{ari}"] = P[:, i]

    out_csv = OUT_DIR / "m1_segments.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} segments)")

    # QGIS-ready exports: stream lines and outlet points, both carrying the
    # M1 / observed values so they can be styled and inspected directly.
    props = {col: df[col].to_numpy() for col in df.columns}
    segments.save(OUT_DIR / "segments.geojson", type="segments", properties=props, overwrite=True)
    for name, export_type in [
        ("m1_segments_lines", "segments"),
        ("m1_segment_outlets", "segment outlets"),  # one point per segment, not just terminal
        ("m1_terminal_outlets", "outlets"),         # terminal drainage outlets only
    ]:
        segments.save(
            QGIS_DIR / f"{name}.gpkg",
            type=export_type,
            properties=props,
            driver="GPKG",
            overwrite=True,
        )
        print(f"Saved {QGIS_DIR / (name + '.gpkg')}")

    complete = df[df["truncated"] == 0]
    print(
        f"\nCorrelation: M1 P vs our p_obs_union, complete catchments only "
        f"(n={complete['our_p_obs_mean'].notna().sum()} of {len(df)})"
    )
    for ari in ARI_YEARS:
        r_all = df[f"M1_P_ari{ari}"].corr(df["our_p_obs_mean"])
        r_ok = complete[f"M1_P_ari{ari}"].corr(complete["our_p_obs_mean"])
        print(f"  ARI={ari:>4} yr: r={r_ok:.3f}   (all segments incl. truncated: {r_all:.3f})")

    print(
        f"\nCorrelation: M1 P vs our C_eff, complete catchments only "
        f"(n={complete['our_c_mean'].notna().sum()}; C only exists where p_obs>0)"
    )
    for ari in ARI_YEARS:
        print(f"  ARI={ari:>4} yr: r={complete[f'M1_P_ari{ari}'].corr(complete['our_c_mean']):.3f}")


if __name__ == "__main__":
    main()
