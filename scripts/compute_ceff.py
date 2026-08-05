"""Invert p_obs and G_mean to estimate spatially variable soil resistance C_eff.

Framework: Istanbulluoglu et al. (2002), Water Resources Research.

Pipeline position:
    compute_dg_lod.py  ->  compute_p_obs.py  ->  compute_ceff.py  (this script)

Steps:
    1. Fit Gamma(k, scale) to G_mean distribution on hillslopes
       (elev > ceff.elev_min AND G_mean < ceff.g_hillslope_max)
    2. For each 10m cell, invert p_obs to recover local soil resistance:
           q          = gamma.ppf(p_obs; k, scale=1)   [standard Gamma quantile]
           scale_local = G_mean / q
           C_eff_mean  = k x scale_local

Outputs (saved to output_dir.labels):
    C_eff_scale_10m.tif  — local Gamma scale parameter
    C_eff_mean_10m.tif   — mean soil resistance  (primary output)
    ceff_fit.json        — fitted k, scale, n_hillslope_cells

Run:
    conda run -n ml_debris python scripts/compute_ceff.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.stats import gamma


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        prof = src.profile.copy()
    return arr, prof


def load_match(path: str | Path, H: int, W: int) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, out_shape=(H, W),
                       resampling=Resampling.bilinear).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
    return arr


def save(path: Path, arr: np.ndarray, prof: dict) -> None:
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    p = prof.copy()
    p.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)
    valid = out[out != -9999.0]
    print(f"  {path.name}  shape={out.shape}  "
          f"median={np.median(valid):.2f}  mean={valid.mean():.2f}  max={valid.max():.2f}")


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/source_zones.yaml")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.experiment:
        with open(args.experiment) as f:
            override = yaml.safe_load(f)
        cfg = deep_merge(cfg, override)
        print(f"Experiment override: {args.experiment}")

    inp     = cfg["inputs"]
    out_dir = Path(cfg["output_dir"]["labels"])
    ceff    = cfg.get("ceff", {})

    p_obs_method    = ceff.get("p_obs_method",    "union")
    elev_min        = float(ceff.get("elev_min",        600))
    g_hillslope_max = float(ceff.get("g_hillslope_max", 200))
    g_mean_min      = float(ceff.get("g_mean_min",        0))

    # ── Load G_mean and p_obs (10m) ──────────────────────────────────────
    g_mean, prof = load(out_dir / "G_mean_10m.tif")
    p_obs, _     = load(out_dir / f"p_obs_{p_obs_method}.tif")
    H, W         = g_mean.shape
    print(f"Loaded G_mean{g_mean.shape}  p_obs_{p_obs_method}{p_obs.shape}")

    # ── Load elevation resampled to 10m grid ─────────────────────────────
    elev = load_match(inp["dem_pre"], H, W)
    print(f"Elevation resampled to {elev.shape}")

    # ── Step 1: Fit Gamma to hillslope G_mean distribution ───────────────
    hillslope = (np.isfinite(g_mean) & np.isfinite(elev) &
                 (elev > elev_min) &
                 (g_mean > g_mean_min) & (g_mean < g_hillslope_max))
    g_flat = g_mean[hillslope].ravel()

    k_fit, _, scale_fit = gamma.fit(g_flat, floc=0)
    print(f"\nStep 1 — Gamma fit on hillslope cells (elev>{elev_min}, G<{g_hillslope_max})")
    print(f"  n = {len(g_flat):,}")
    print(f"  k={k_fit:.4f}  scale={scale_fit:.4f}  mean G={k_fit * scale_fit:.2f}")

    fit_path = out_dir / "ceff_fit.json"
    with open(fit_path, "w") as f:
        json.dump({"k": float(k_fit), "scale": float(scale_fit),
                   "mean_g": float(k_fit * scale_fit),
                   "n_hillslope": int(len(g_flat)),
                   "elev_min": elev_min,
                   "g_hillslope_max": g_hillslope_max,
                   "p_obs_method": p_obs_method}, f, indent=2)
    print(f"  Saved -> {fit_path}")

    # ── Step 2: Inversion — recover local scale and mean C_eff ───────────
    valid = (np.isfinite(g_mean) & np.isfinite(p_obs) &
             (g_mean > g_mean_min) & (g_mean < g_hillslope_max) &
             (p_obs > 0))

    ratio = np.full_like(p_obs, np.nan)
    ratio[valid] = np.clip(p_obs[valid], 1e-6, 1 - 1e-6)

    inv_std = np.full_like(g_mean, np.nan)
    inv_std[valid] = gamma.ppf(ratio[valid], a=k_fit, scale=1)

    scale_map = np.full_like(g_mean, np.nan)
    mask = valid & (inv_std > 0)
    scale_map[mask] = g_mean[mask] / inv_std[mask]
    mean_map = k_fit * scale_map

    print(f"\nStep 2 — Inversion complete  (valid cells: {mask.sum():,})")

    # ── Write outputs ─────────────────────────────────────────────────────
    print("\nWriting outputs...")
    save(out_dir / "C_eff_scale_10m.tif", scale_map, prof)
    save(out_dir / "C_eff_mean_10m.tif",  mean_map,  prof)
    print("Done.")


if __name__ == "__main__":
    main()
