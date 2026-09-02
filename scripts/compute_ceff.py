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
    C_p_obs_inter.tif      — C_eff from p_obs_intersection
    C_p_obs_union.tif      — C_eff from p_obs_union
    C_p_obs_*_log.tif      — log versions of above
    C_p005.tif             — C upper bound: inverted at p_obs=0.05 (Erkan)
                             Use for p_obs=0 cells — high resistance estimate
    C_p005_log.tif         — log version
    ceff_fit.json          — fitted k, scale, n_hillslope_cells

Run:
    conda run -n ml_debris python scripts/compute_ceff.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import fiona
from rasterio.enums import Resampling
from rasterio.features import rasterize
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


def build_aoi_mask(aoi_path: str, shape: tuple, transform) -> np.ndarray:
    """Rasterize AOI shapefile to grid. Returns bool array (True = inside AOI)."""
    with fiona.open(aoi_path) as src:
        geoms = [feat["geometry"] for feat in src]
    mask = rasterize(geoms, out_shape=shape, transform=transform,
                     fill=0, default_value=1, dtype="uint8")
    return mask.astype(bool)


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

    elev_min             = float(ceff.get("elev_min",             600))
    g_hillslope_max      = float(ceff.get("g_hillslope_max",      200))
    g_mean_min           = float(ceff.get("g_mean_min",             0))
    p_obs_ceil           = float(ceff.get("p_obs_ceil",          0.05))
    elev_filter_outputs  = bool(ceff.get("elev_filter_outputs",  False))

    # ── Load G_mean (10m) ────────────────────────────────────────────────
    g_mean, prof = load(out_dir / "G_mean_10m.tif")
    H, W         = g_mean.shape
    print(f"Loaded G_mean{g_mean.shape}")

    # ── Load elevation resampled to 10m grid ─────────────────────────────
    elev = load_match(inp["dem_pre"], H, W)
    print(f"Elevation resampled to {elev.shape}")

    # ── Load channel mask (hillslope=1, channel/flat=0) ─────────────────
    mask_path = out_dir / "channel_mask_10m.tif"
    if mask_path.exists():
        ch_mask_raw = load_match(mask_path, H, W)   # resample to G_mean grid
        ch_mask = ch_mask_raw > 0.5                  # nearest-neighbour equivalent
        print(f"Channel mask loaded: {ch_mask.sum():,} hillslope pixels")
    else:
        ch_mask = (g_mean > g_mean_min) & (g_mean < g_hillslope_max)
        print("Channel mask not found — falling back to g_mean threshold")

    # ── Step 1: Fit Gamma to hillslope G_mean distribution ───────────────
    hillslope = (np.isfinite(g_mean) & np.isfinite(elev) &
                 (elev > elev_min) & ch_mask)
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
                   "p_obs_methods": ["intersection", "union"]}, f, indent=2)
    print(f"  Saved -> {fit_path}")

    # ── Optional AOI mask ─────────────────────────────────────────────────
    aoi_path = inp.get("aoi")
    aoi_mask = None
    if aoi_path:
        aoi_mask = build_aoi_mask(aoi_path, g_mean.shape, prof["transform"])
        print(f"AOI mask: {np.sum(~aoi_mask):,} pixels outside AOI → NaN")

    if elev_filter_outputs:
        print(f"Elevation filter ON for outputs: elev > {elev_min} m")
    else:
        print("Elevation filter OFF for outputs (applied to Gamma fit only)")

    def invert(p_obs: np.ndarray, label: str) -> np.ndarray:
        """Gamma-invert one p_obs layer → C_eff mean map."""
        valid = np.isfinite(g_mean) & np.isfinite(p_obs) & ch_mask & (p_obs > 0)
        if elev_filter_outputs:
            valid &= (elev > elev_min)
        ratio = np.full_like(p_obs, np.nan)
        ratio[valid] = np.clip(p_obs[valid], 1e-6, 1 - 1e-6)
        inv_std = np.full_like(g_mean, np.nan)
        inv_std[valid] = gamma.ppf(ratio[valid], a=k_fit, scale=1)
        scale_map = np.full_like(g_mean, np.nan)
        ok = valid & (inv_std > 0)
        scale_map[ok] = g_mean[ok] / inv_std[ok]
        mean_map = k_fit * scale_map
        if aoi_mask is not None:
            mean_map[~aoi_mask] = np.nan
        print(f"  {label}: {ok.sum():,} valid cells inverted")
        return mean_map

    # ── Step 2: Invert both p_obs methods ────────────────────────────────
    print("\nStep 2 — Inversion (both methods)...")
    methods = ["intersection", "union"]
    results = {}
    for method in methods:
        p_path = out_dir / f"p_obs_{method}.tif"
        if not p_path.exists():
            print(f"  SKIP {method} — {p_path.name} not found")
            continue
        p_obs, _ = load(p_path)
        results[method] = invert(p_obs, method)

    # ── Write outputs ─────────────────────────────────────────────────────
    print("\nWriting outputs...")
    for method, mean_map in results.items():
        save(out_dir / f"C_p_obs_{method[:5]}.tif",      mean_map,                prof)
        save(out_dir / f"C_p_obs_{method[:5]}_log.tif",  np.log(np.clip(mean_map, 1e-6, None)), prof)

    # backward-compat alias — C_eff_mean_10m matches whichever method ran last
    if "union" in results:
        save(out_dir / "C_eff_mean_10m.tif", results["union"], prof)

    # ── Step 3: C upper bound — invert at fixed p_obs = p_obs_ceil ───────
    # Erkan: "infer C for p as low as 0.05 and call it the highest value of C"
    # For p_obs=0 cells we cannot invert directly. Instead, invert at p_ceil
    # (small but non-zero) → gives large C (high resistance) as upper bound.
    # Spatially variable through G_mean; p_ceil is the same for every cell.
    print(f"\nStep 3 — C upper bound at p_obs={p_obs_ceil} ...")
    q_ceil  = gamma.ppf(p_obs_ceil, a=k_fit, scale=1)   # scalar Gamma quantile
    valid_hs = np.isfinite(g_mean) & ch_mask
    if elev_filter_outputs:
        valid_hs &= (elev > elev_min)
    C_ceil   = np.full_like(g_mean, np.nan)
    C_ceil[valid_hs] = k_fit * (g_mean[valid_hs] / q_ceil)
    if aoi_mask is not None:
        C_ceil[~aoi_mask] = np.nan

    tag = f"p{int(p_obs_ceil * 100):03d}"   # e.g. "p005" for 0.05
    save(out_dir / f"C_{tag}.tif",     C_ceil,                                  prof)
    save(out_dir / f"C_{tag}_log.tif", np.log(np.clip(C_ceil, 1e-6, None)),     prof)

    print("Done.")


if __name__ == "__main__":
    main()
