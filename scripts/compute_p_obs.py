"""Compute p_obs and G_star at 10m from 1m LiDAR evidence.

p_obs  = fraction of 1m pixels per 10m cell where BOTH C2 AND C3 fire
           C2: DeltaG > dg_lod  (flow network reorganized)
           C3: dDEM  < 0        (surface lowered; LoD filtering already applied)

G_star = max G_pre per 10m cell (terrain forcing for Gamma probability theory)

Reads dg_lod from <labels_dir>/dg_lod.json (output of compute_dg_lod.py).

Run:
    conda run -n ml_debris python scripts/compute_p_obs.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import fiona
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject as warp_reproject


def load(path: str, nodata=None) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd  = nodata if nodata is not None else src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        profile = src.profile.copy()
    return arr, profile


def block_aggregate(arr: np.ndarray, block: int, func) -> np.ndarray:
    H, W  = arr.shape
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=np.nan)
    H2, W2 = padded.shape
    blocks = padded.reshape(H2 // block, block, W2 // block, block)
    with np.errstate(all="ignore"):
        return func(blocks).astype("float32")


def write(path: Path, arr: np.ndarray, profile: dict, transform10: Affine) -> None:
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    H, W = out.shape
    prof = profile.copy()
    prof.update(driver="GTiff", height=H, width=W, count=1,
                dtype="float32", nodata=-9999.0,
                transform=transform10, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
    valid = out[out != -9999.0]
    print(f"  {path.name}  shape={out.shape}  "
          f"max={valid.max():.4f}  mean={valid.mean():.4f}")


def build_aoi_mask(aoi_path: str, shape: tuple, transform: Affine) -> np.ndarray:
    """Rasterize AOI shapefile to grid. Returns bool array (True = inside AOI)."""
    with fiona.open(aoi_path) as src:
        geoms = [feat["geometry"] for feat in src]
    mask = rasterize(geoms, out_shape=shape, transform=transform,
                     fill=0, default_value=1, dtype="uint8")
    return mask.astype(bool)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base — override wins on conflicts."""
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
    parser.add_argument("--experiment", default=None,
                        help="Experiment override config (e.g. config/experiments/exp02_union.yaml)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.experiment:
        with open(args.experiment) as f:
            override = yaml.safe_load(f)
        cfg = deep_merge(cfg, override)
        print(f"Experiment override: {args.experiment}")

    inp       = cfg["inputs"]
    params    = cfg["params"]
    out_dir   = Path(cfg["output_dir"]["labels"])
    block    = int(params["block_size"])
    ddem_thr = float(params["ddem_threshold"])

    # Load dg_lod
    lod_path = out_dir / "dg_lod.json"
    if not lod_path.exists():
        raise FileNotFoundError(
            f"{lod_path} not found — run compute_dg_lod.py first."
        )
    with open(lod_path) as f:
        dg_lod = float(json.load(f)["delta_g_lod"])
    print(f"Using DELTA_G_LOD = {dg_lod:.4f}")

    # Load rasters
    g_pre,   prof = load(inp["g_pre_1m"])
    delta_g, _    = load(inp["delta_g_1m"])
    ddem,    _    = load(inp["ddem_1m"], nodata=-9999.0)
    print(f"Loaded  g_pre{g_pre.shape}  delta_g{delta_g.shape}  ddem{ddem.shape}")

    # C2 and C3 at 1m
    # NaN comparisons return False — propagate nodata explicitly before casting
    c2 = (delta_g > dg_lod).astype("float32")
    c2[~np.isfinite(delta_g)] = np.nan

    c3 = (ddem < ddem_thr).astype("float32")
    c3[~np.isfinite(ddem)] = np.nan

    # Optional knowledge-based masking (separate from core data-driven p_obs)
    masking    = cfg.get("masking", {})
    g_pre_masked = g_pre.copy()
    if masking.get("apply", False):
        g_min     = float(masking.get("g_min",     0))
        g_channel = float(masking.get("g_channel", float("inf")))
        kmask = (g_pre < g_min) | (g_pre >= g_channel)
        c2[kmask] = np.nan
        c3[kmask] = np.nan
        g_pre_masked[kmask] = np.nan
        print(f"Knowledge mask applied: G < {g_min} or G >= {g_channel}")
    else:
        g_min     = 0
        g_channel = float("inf")
        print("No knowledge masking applied — fully data-driven")

    # 10m aggregation — always compute both intersection and union
    valid  = np.isfinite(c2) & np.isfinite(c3)
    both   = np.where(valid, (c2 == 1) & (c3 == 1), np.nan).astype("float32")

    p_c2          = block_aggregate(np.where(valid, c2,   np.nan).astype("float32"),
                                    block, lambda b: np.nanmean(b, axis=(1, 3)))
    p_c3          = block_aggregate(np.where(valid, c3,   np.nan).astype("float32"),
                                    block, lambda b: np.nanmean(b, axis=(1, 3)))
    p_obs_inter   = block_aggregate(both, block, lambda b: np.nanmean(b, axis=(1, 3)))
    p_obs_union   = np.clip(p_c2 + p_c3 - p_obs_inter, 0.0, 1.0)
    g_star = block_aggregate(g_pre_masked, block,
                             lambda b: np.nanmax(b, axis=(1, 3)))
    g_mean = block_aggregate(g_pre_masked, block,
                             lambda b: np.nanmean(b, axis=(1, 3)))

    # Optional G_star cap (removes Dinf SCA artifacts near main channels)
    g_star_cap = params.get("g_star_cap", None)
    if g_star_cap is not None:
        g_star_cap = float(g_star_cap)
        clipped = np.sum(np.isfinite(g_star) & (g_star > g_star_cap))
        g_star = np.clip(g_star, None, g_star_cap)
        print(f"G_star capped at {g_star_cap} ({clipped} cells clipped)")

    # Build 10m transform
    t0  = prof["transform"]
    t10 = Affine(t0.a * block, t0.b, t0.c, t0.d, t0.e * block, t0.f)

    # ── Channel mask from G_before (raw unmasked 10m G) ─────────────────────
    # G_mean is computed from already-masked 1m pixels so it never reaches
    # g_channel (300) at 10m — useless for channel exclusion at 10m scale.
    # Instead reproject G_before.tif (unmasked 10m G) to the pipeline grid
    # and apply the same thresholds. Save as channel_mask_10m.tif for reuse.
    H10, W10 = p_obs_inter.shape
    g_pre_10m_path = inp.get("g_pre_10m")
    ch_mask_path   = out_dir / "channel_mask_10m.tif"

    if g_pre_10m_path and Path(g_pre_10m_path).exists():
        g10 = np.full((H10, W10), np.nan, dtype="float32")
        with rasterio.open(g_pre_10m_path) as src:
            warp_reproject(
                source=rasterio.band(src, 1),
                destination=g10,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=t10,
                dst_crs=prof["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        ch_valid = np.isfinite(g10) & (g10 >= g_min) & (g10 < g_channel)
        for arr in [p_obs_inter, p_obs_union, p_c2, p_c3, g_star, g_mean]:
            arr[~ch_valid] = np.nan
        print(f"Channel mask (G_before 10m): {ch_valid.sum():,} hillslope kept, "
              f"{(~ch_valid).sum():,} masked (channel/flat/nodata)")

        # Save channel_mask_10m.tif so compute_ceff.py and others can reuse it
        ch_prof = prof.copy()
        ch_prof.update(dtype="uint8", count=1, nodata=255,
                       height=H10, width=W10, transform=t10, compress="lzw")
        ch_out = np.where(ch_valid, np.uint8(1), np.uint8(0))
        with rasterio.open(ch_mask_path, "w", **ch_prof) as dst:
            dst.write(ch_out, 1)
        print(f"  Saved -> {ch_mask_path.name}")
    else:
        print("WARNING: g_pre_10m not found — channel mask not applied to p_obs")

    # Optional AOI mask — pixels outside watershed boundary → NaN
    aoi_path = inp.get("aoi")
    if aoi_path:
        aoi_mask = build_aoi_mask(aoi_path, p_obs_inter.shape, t10)
        for arr in [p_obs_inter, p_obs_union, g_star, g_mean]:
            arr[~aoi_mask] = np.nan
        print(f"AOI mask applied: {np.sum(~aoi_mask):,} pixels set to NaN outside {aoi_path}")

    print("Writing outputs...")
    write(out_dir / "p_obs_intersection.tif", p_obs_inter, prof, t10)
    write(out_dir / "p_obs_union.tif",        p_obs_union, prof, t10)
    write(out_dir / "G_star_10m.tif",         g_star,      prof, t10)
    write(out_dir / "G_mean_10m.tif",         g_mean,      prof, t10)

    # Optional: save 1m LoD-filtered dG intermediate
    if params.get("save_intermediates", False):
        dg_filtered = np.where(np.isfinite(c2), c2, -9999.0).astype("float32")
        H1, W1 = dg_filtered.shape
        p1 = prof.copy()
        p1.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
        with rasterio.open(out_dir / "dg_lod_filtered_c2_1m.tif", "w", **p1) as dst:
            dst.write(dg_filtered, 1)
        print(f"  dg_lod_filtered_c2_1m.tif  (1m binary, dG > {dg_lod:.2f})")

    print("Done.")


if __name__ == "__main__":
    main()
