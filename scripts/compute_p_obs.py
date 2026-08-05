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
from rasterio.transform import Affine


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
    c2 = (delta_g > dg_lod).astype("float32")
    c3 = (ddem    < ddem_thr).astype("float32")

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
        print("No knowledge masking applied — fully data-driven")

    # 10m aggregation
    method = params.get("p_obs_method", "intersection")
    valid  = np.isfinite(c2) & np.isfinite(c3)
    both   = np.where(valid, (c2 == 1) & (c3 == 1), np.nan).astype("float32")

    if method == "union":
        p_c2   = block_aggregate(np.where(valid, c2, np.nan).astype("float32"),
                                 block, lambda b: np.nanmean(b, axis=(1, 3)))
        p_c3   = block_aggregate(np.where(valid, c3, np.nan).astype("float32"),
                                 block, lambda b: np.nanmean(b, axis=(1, 3)))
        p_both = block_aggregate(both, block, lambda b: np.nanmean(b, axis=(1, 3)))
        p_obs  = p_c2 + p_c3 - p_both
        print("p_obs method: UNION  P(C2) + P(C3) - P(C2&C3)")
    else:
        p_obs  = block_aggregate(both, block, lambda b: np.nanmean(b, axis=(1, 3)))
        print("p_obs method: INTERSECTION  P(C2 AND C3)")
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

    print("Writing outputs...")
    write(out_dir / f"p_obs_{method}.tif", p_obs,  prof, t10)
    write(out_dir / "G_star_10m.tif",      g_star, prof, t10)
    write(out_dir / "G_mean_10m.tif",      g_mean, prof, t10)

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
