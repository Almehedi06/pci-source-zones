"""Compute a data-driven LoD threshold for DeltaG from stable cells.

Uses Lauren's stable_cells mask to extract DeltaG values where no real
change occurred (pure noise).

Method: NMAD (Normalized Median Absolute Deviation)
    NMAD = 1.4826 x median(|DeltaG|) in stable cells
    LoD  = 2 x NMAD  (equivalent to 2-sigma for Gaussian noise)

Writes result to <labels_dir>/dg_lod.json so compute_p_obs.py can read it.

Run:
    conda run -n ml_debris python scripts/compute_dg_lod.py --config config/source_zones.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def load(path: str, nodata=None) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = nodata if nodata is not None else src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
    return arr


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

    delta_g = load(cfg["inputs"]["delta_g_1m"])
    out_dir = Path(cfg["output_dir"]["labels"])

    excl           = cfg.get("lod_channel_exclusion", {})
    use_stable_only = excl.get("use_stable_only", True)

    # Build sample mask: stable cells only OR all AOI pixels
    if use_stable_only:
        stable      = load(cfg["inputs"]["stable_cells"])
        stable_mask = (stable == 1) & np.isfinite(delta_g)
        print("Sample: stable cells only")
    else:
        stable_mask = np.isfinite(delta_g)
        print("Sample: ALL AOI pixels (stable_cells ignored)")

    # Optional: exclude channel pixels from LoD sample (Erkan: channels bias threshold upward)
    if excl.get("apply", False):
        g_pre      = load(cfg["inputs"]["g_pre_1m"])
        g_channel  = float(excl.get("g_channel", 120))
        channel    = (g_pre >= g_channel)
        n_before   = int(stable_mask.sum())
        stable_mask = stable_mask & ~channel
        n_excluded = n_before - int(stable_mask.sum())
        print(f"Channel exclusion ON  (G_pre >= {g_channel}): {n_excluded:,} pixels removed")
    else:
        print("Channel exclusion OFF")

    dg_stable = np.abs(delta_g[stable_mask])
    n_stable  = int(stable_mask.sum())

    nmad = 1.4826 * float(np.median(dg_stable))
    lod  = 2.0 * nmad

    print(f"Stable pixels : {n_stable:,}")
    print(f"median(|DeltaG|) in stable cells = {np.median(dg_stable):.4f}")
    print(f"NMAD = {nmad:.4f}")
    print(f"-> DELTA_G_LOD = {lod:.4f}  (2 x NMAD, used as C2 threshold)")

    result = {
        "delta_g_lod": lod,
        "percentile":  "NMAD-based",
        "nmad":        nmad,
        "n_stable":    n_stable,
    }
    out_path = out_dir / "dg_lod.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
