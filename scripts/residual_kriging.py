"""Residual Kriging post-processor.

Loads an existing ML prediction raster, computes residuals at training pixels,
fits Ordinary Kriging on the residuals, and adds the kriged correction to the
prediction raster. Works on top of any model (RF, XGB, UNet).

Run:
    conda run -n ml_debris python scripts/residual_kriging.py \
        --config config/ml_random_forest.yaml \
        --prediction /mnt/c/.../random_forest_regressor_prediction.tif

Outputs (saved next to --prediction):
    *_rk_corrected.tif   — RF/XGB prediction + kriged residual
    *_rk_metrics.json    — train / test metrics after correction
    *_rk_variogram.json  — fitted variogram parameters
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import add_src_to_path
add_src_to_path()

from pci_source_zones.config import load_config


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile.copy()


def pixel_coords(flat_indices: np.ndarray, shape: tuple, transform) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) map coordinates for flat pixel indices."""
    rows, cols = np.unravel_index(flat_indices, shape)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    return xs.astype("float64"), ys.astype("float64")


def build_mask(gdf, ids, shape, transform) -> np.ndarray:
    shapes = [(g, 1) for g, i in zip(gdf.geometry, gdf["id"]) if i in ids]
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0).astype(bool)


def regression_metrics(y_true, y_pred, label: str) -> dict:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ml_random_forest.yaml")
    parser.add_argument("--prediction", required=True, help="Path to ML prediction raster (log1p space)")
    parser.add_argument("--n_sample", type=int, default=3000,
                        help="Training pixels to subsample for kriging (default 3000)")
    parser.add_argument("--variogram", default="spherical",
                        choices=["spherical", "exponential", "gaussian", "linear"],
                        help="Variogram model (default: spherical)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    poly_cfg = cfg["ml"]["split"]["polygons"]
    train_ids = list(poly_cfg["train_ids"])
    test_ids  = list(poly_cfg["test_ids"])
    poly_path = poly_cfg["path"]
    id_field  = poly_cfg.get("id_field", "id")
    target_path = cfg["ml"]["target"]["path"]

    pred_path = Path(args.prediction)
    out_dir   = pred_path.parent

    print(f"[rk] Prediction : {pred_path.name}")
    print(f"[rk] Target     : {Path(target_path).name}")

    # Load rasters
    pred, prof  = load_raster(pred_path)
    target, _   = load_raster(target_path)
    shape       = pred.shape
    transform   = prof["transform"]

    # Load polygon masks
    with rasterio.open(pred_path) as src:
        raster_crs = src.crs.to_string()
    gdf = gpd.read_file(poly_path).to_crs(raster_crs)
    train_mask = build_mask(gdf, train_ids, shape, transform)
    test_mask  = build_mask(gdf, test_ids,  shape, transform)

    valid = np.isfinite(pred) & np.isfinite(target) & (target > 0)
    tr_idx = np.flatnonzero((valid & train_mask).ravel())
    te_idx = np.flatnonzero((valid & test_mask).ravel())

    print(f"[rk] Train pixels: {len(tr_idx):,}  |  Test pixels: {len(te_idx):,}")

    # Residuals at training pixels
    y_tr   = target.ravel()[tr_idx]
    yh_tr  = pred.ravel()[tr_idx]
    resid  = y_tr - yh_tr

    print(f"[rk] Residual stats — mean: {resid.mean():.4f}  std: {resid.std():.4f}  "
          f"min: {resid.min():.4f}  max: {resid.max():.4f}")

    # Subsample for kriging (full set too expensive)
    rng = np.random.default_rng(42)
    n_sample = min(args.n_sample, len(tr_idx))
    sel = rng.choice(len(tr_idx), size=n_sample, replace=False)
    tr_sel = tr_idx[sel]
    resid_sel = resid[sel]
    xs_tr, ys_tr = pixel_coords(tr_sel, shape, transform)

    print(f"[rk] Fitting variogram on {n_sample} subsampled training points ...")

    # Fit Ordinary Kriging on residuals
    from pykrige.ok import OrdinaryKriging

    OK = OrdinaryKriging(
        xs_tr, ys_tr, resid_sel,
        variogram_model=args.variogram,
        verbose=False,
        enable_plotting=False,
    )

    params = OK.variogram_model_parameters
    print(f"[rk] Variogram ({args.variogram}): nugget={params[2]:.4f}  "
          f"sill={params[0]:.4f}  range={params[1]:.1f}m")

    # Predict kriged residuals on ALL valid pixels — chunked to avoid memory crash
    all_idx = np.flatnonzero(valid.ravel())
    print(f"[rk] Predicting kriged residuals on {len(all_idx):,} pixels (chunked) ...")
    chunk_size = 5000
    krig_resid = np.zeros(len(all_idx), dtype="float32")
    for i in range(0, len(all_idx), chunk_size):
        chunk = all_idx[i:i + chunk_size]
        xs_c, ys_c = pixel_coords(chunk, shape, transform)
        kr, _ = OK.execute("points", xs_c, ys_c, backend="vectorized")
        krig_resid[i:i + len(chunk)] = np.asarray(kr, dtype="float32")
        if i % 50000 == 0 and i > 0:
            print(f"[rk]   {i:,}/{len(all_idx):,} pixels done")

    # Build corrected prediction raster
    corrected = pred.copy()
    corrected.ravel()[all_idx] += krig_resid

    # Evaluate
    corr_flat = corrected.ravel()
    yh_tr_corr = corr_flat[tr_idx]
    yh_te_corr = corr_flat[te_idx]
    y_te = target.ravel()[te_idx]

    metrics = {
        "train_before": regression_metrics(y_tr,  pred.ravel()[tr_idx], "train_before"),
        "train_after":  regression_metrics(y_tr,  yh_tr_corr,           "train_after"),
        "test_before":  regression_metrics(y_te,  pred.ravel()[te_idx],  "test_before"),
        "test_after":   regression_metrics(y_te,  yh_te_corr,           "test_after"),
    }

    for k, v in metrics.items():
        print(f"  {k:<16}: R²={v['r2']:.4f}  MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}")

    # Write outputs
    stem = pred_path.stem
    corr_path = out_dir / f"{stem}_rk_corrected.tif"
    prof.update(dtype="float32", nodata=-9999.0, compress="lzw")
    out_arr = np.where(np.isfinite(corrected), corrected, -9999.0).astype("float32")
    with rasterio.open(corr_path, "w", **prof) as dst:
        dst.write(out_arr[np.newaxis])
    print(f"[rk] Corrected raster → {corr_path}")

    metrics_path = out_dir / f"{stem}_rk_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[rk] Metrics        → {metrics_path}")

    vario_path = out_dir / f"{stem}_rk_variogram.json"
    with open(vario_path, "w") as f:
        json.dump({
            "model": args.variogram,
            "nugget": float(params[2]),
            "sill":   float(params[0]),
            "range_m": float(params[1]),
            "n_sample": n_sample,
        }, f, indent=2)
    print(f"[rk] Variogram      → {vario_path}")


if __name__ == "__main__":
    main()
