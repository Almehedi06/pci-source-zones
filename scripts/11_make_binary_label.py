"""Create a binary source-zone label from the continuous src_hi raster.

A pixel is labelled 1 (source zone) if src_hi >= threshold, else 0.
Nodata pixels in the source are written as 255.

Usage:
    conda run -n ml_debris python scripts/11_make_binary_label.py --threshold 0.1
    conda run -n ml_debris python scripts/11_make_binary_label.py --threshold 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

NODATA = 255


def make_binary_label(src_path: Path, out_dir: Path, threshold: float) -> Path:
    tag = f"{threshold:.2f}".replace(".", "")          # 0.10 → "010"
    out_path = out_dir / f"target_binary_t{tag}.tif"

    with rasterio.open(src_path) as src:
        data = src.read(1).astype("float32")
        profile = src.profile.copy()
        src_nodata = src.nodata

    valid = np.isfinite(data)
    if src_nodata is not None:
        valid &= data != src_nodata

    label = np.full(data.shape, NODATA, dtype="uint8")
    label[valid] = (data[valid] >= threshold).astype("uint8")

    profile.update(dtype="uint8", count=1, nodata=NODATA)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(label, 1)

    n_pos = int((label == 1).sum())
    n_valid = int(valid.sum())
    pct = 100 * n_pos / n_valid if n_valid > 0 else 0
    print(f"threshold : {threshold}")
    print(f"positives : {n_pos:,} / {n_valid:,} valid pixels  ({pct:.1f}%)")
    print(f"saved     : {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold src_hi → binary label")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="src_hi >= threshold → source zone (default: 0.1)")
    parser.add_argument("--src", type=Path, required=True,
                        help="Path to continuous src_hi raster")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: same directory as --src)")
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Source raster not found: {args.src}")

    out_dir = args.out_dir if args.out_dir is not None else args.src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    make_binary_label(args.src, out_dir, args.threshold)


if __name__ == "__main__":
    main()
