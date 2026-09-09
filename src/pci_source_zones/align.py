"""Config-driven alignment of predictor and target rasters to a reference grid.

Replaces the hardcoded layer list in `scripts/10_align_features.py`, which named
Montecito-specific files, PRISM dates and folder layout directly in the source.
That single file previously blocked every new site.

An alignment config declares a reference grid, an optional AOI clip, and a list
of layers. Nothing in this module knows any site name.

Config shape:

    reference: /path/to/reference.tif      # grid definition
    aoi: /path/to/aoi.shp                  # optional; outside -> nodata
    output_dir: /path/to/aligned
    nodata: -9999.0
    layers:
      - name: slope                        # output becomes slope_aligned.tif
        path: /path/to/slope.tif
        resampling: bilinear               # bilinear | nearest | cubic | average
      - name: burn_severity
        path: /path/to/severity.tif
        resampling: nearest                # categorical
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .geometry import rasterize_aoi

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "min": Resampling.min,
    "max": Resampling.max,
}

CATEGORICAL_HINT = ("severity", "class", "mask", "landcover", "lithology", "texture")


class AlignError(ValueError):
    """Alignment configuration or execution problem."""


@dataclass
class Layer:
    name: str
    path: Path
    resampling: Resampling
    required: bool = False

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "Layer":
        for key in ("name", "path"):
            if key not in d:
                raise AlignError(f"layer entry missing '{key}': {d}")
        method = str(d.get("resampling", "bilinear")).lower()
        if method not in RESAMPLING:
            raise AlignError(
                f"layer '{d['name']}': unknown resampling {method!r}; "
                f"valid: {sorted(RESAMPLING)}")
        name = str(d["name"])
        if method != "nearest" and any(h in name.lower() for h in CATEGORICAL_HINT):
            print(f"  WARNING: layer '{name}' looks categorical but uses "
                  f"'{method}' resampling — interpolating class codes is usually wrong")
        return cls(name=name, path=Path(d["path"]).expanduser(),
                   resampling=RESAMPLING[method], required=bool(d.get("required", False)))


@dataclass
class AlignConfig:
    reference: Path
    output_dir: Path
    layers: list[Layer]
    aoi: Path | None = None
    nodata: float = -9999.0
    suffix: str = "_aligned"

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "AlignConfig":
        allowed = {"reference", "output_dir", "layers", "aoi", "nodata", "suffix"}
        extra = set(d) - allowed
        if extra:
            raise AlignError(f"unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")
        for key in ("reference", "output_dir", "layers"):
            if key not in d:
                raise AlignError(f"missing required key '{key}'")
        layers = [Layer.parse(x) for x in d["layers"]]
        names = [x.name for x in layers]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise AlignError(f"duplicate layer name(s): {sorted(dupes)}")
        return cls(
            reference=Path(d["reference"]).expanduser(),
            output_dir=Path(d["output_dir"]).expanduser(),
            layers=layers,
            aoi=Path(d["aoi"]).expanduser() if d.get("aoi") else None,
            nodata=float(d.get("nodata", -9999.0)),
            suffix=str(d.get("suffix", "_aligned")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AlignConfig":
        with Path(path).open() as f:
            return cls.parse(yaml.safe_load(f) or {})


def align_layer(layer: Layer, cfg: AlignConfig, ref: dict[str, Any],
                aoi_mask: np.ndarray | None) -> Path:
    """Reproject one layer onto the reference grid and write it."""
    dst = np.full((ref["height"], ref["width"]), cfg.nodata, dtype="float32")
    with rasterio.open(layer.path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            resampling=layer.resampling,
            src_nodata=src.nodata,
            dst_nodata=cfg.nodata,
        )
    dst[~np.isfinite(dst)] = cfg.nodata
    # guard against a foreign sentinel surviving the warp
    dst[(dst < -9000) & (dst != cfg.nodata)] = cfg.nodata
    if aoi_mask is not None:
        dst[~aoi_mask] = cfg.nodata

    out_path = cfg.output_dir / f"{layer.name}{cfg.suffix}.tif"
    prof = {k: v for k, v in ref.items() if k != "crs_wkt"}
    prof.update(driver="GTiff", dtype="float32", count=1,
                nodata=cfg.nodata, compress="lzw")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as out:
        out.write(dst, 1)
    return out_path


def run_alignment(cfg: AlignConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Align every configured layer to the reference grid."""
    if not cfg.reference.exists():
        raise AlignError(f"reference grid not found: {cfg.reference}")

    with rasterio.open(cfg.reference) as r:
        ref = dict(r.profile)
        ref.update(height=r.height, width=r.width, transform=r.transform, crs=r.crs)
    print(f"reference : {cfg.reference.name}  {ref['height']}x{ref['width']}  "
          f"{ref['crs']}")

    aoi_mask = None
    if cfg.aoi:
        aoi_mask = rasterize_aoi(cfg.aoi, (ref["height"], ref["width"]),
                                 ref["transform"], ref["crs"])
        pct = 100 * aoi_mask.mean()
        print(f"aoi       : {cfg.aoi.name}  {aoi_mask.sum():,} cells ({pct:.1f}%)")

    written, skipped, missing = [], [], []
    for layer in cfg.layers:
        out_path = cfg.output_dir / f"{layer.name}{cfg.suffix}.tif"
        if not layer.path.exists():
            missing.append(layer.name)
            level = "MISSING" if layer.required else "skip"
            print(f"  {level:8s} {layer.name}  (source not found: {layer.path})")
            if layer.required:
                raise AlignError(f"required layer '{layer.name}' not found at {layer.path}")
            continue
        if out_path.exists() and not overwrite:
            skipped.append(layer.name)
            print(f"  {'exists':8s} {layer.name}")
            continue
        align_layer(layer, cfg, ref, aoi_mask)
        written.append(layer.name)
        print(f"  {'aligned':8s} {layer.name}")

    summary = {"reference": str(cfg.reference), "output_dir": str(cfg.output_dir),
               "n_written": len(written), "n_skipped": len(skipped),
               "n_missing": len(missing), "written": written,
               "skipped": skipped, "missing": missing}
    print(f"\naligned {len(written)}, skipped {len(skipped)}, missing {len(missing)}")
    return summary
