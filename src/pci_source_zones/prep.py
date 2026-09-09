"""Put delivered rasters onto one analysis-ready grid.

Survey products arrive in whatever shape the vendor exported them: three
different grids, non-square pixels, several undeclared fill values. The pipeline
requires one square grid with real nodata, so something has to bridge the two.
That bridge is this module rather than a hand-written script per fire, because
the same two mistakes are easy to make every time and silent when made:

**Mask before resampling.** Warping first lets an interpolating resampler blend
a fill value into real data. A -999 cell beside a 1000 m ridge produces
plausible elevations like -654 that no later threshold can find. Masking first
makes the warp skip those cells entirely.

**A DEM and a difference are not the same kind of raster.** For elevation,
values at or below the ground datum are fill. For a difference, negatives are
the erosion signal — applying a `valid_min` there deletes exactly the cells the
analysis is looking for and leaves a dDEM that is entirely positive. `valid_min`
is therefore rejected outright on `kind: diff`.

Config shape:

    target:
      crs: EPSG:6806          # optional; taken from the first raster if absent
      res: 2.0                # square pixels, always
      extent: intersection    # intersection | union | [xmin, ymin, xmax, ymax]
      nodata: -9999.0
    output_dir: /path/to/prep
    rasters:
      - {name: dem_pre,  path: 2018.tif,  kind: dem,  fill: [0, -999], valid_min: 0}
      - {name: dem_post, path: 2021.tif,  kind: dem,  fill: [0, -999], valid_min: 0}
      - {name: dz,       path: 21_18.tif, kind: diff, fill: [0, -999]}
"""
from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

KINDS = ("dem", "diff")
EXTENT_MODES = ("intersection", "union")
RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


class PrepError(ValueError):
    """Preparation configuration or execution problem."""


# GTiff tiling requires block dimensions that are multiples of 16, which a small
# raster cannot satisfy. Tiling only helps on large grids anyway.
TILE_MIN = 256


def _storage_opts(width: int, height: int) -> dict[str, Any]:
    if width >= TILE_MIN and height >= TILE_MIN:
        return {"tiled": True, "blockxsize": 256, "blockysize": 256,
                "BIGTIFF": "YES"}
    return {"tiled": False}


# ────────────────────────────────────────────────────────── config

@dataclass
class RasterSpec:
    """One input raster and how its fill values should be read."""
    name: str
    path: Path
    kind: str = "dem"
    fill: list[float] = field(default_factory=list)   # exact values that mean "no data"
    valid_min: float | None = None                    # dem only: <= this is fill
    resampling: str = "bilinear"

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "RasterSpec":
        allowed = set(cls.__dataclass_fields__)
        extra = set(d) - allowed
        if extra:
            raise PrepError(f"raster entry: unknown key(s) {sorted(extra)}; "
                            f"allowed {sorted(allowed)}")
        for key in ("name", "path"):
            if key not in d:
                raise PrepError(f"raster entry missing '{key}': {d}")
        out = cls(name=str(d["name"]), path=Path(d["path"]).expanduser(),
                  kind=str(d.get("kind", "dem")).lower(),
                  fill=[float(x) for x in d.get("fill", [])],
                  valid_min=(None if d.get("valid_min") is None
                             else float(d["valid_min"])),
                  resampling=str(d.get("resampling", "bilinear")).lower())
        if out.kind not in KINDS:
            raise PrepError(f"raster '{out.name}': kind must be one of "
                            f"{list(KINDS)}, got {out.kind!r}")
        if out.resampling not in RESAMPLING:
            raise PrepError(f"raster '{out.name}': unknown resampling "
                            f"{out.resampling!r}; valid {sorted(RESAMPLING)}")
        if out.kind == "diff" and out.valid_min is not None:
            raise PrepError(
                f"raster '{out.name}': valid_min is not allowed on kind 'diff'. "
                "Negative values in a difference raster are the erosion signal; "
                "a lower bound would delete them. List exact fill values under "
                "'fill' instead.")
        return out

    def fill_mask(self, arr: np.ndarray, declared_nodata: float | None) -> np.ndarray:
        """True where the cell carries no real measurement."""
        mask = ~np.isfinite(arr)
        if declared_nodata is not None:
            mask |= arr == declared_nodata
        for value in self.fill:
            mask |= arr == value
        if self.valid_min is not None:
            mask |= arr <= self.valid_min
        return mask


@dataclass
class TargetGrid:
    """The one grid every output lands on."""
    res: float
    crs: Any = None
    extent: str | list[float] = "intersection"
    nodata: float = -9999.0

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "TargetGrid":
        allowed = {"res", "crs", "extent", "nodata"}
        extra = set(d) - allowed
        if extra:
            raise PrepError(f"target: unknown key(s) {sorted(extra)}; "
                            f"allowed {sorted(allowed)}")
        if "res" not in d:
            raise PrepError("target: 'res' is required")
        res = float(d["res"])
        if res <= 0:
            raise PrepError(f"target.res must be > 0, got {res}")
        extent = d.get("extent", "intersection")
        if isinstance(extent, str):
            if extent not in EXTENT_MODES:
                raise PrepError(f"target.extent must be one of {list(EXTENT_MODES)} "
                                f"or [xmin, ymin, xmax, ymax], got {extent!r}")
        else:
            extent = [float(x) for x in extent]
            if len(extent) != 4:
                raise PrepError("target.extent bbox must have 4 values")
            if extent[0] >= extent[2] or extent[1] >= extent[3]:
                raise PrepError(f"target.extent bbox is empty: {extent}")
        return cls(res=res, crs=d.get("crs"), extent=extent,
                   nodata=float(d.get("nodata", -9999.0)))

    def resolve(self, rasters: list[RasterSpec]) -> dict[str, Any]:
        """Compute the concrete grid: crs, transform, width, height."""
        crs = self.crs
        if crs is None:
            with rasterio.open(rasters[0].path) as src:
                crs = src.crs
            if crs is None:
                raise PrepError(
                    f"target.crs is not set and '{rasters[0].name}' has no CRS")
        crs = rasterio.crs.CRS.from_user_input(crs)

        boxes = []
        for spec in rasters:
            with rasterio.open(spec.path) as src:
                if src.crs is None:
                    raise PrepError(f"raster '{spec.name}' has no CRS")
                boxes.append(transform_bounds(src.crs, crs, *src.bounds))

        if self.extent == "intersection":
            left = max(b[0] for b in boxes); bottom = max(b[1] for b in boxes)
            right = min(b[2] for b in boxes); top = min(b[3] for b in boxes)
            if left >= right or bottom >= top:
                raise PrepError(
                    "the input rasters do not overlap, so 'intersection' is empty; "
                    "check their CRS and extents")
        elif self.extent == "union":
            left = min(b[0] for b in boxes); bottom = min(b[1] for b in boxes)
            right = max(b[2] for b in boxes); top = max(b[3] for b in boxes)
        else:
            left, bottom, right, top = self.extent

        # snap outward to whole multiples of the resolution
        left = np.floor(left / self.res) * self.res
        bottom = np.floor(bottom / self.res) * self.res
        right = np.ceil(right / self.res) * self.res
        top = np.ceil(top / self.res) * self.res

        width = int(round((right - left) / self.res))
        height = int(round((top - bottom) / self.res))
        if width < 1 or height < 1:
            raise PrepError(f"target grid is empty: {width} x {height}")
        return {"crs": crs, "transform": from_origin(left, top, self.res, self.res),
                "width": width, "height": height,
                "bounds": (left, bottom, right, top)}


@dataclass
class PrepConfig:
    target: TargetGrid
    output_dir: Path
    rasters: list[RasterSpec]
    suffix: str = ""

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "PrepConfig":
        allowed = {"target", "output_dir", "rasters", "suffix"}
        extra = set(d) - allowed
        if extra:
            raise PrepError(f"unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")
        for key in ("target", "output_dir", "rasters"):
            if key not in d:
                raise PrepError(f"missing required key '{key}'")
        rasters = [RasterSpec.parse(x) for x in d["rasters"]]
        if not rasters:
            raise PrepError("'rasters' is empty; nothing to prepare")
        names = [r.name for r in rasters]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise PrepError(f"duplicate raster name(s): {sorted(dupes)}")
        return cls(target=TargetGrid.parse(d["target"]),
                   output_dir=Path(d["output_dir"]).expanduser(),
                   rasters=rasters, suffix=str(d.get("suffix", "")))

    @classmethod
    def load(cls, path: str | Path) -> "PrepConfig":
        with Path(path).open() as f:
            return cls.parse(yaml.safe_load(f) or {})


# ────────────────────────────────────────────────────────── fill discovery

def scan_fills(path: str | Path, top: int = 5, max_cells: int = 4_000_000,
               seed: int = 0) -> dict[str, Any]:
    """Report the most repeated values in a raster, to expose undeclared fills.

    A continuous surface repeats no value often, so a value occupying a large
    share of the raster is a fill somebody forgot to declare — and reading it as
    data is how a -999 ends up averaged into a hillslope.

    Samples blocks rather than reading the whole file: these rasters are
    routinely multi-gigabyte, and a scan nobody waits for is a scan nobody runs.
    """
    rng = np.random.default_rng(seed)
    with rasterio.open(path) as src:
        declared = src.nodata
        windows = [w for _, w in src.block_windows(1)]
        if not windows:
            raise PrepError(f"cannot read blocks from {path}")
        per_block = max(1, windows[0].width * windows[0].height)
        n_take = max(1, min(len(windows), int(max_cells // per_block)))
        picks = (windows if n_take >= len(windows)
                 else [windows[i] for i in
                       rng.choice(len(windows), n_take, replace=False)])
        chunks = [src.read(1, window=w).astype("float64").ravel() for w in picks]

    arr = np.concatenate(chunks)
    finite = arr[np.isfinite(arr)]
    counts = Counter(finite.tolist())
    total = arr.size
    common = [{"value": v, "fraction": n / total}
              for v, n in counts.most_common(top) if n / total > 0.001]
    return {"declared_nodata": declared, "n_sampled": int(total),
            "blocks_sampled": len(picks), "blocks_total": len(windows),
            "nonfinite_fraction": float((~np.isfinite(arr)).mean()),
            "most_common": common}


# ────────────────────────────────────────────────────────── execution

def _mask_to_temp(spec: RasterSpec, nodata: float, tmp_dir: Path) -> tuple[Path, int]:
    """Write a copy with every fill value marked as nodata, before any warp."""
    out = tmp_dir / f"{spec.name}_masked.tif"
    with rasterio.open(spec.path) as src:
        profile = src.profile.copy()
        for key in ("photometric", "blockxsize", "blockysize", "tiled",
                    "interleave"):
            profile.pop(key, None)
        profile.update(dtype="float32", nodata=nodata, compress="LZW",
                       **_storage_opts(src.width, src.height))
        n_fill = 0
        with rasterio.open(out, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window).astype("float32")
                mask = spec.fill_mask(arr, src.nodata)
                n_fill += int(mask.sum())
                arr[mask] = nodata
                dst.write(arr, 1, window=window)
    return out, n_fill


def _warp_to_grid(source: Path, spec: RasterSpec, grid: dict[str, Any],
                  nodata: float, out_path: Path) -> None:
    """Reproject onto the target grid, windowed so memory stays flat."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "nodata": nodata,
               "crs": grid["crs"], "transform": grid["transform"],
               "width": grid["width"], "height": grid["height"],
               "compress": "LZW", **_storage_opts(grid["width"], grid["height"])}
    with rasterio.open(source) as src:
        with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"],
                       width=grid["width"], height=grid["height"],
                       resampling=RESAMPLING[spec.resampling],
                       src_nodata=nodata, nodata=nodata) as vrt:
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    dst.write(vrt.read(1, window=window), 1, window=window)


def run_prep(cfg: PrepConfig, *, overwrite: bool = False,
             scan_only: bool = False) -> dict[str, Any]:
    """Mask fills, then put every raster on the one target grid."""
    for spec in cfg.rasters:
        if not spec.path.exists():
            raise PrepError(f"raster '{spec.name}' not found: {spec.path}")

    if scan_only:
        print("scanning for undeclared fill values "
              "(a continuous surface should repeat no value often)\n")
        report = {}
        for spec in cfg.rasters:
            info = scan_fills(spec.path)
            report[spec.name] = info
            print(f"  {spec.name}  [{spec.kind}]  declared nodata: "
                  f"{info['declared_nodata']}")
            print(f"    non-finite: {100 * info['nonfinite_fraction']:.1f}%")
            for entry in info["most_common"]:
                flag = "  <-- likely fill" if entry["fraction"] > 0.02 else ""
                print(f"    value {entry['value']:>12.4f}: "
                      f"{100 * entry['fraction']:5.1f}%{flag}")
            print()
        return {"scan": report}

    grid = cfg.target.resolve(cfg.rasters)
    n_cells = grid["width"] * grid["height"]
    print(f"target grid : {grid['width']} x {grid['height']} "
          f"({n_cells / 1e6:.1f} M cells) @ {cfg.target.res} m  {grid['crs']}")
    print(f"bounds      : {tuple(round(b, 1) for b in grid['bounds'])}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    written, skipped, details = [], [], {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="pci_prep_"))
    try:
        for spec in cfg.rasters:
            out_path = cfg.output_dir / f"{spec.name}{cfg.suffix}.tif"
            if out_path.exists() and not overwrite:
                skipped.append(spec.name)
                print(f"  {'exists':8s} {spec.name}")
                continue
            masked, n_fill = _mask_to_temp(spec, cfg.target.nodata, tmp_dir)
            _warp_to_grid(masked, spec, grid, cfg.target.nodata, out_path)
            masked.unlink(missing_ok=True)
            written.append(spec.name)
            details[spec.name] = {"n_fill_masked": n_fill, "path": str(out_path)}
            print(f"  {'prepared':8s} {spec.name}  ({n_fill:,} fill cells masked)"
                  f"  -> {out_path.name}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = {"output_dir": str(cfg.output_dir), "res": cfg.target.res,
               "crs": str(grid["crs"]), "width": grid["width"],
               "height": grid["height"], "n_cells": n_cells,
               "written": written, "skipped": skipped, "rasters": details}
    print(f"\nprepared {len(written)}, skipped {len(skipped)} "
          f"-> {cfg.output_dir}")
    return summary
