"""Vector geometry helpers: AOI rasterisation, CRS handling, ID discovery.

Kept separate from `raster` so that raster work has no vector dependency.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def rasterize_aoi(aoi_path: str | Path, shape: tuple[int, int], transform,
                  dst_crs: Any = None) -> np.ndarray:
    """Rasterise an AOI polygon onto a grid.

    Reprojects the geometry to `dst_crs` when the AOI is in a different CRS —
    silently rasterising an unprojected geometry produces an empty or wrong mask,
    which is a failure mode we have hit before.

    Returns:
        bool array, True inside the AOI.
    """
    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.features import rasterize

    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError(f"no geometries found in {aoi_path}")

    if dst_crs is not None and gdf.crs is not None:
        target = CRS.from_user_input(dst_crs)
        if CRS.from_user_input(gdf.crs) != target:
            gdf = gdf.to_crs(target.to_wkt())

    geoms = [g.__geo_interface__ for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError(f"no usable geometries in {aoi_path}")

    mask = rasterize(geoms, out_shape=shape, transform=transform,
                     fill=0, default_value=1, dtype="uint8")
    out = mask.astype(bool)
    if not out.any():
        raise ValueError(
            f"AOI {Path(aoi_path).name} rasterised to an empty mask — "
            "check that the AOI overlaps the raster grid and that CRSs agree"
        )
    return out


def natural_key(text: str) -> list:
    """Sort key that orders '2' before '10' in otherwise-alphabetical IDs."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


def sort_ids(ids: Iterable[str]) -> list[str]:
    """Natural-sorted list of analysis-area IDs."""
    return sorted(ids, key=natural_key)


def parse_ids(spec: str | None) -> list[str] | None:
    """Parse a CLI `--ids` specification.

    Accepts comma-separated IDs and inclusive numeric ranges:
        "1,2,5"    -> ["1", "2", "5"]
        "1-4"      -> ["1", "2", "3", "4"]
        "1-3,7"    -> ["1", "2", "3", "7"]
        None or "" -> None  (meaning: all discovered IDs)
    """
    if not spec:
        return None
    out: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and all(p.strip().isdigit() for p in part.split("-", 1)):
            lo, hi = (int(p) for p in part.split("-", 1))
            out.extend(str(i) for i in range(lo, hi + 1))
        else:
            out.append(part)
    return sort_ids(dict.fromkeys(out))
