from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape


def extract_source_polygons(
    pci: np.ndarray,
    profile: dict[str, Any],
    threshold: float = 0.7,
    min_area_m2: float = 0.0,
    out_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    mask = np.isfinite(pci) & (pci >= threshold)
    geoms = [
        shape(geom)
        for geom, value in shapes(mask.astype("uint8"), mask=mask, transform=profile["transform"])
        if int(value) == 1
    ]

    columns = [
        "source_id",
        "mass__wasting_id",
        "area_m2",
        "pci_mean",
        "pci_min",
        "pci_max",
        "geometry",
    ]
    if not geoms:
        gdf = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=profile.get("crs"))
        if out_path is not None:
            _write_gdf(gdf, out_path)
        return gdf

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=profile.get("crs"))
    gdf["area_m2"] = _area_m2(gdf)
    if min_area_m2 > 0:
        gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()

    stats = [_pci_stats_for_geom(pci, profile, geom) for geom in gdf.geometry]
    if stats:
        gdf["pci_mean"] = [s[0] for s in stats]
        gdf["pci_min"] = [s[1] for s in stats]
        gdf["pci_max"] = [s[2] for s in stats]
    else:
        gdf["pci_mean"] = []
        gdf["pci_min"] = []
        gdf["pci_max"] = []

    gdf.insert(0, "source_id", np.arange(1, len(gdf) + 1, dtype=int))
    gdf.insert(1, "mass__wasting_id", gdf["source_id"].astype(int))
    gdf = gdf[columns]

    if out_path is not None:
        _write_gdf(gdf, out_path)
    return gdf


def _area_m2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    if gdf.crs is not None and gdf.crs.is_geographic:
        projected_crs = gdf.estimate_utm_crs()
        if projected_crs is not None:
            return gdf.to_crs(projected_crs).geometry.area.to_numpy()
    return gdf.geometry.area.to_numpy()


def _pci_stats_for_geom(
    pci: np.ndarray, profile: dict[str, Any], geom: Any
) -> tuple[float, float, float]:
    inside = geometry_mask(
        [geom],
        out_shape=pci.shape,
        transform=profile["transform"],
        invert=True,
    )
    values = pci[inside & np.isfinite(pci)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def _write_gdf(gdf: gpd.GeoDataFrame, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")
