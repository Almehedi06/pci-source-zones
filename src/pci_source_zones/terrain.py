from __future__ import annotations

from typing import Any

import numpy as np

from .config import output_path, resolve_path
from .inputs import read_raster, write_raster


def prepare_terrain(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute S, specific catchment area a, and G = a S^alpha from a DEM."""

    try:
        from landlab import RasterModelGrid
        from landlab.components import FlowAccumulator, SinkFillerBarnes
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Landlab is required for terrain preparation. "
            "Use `conda activate ml_debris` or install landlab."
        ) from exc

    dem_path = resolve_path(cfg, cfg["paths"]["dem"])
    dem, profile = read_raster(dem_path)
    if dem.ndim != 2:
        raise ValueError(f"DEM must be single-band raster: {dem_path}")

    transform = profile["transform"]
    dx = abs(float(transform.a))
    dy = abs(float(transform.e))
    if not np.isclose(dx, dy):
        raise ValueError(f"DEM cells must be square for Landlab. Got dx={dx}, dy={dy}.")
    cell_size = dx

    valid = np.isfinite(dem)
    if not np.any(valid):
        raise ValueError(f"DEM has no valid cells: {dem_path}")

    z = np.where(valid, dem, np.nanmin(dem[valid]))
    grid = RasterModelGrid(dem.shape, xy_spacing=cell_size)
    grid.add_field("topographic__elevation", z.ravel(), at="node", clobber=True)
    if not np.all(valid):
        grid.status_at_node[~valid.ravel()] = grid.BC_NODE_IS_CLOSED

    terrain_cfg = cfg.get("terrain", {})
    if bool(terrain_cfg.get("fill_sinks", True)):
        SinkFillerBarnes(grid, method="Steepest").run_one_step()

    flow_director = str(terrain_cfg.get("flow_director", "D8")).upper()
    if flow_director == "D8":
        director = "FlowDirectorD8"
    elif flow_director in {"D4", "STEEPEST"}:
        director = "FlowDirectorSteepest"
    else:
        director = flow_director

    FlowAccumulator(grid, flow_director=director).run_one_step()

    slope = np.asarray(grid.at_node["topographic__steepest_slope"]).reshape(dem.shape)
    slope = np.where(valid, np.maximum(slope, 0.0), np.nan)

    drainage_area = np.asarray(grid.at_node["drainage_area"]).reshape(dem.shape)
    specific_area = np.where(valid, drainage_area / cell_size, np.nan)

    alpha = float(cfg.get("parameters", {}).get("alpha", 1.167))
    driving_index = np.where(valid, specific_area * np.power(slope, alpha), np.nan)

    slope_path = output_path(cfg, "slope", "slope.tif")
    area_path = output_path(cfg, "specific_catchment_area", "specific_catchment_area.tif")
    g_path = output_path(cfg, "topographic_driving_index", "topographic_driving_index.tif")

    write_raster(slope_path, slope, profile)
    write_raster(area_path, specific_area, profile)
    write_raster(g_path, driving_index, profile)

    return {
        "slope": slope_path,
        "specific_catchment_area": area_path,
        "topographic_driving_index": g_path,
    }
