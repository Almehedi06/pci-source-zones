from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pci_source_zones.config import resolve_path

from .dataset import MLData


@dataclass
class CVPlan:
    splitter: Any
    groups: np.ndarray | None
    method: str
    n_splits: int
    summary: dict[str, Any]


def build_cv_plan(cfg: dict[str, Any], data: MLData, rows: np.ndarray) -> CVPlan:
    """Build a cross-validation splitter for rows from the training split."""

    cv_cfg = cfg.get("ml", {}).get("tuning", {}).get("cv", {})
    method = str(cv_cfg.get("method", "spatial_blocks")).lower()
    y = data.y[rows]
    seed = int(cv_cfg.get("seed", cfg.get("ml", {}).get("split", {}).get("seed", 42)))

    if method == "random_kfold":
        from sklearn.model_selection import KFold

        n_splits = _safe_n_splits(cv_cfg, y)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return CVPlan(splitter, None, method, n_splits, {"method": method})

    if method == "stratified_kfold":
        from sklearn.model_selection import StratifiedKFold

        n_splits = _safe_n_splits(cv_cfg, y)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return CVPlan(splitter, None, method, n_splits, {"method": method})

    if method == "spatial_blocks":
        groups = _spatial_block_groups(cfg, data, rows, cv_cfg)
        splitter, n_splits = _group_splitter(cv_cfg, y, groups, seed)
        return CVPlan(
            splitter,
            groups,
            method,
            n_splits,
            {
                "method": method,
                "n_groups": int(np.unique(groups).size),
                "block_size_m": float(cv_cfg.get("block_size_m", 250.0)),
            },
        )

    if method == "polygon_groups":
        groups = _polygon_groups(cfg, data, rows, cv_cfg)
        splitter, n_splits = _group_splitter(cv_cfg, y, groups, seed)
        return CVPlan(
            splitter,
            groups,
            method,
            n_splits,
            {"method": method, "n_groups": int(np.unique(groups).size)},
        )

    raise ValueError(f"Unsupported ml.tuning.cv.method: {method!r}")


def _safe_n_splits(cv_cfg: dict[str, Any], y: np.ndarray) -> int:
    requested = int(cv_cfg.get("n_splits", 5))
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        raise ValueError("CV requires both positive and negative training labels.")
    n_splits = min(requested, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Not enough cells per class for cross-validation.")
    return n_splits


def _group_splitter(
    cv_cfg: dict[str, Any],
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[Any, int]:
    unique_groups = np.unique(groups)
    requested = int(cv_cfg.get("n_splits", 5))
    n_splits = min(requested, int(unique_groups.size))
    if n_splits < 2:
        raise ValueError("Group CV needs at least two spatial/polygon groups.")

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        return (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            n_splits,
        )
    except ImportError:
        from sklearn.model_selection import GroupKFold

        return GroupKFold(n_splits=n_splits), n_splits


def _spatial_block_groups(
    cfg: dict[str, Any],
    data: MLData,
    rows: np.ndarray,
    cv_cfg: dict[str, Any],
) -> np.ndarray:
    shape = data.target_data.target.shape
    flat = data.flat_indices[rows]
    raster_rows, raster_cols = np.unravel_index(flat, shape)

    transform = data.target_data.profile["transform"]
    cell_size = abs(float(transform.a))
    block_size_m = float(cv_cfg.get("block_size_m", 250.0))
    block_cells = max(1, int(np.ceil(block_size_m / cell_size)))

    n_col_blocks = int(np.ceil(shape[1] / block_cells))
    block_row = raster_rows // block_cells
    block_col = raster_cols // block_cells
    return (block_row * n_col_blocks + block_col).astype("int64")


def _polygon_groups(
    cfg: dict[str, Any],
    data: MLData,
    rows: np.ndarray,
    cv_cfg: dict[str, Any],
) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize

    poly_cfg = cv_cfg.get("polygons", cfg.get("ml", {}).get("split", {}).get("polygons", {}))
    if "path" not in poly_cfg:
        raise ValueError("polygon_groups CV requires ml.tuning.cv.polygons.path or ml.split.polygons.path.")

    path = resolve_path(cfg, poly_cfg["path"])
    id_field = str(poly_cfg.get("id_field", "poly_id"))
    gdf = gpd.read_file(path)
    if id_field not in gdf.columns:
        raise ValueError(f"Polygon field {id_field!r} not found in {path}.")
    if gdf.crs is not None and data.target_data.profile.get("crs") is not None:
        gdf = gdf.to_crs(data.target_data.profile["crs"])

    labels = {value: i + 1 for i, value in enumerate(sorted(gdf[id_field].dropna().unique()))}
    shapes = [
        (geom, labels[value])
        for value, geom in zip(gdf[id_field], gdf.geometry)
        if value in labels and geom is not None and not geom.is_empty
    ]
    group_raster = rasterize(
        shapes,
        out_shape=data.target_data.target.shape,
        transform=data.target_data.profile["transform"],
        fill=0,
        dtype="int32",
    )

    groups = group_raster.ravel()[data.flat_indices[rows]]
    if np.any(groups == 0):
        raise ValueError("Some training cells do not overlap polygon group labels.")
    return groups.astype("int64")
