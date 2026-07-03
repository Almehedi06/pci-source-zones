from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask

from pci_source_zones.config import resolve_path
from pci_source_zones.inputs import read_raster


def make_splits(
    cfg: dict[str, Any],
    y_full: np.ndarray,
    valid_mask: np.ndarray,
    profile: dict[str, Any],
) -> dict[str, np.ndarray]:
    split_cfg = cfg.get("ml", {}).get("split", {})
    method = str(split_cfg.get("method", "random")).lower()

    if method == "random":
        splits = _random_splits(y_full, valid_mask, split_cfg)
    elif method == "polygons":
        splits = _polygon_splits(cfg, y_full, valid_mask, profile, split_cfg)
    elif method == "raster":
        splits = _raster_splits(cfg, valid_mask, split_cfg)
    else:
        raise ValueError(f"Unsupported ml.split.method: {method!r}")

    ratio = split_cfg.get("negative_to_positive_ratio", None)
    if ratio is not None and "train" in splits:
        splits["train"] = _balance_training_indices(
            splits["train"],
            y_full.ravel(),
            float(ratio),
            int(split_cfg.get("seed", 42)),
        )
    return splits


def _random_splits(
    y_full: np.ndarray,
    valid_mask: np.ndarray,
    split_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.flatnonzero(valid_mask.ravel())
    y = y_full.ravel()[idx]
    seed = int(split_cfg.get("seed", 42))
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.0))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y if _can_stratify(y) else None,
    )

    splits = {"train": train_idx, "test": test_idx}
    if val_size > 0:
        y_train = y_full.ravel()[train_idx]
        rel_val = val_size / max(1e-9, 1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=rel_val,
            random_state=seed,
            stratify=y_train if _can_stratify(y_train) else None,
        )
        splits["train"] = train_idx
        splits["val"] = val_idx
    return splits


def _polygon_splits(
    cfg: dict[str, Any],
    y_full: np.ndarray,
    valid_mask: np.ndarray,
    profile: dict[str, Any],
    split_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    poly_cfg = split_cfg.get("polygons", {})
    if "path" in poly_cfg:
        return _polygon_id_splits(cfg, y_full, valid_mask, profile, split_cfg, poly_cfg)

    splits: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        paths = poly_cfg.get(split_name, [])
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if not paths:
            continue
        mask = _mask_from_polygons(cfg, paths, profile, valid_mask.shape)
        splits[split_name] = np.flatnonzero((mask & valid_mask).ravel())

    if "train" not in splits:
        raise ValueError("Polygon split requires ml.split.polygons.train.")
    if "test" not in splits:
        raise ValueError("Polygon split requires ml.split.polygons.test.")

    if "val" not in splits and float(split_cfg.get("val_size", 0.0)) > 0:
        from sklearn.model_selection import train_test_split

        seed = int(split_cfg.get("seed", 42))
        y_train = y_full.ravel()[splits["train"]]
        train_idx, val_idx = train_test_split(
            splits["train"],
            test_size=float(split_cfg["val_size"]),
            random_state=seed,
            stratify=y_train if _can_stratify(y_train) else None,
        )
        splits["train"] = train_idx
        splits["val"] = val_idx
    return splits


def _polygon_id_splits(
    cfg: dict[str, Any],
    y_full: np.ndarray,
    valid_mask: np.ndarray,
    profile: dict[str, Any],
    split_cfg: dict[str, Any],
    poly_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    path = resolve_path(cfg, poly_cfg["path"])
    id_field = str(poly_cfg.get("id_field", "poly_id"))
    gdf = gpd.read_file(path)
    if id_field not in gdf.columns:
        raise ValueError(f"Polygon field {id_field!r} not found in {path}.")
    if gdf.crs is not None and profile.get("crs") is not None:
        gdf = gdf.to_crs(profile["crs"])

    splits: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        ids = _split_ids(poly_cfg, split_name)
        if ids is None:
            continue
        chosen = gdf[gdf[id_field].isin(ids)]
        if chosen.empty:
            raise ValueError(f"No {split_name} polygons found for {id_field} in {ids}.")
        mask = geometry_mask(
            [geom for geom in chosen.geometry if geom is not None and not geom.is_empty],
            out_shape=valid_mask.shape,
            transform=profile["transform"],
            invert=True,
        )
        splits[split_name] = np.flatnonzero((mask & valid_mask).ravel())

    if "train" not in splits:
        raise ValueError("Polygon split requires ml.split.polygons.train_ids.")
    if "test" not in splits:
        raise ValueError("Polygon split requires ml.split.polygons.test_ids.")

    if "val" not in splits and float(split_cfg.get("val_size", 0.0)) > 0:
        from sklearn.model_selection import train_test_split

        seed = int(split_cfg.get("seed", 42))
        y_train = y_full.ravel()[splits["train"]]
        train_idx, val_idx = train_test_split(
            splits["train"],
            test_size=float(split_cfg["val_size"]),
            random_state=seed,
            stratify=y_train if _can_stratify(y_train) else None,
        )
        splits["train"] = train_idx
        splits["val"] = val_idx
    return splits


def _split_ids(poly_cfg: dict[str, Any], split_name: str) -> list[Any] | None:
    ids = poly_cfg.get(f"{split_name}_ids", None)
    if ids is None:
        ids = poly_cfg.get(split_name, None)
    if ids is None:
        return None
    if not isinstance(ids, list):
        ids = [ids]
    return ids


def _raster_splits(
    cfg: dict[str, Any],
    valid_mask: np.ndarray,
    split_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    raster_cfg = split_cfg.get("raster", {})
    split_arr, _ = read_raster(resolve_path(cfg, raster_cfg["path"]))
    if split_arr.shape != valid_mask.shape:
        raise ValueError(f"Split raster shape {split_arr.shape} does not match {valid_mask.shape}.")
    values = {
        "train": int(raster_cfg.get("train_value", 1)),
        "val": int(raster_cfg.get("val_value", 2)),
        "test": int(raster_cfg.get("test_value", 3)),
    }
    return {
        name: np.flatnonzero((valid_mask & (split_arr == value)).ravel())
        for name, value in values.items()
        if np.any(valid_mask & (split_arr == value))
    }


def _mask_from_polygons(
    cfg: dict[str, Any],
    paths: list[str | Path],
    profile: dict[str, Any],
    shape: tuple[int, int],
) -> np.ndarray:
    frames = []
    for path in paths:
        gdf = gpd.read_file(resolve_path(cfg, path))
        if gdf.crs is not None and profile.get("crs") is not None:
            gdf = gdf.to_crs(profile["crs"])
        frames.append(gdf)
    geom = [geom for gdf in frames for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return geometry_mask(geom, out_shape=shape, transform=profile["transform"], invert=True)


def _balance_training_indices(
    idx: np.ndarray,
    y_flat: np.ndarray,
    negative_to_positive_ratio: float,
    seed: int,
) -> np.ndarray:
    pos = idx[y_flat[idx] == 1]
    neg = idx[y_flat[idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    rng = np.random.default_rng(seed)
    n_neg = min(len(neg), int(np.ceil(len(pos) * negative_to_positive_ratio)))
    neg_keep = rng.choice(neg, size=n_neg, replace=False)
    out = np.concatenate([pos, neg_keep])
    rng.shuffle(out)
    return out


def _can_stratify(y: np.ndarray) -> bool:
    if y.size < 4:
        return False
    _, counts = np.unique(y, return_counts=True)
    return counts.size == 2 and np.all(counts >= 2)
