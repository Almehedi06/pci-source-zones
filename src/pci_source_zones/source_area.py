from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from .c_model import modeled_threshold_c
from .config import output_path, resolve_path
from .inputs import InputProvider, read_raster, write_raster
from .terrain import prepare_terrain


def source_area_output_dir(cfg: dict[str, Any]) -> Path:
    source_cfg = cfg.get("source_area", {})
    subdir = source_cfg.get("output_subdir", "source_area_workflow")
    out_dir = resolve_path(cfg, cfg.get("paths", {}).get("output_dir", "data/outputs"))
    return out_dir / str(subdir)


def build_c_sample_table(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    mc = cfg.get("monte_carlo", {})
    n_samples = int(mc.get("n_samples", 1500))
    seed = int(mc.get("seed", 42))
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    specs = cfg.get("c_sampling", cfg.get("inputs", {}))
    required = {"d50", "na", "rainfall_excess"}
    missing = required.difference(specs)
    if missing:
        raise ValueError(f"Missing C sampling inputs: {sorted(missing)}")

    providers = {
        "d50": InputProvider(specs["d50"], "d50", cfg),
        "na": InputProvider(specs["na"], "na", cfg),
        "rainfall_excess": InputProvider(
            specs["rainfall_excess"],
            "rainfall_excess",
            cfg,
        ),
    }

    rng = np.random.default_rng(seed)
    d50_m = np.zeros(n_samples, dtype="float64")
    na = np.zeros(n_samples, dtype="float64")
    rainfall_m_s = np.zeros(n_samples, dtype="float64")
    c_values = np.zeros(n_samples, dtype="float64")

    for i in range(n_samples):
        d50_i = _as_scalar(providers["d50"].sample(rng), "d50")
        na_i = _as_scalar(providers["na"].sample(rng), "na")
        rainfall_i = _as_scalar(
            providers["rainfall_excess"].sample(rng),
            "rainfall_excess",
        )
        c_i = modeled_threshold_c(d50_i, na_i, rainfall_i, cfg.get("parameters", {}))

        d50_m[i] = d50_i
        na[i] = na_i
        rainfall_m_s[i] = rainfall_i
        c_values[i] = _as_scalar(c_i, "C")

    return {
        "d50_mm": d50_m * 1000.0,
        "na": na,
        "r_mm_hr": rainfall_m_s * 1000.0 * 3600.0,
        "C": c_values,
    }


def pci_from_c_samples(
    driving_index: np.ndarray,
    c_samples: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    g = np.asarray(driving_index, dtype="float64")
    c_sorted = np.sort(np.asarray(c_samples, dtype="float64"))
    c_sorted = c_sorted[np.isfinite(c_sorted)]
    if c_sorted.size == 0:
        raise ValueError("C samples contain no finite values.")

    if valid_mask is None:
        valid_mask = np.isfinite(g)

    pci = np.full(g.shape, np.nan, dtype="float64")
    pci[valid_mask] = np.searchsorted(c_sorted, g[valid_mask], side="right") / float(
        c_sorted.size
    )
    return pci


def erosion_mask_from_dem_diff(
    dem_diff: np.ndarray,
    threshold_m: float,
    reference_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if reference_shape is not None and dem_diff.shape != reference_shape:
        raise ValueError(
            f"DEM-difference raster shape {dem_diff.shape} does not match "
            f"reference shape {reference_shape}."
        )
    return np.isfinite(dem_diff) & (dem_diff < float(threshold_m))


def source_mask_from_top_percent(
    pci: np.ndarray,
    top_percent: float,
    erosion_mask: np.ndarray | None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if not 0 < float(top_percent) <= 100:
        raise ValueError("top_percent must be between 0 and 100.")

    pci_arr = np.asarray(pci, dtype="float64")
    if valid_mask is None:
        valid_mask = np.isfinite(pci_arr)

    values = pci_arr[valid_mask & np.isfinite(pci_arr)]
    if values.size == 0:
        raise ValueError("PCI raster has no finite valid values.")

    cutoff = float(np.nanpercentile(values, 100.0 - float(top_percent)))
    source = (pci_arr >= cutoff) & _optional_mask(erosion_mask, pci_arr.shape) & valid_mask
    return source.astype("uint8"), cutoff


def source_mask_from_c_band(
    driving_index: np.ndarray,
    c_min: float,
    c_max: float | None = None,
    erosion_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    g = np.asarray(driving_index, dtype="float64")
    if valid_mask is None:
        valid_mask = np.isfinite(g)

    source = (g >= float(c_min)) & valid_mask & _optional_mask(erosion_mask, g.shape)
    if c_max is not None:
        source &= g < float(c_max)
    return source.astype("uint8")


def source_polygons_from_binary(
    source: np.ndarray,
    profile: dict[str, Any],
    out_path: str | Path | None = None,
    min_area_m2: float = 0.0,
    top_percent: float | None = None,
    pci_cutoff: float | None = None,
    c_min: float | None = None,
    c_max: float | None = None,
    erosion_threshold_m: float | None = None,
    method: str | None = None,
) -> gpd.GeoDataFrame:
    source_arr = np.asarray(source).astype("uint8")
    geoms = [
        shape(geom)
        for geom, value in shapes(
            source_arr,
            mask=source_arr == 1,
            transform=profile["transform"],
        )
        if int(value) == 1
    ]

    columns = [
        "source_id",
        "mass__wasting_id",
        "top_percent",
        "pci_cutoff",
        "c_min",
        "c_max",
        "dod_thresh",
        "method",
        "area_m2",
        "area_km2",
        "geometry",
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=profile.get("crs"))

    if len(gdf) > 0:
        gdf["area_m2"] = _area_m2(gdf)
        if min_area_m2 > 0:
            gdf = gdf[gdf["area_m2"] >= float(min_area_m2)].copy()
        gdf["area_km2"] = gdf["area_m2"] / 1_000_000.0
    else:
        gdf["area_m2"] = []
        gdf["area_km2"] = []

    gdf.insert(0, "source_id", np.arange(1, len(gdf) + 1, dtype=int))
    gdf.insert(1, "mass__wasting_id", gdf["source_id"].astype(int))
    gdf.insert(2, "top_percent", top_percent)
    gdf.insert(3, "pci_cutoff", pci_cutoff)
    gdf.insert(4, "c_min", c_min)
    gdf.insert(5, "c_max", c_max)
    gdf.insert(6, "dod_thresh", erosion_threshold_m)
    gdf.insert(7, "method", method)
    gdf = gdf[columns]

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        gdf.to_file(path, driver="GPKG")

    return gdf


def write_binary_raster(
    path: str | Path,
    source: np.ndarray,
    profile: dict[str, Any],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = profile.copy()
    meta.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )
    if not meta.get("tiled", False):
        meta.pop("blockxsize", None)
        meta.pop("blockysize", None)
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(np.asarray(source, dtype="uint8"), 1)
    return out


def run_source_area_workflow(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = source_area_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    g_path = output_path(cfg, "topographic_driving_index", "topographic_driving_index.tif")
    if not g_path.exists():
        prepare_terrain(cfg)

    driving_index, profile = read_raster(g_path)
    valid = np.isfinite(driving_index)
    cell_area_m2 = abs(float(profile["transform"].a) * float(profile["transform"].e))

    c_table = build_c_sample_table(cfg)
    c_csv = out_dir / "c_samples.csv"
    write_c_sample_table(c_table, c_csv)

    pci = pci_from_c_samples(driving_index, c_table["C"], valid)
    pci_path = write_raster(out_dir / "pci_from_c_cdf.tif", pci, profile)

    source_cfg = cfg.get("source_area", {})
    threshold_m = float(source_cfg.get("erosion_threshold_m", -0.277))
    use_dod_filter = bool(source_cfg.get("use_dod_filter", True))
    erosion: np.ndarray | None = None
    erosion_path: Path | None = None
    if use_dod_filter:
        dem_diff_path = _dem_diff_path(cfg)
        dem_diff, _dem_diff_profile = read_raster(dem_diff_path)
        erosion = erosion_mask_from_dem_diff(dem_diff, threshold_m, driving_index.shape)
        erosion_path = write_binary_raster(
            out_dir / _erosion_name(threshold_m),
            erosion.astype("uint8"),
            profile,
        )

    top_values = source_cfg.get("top_percent_values", [10, 5])
    c_min_values = source_cfg.get("c_min_values", [])
    c_channel_max = source_cfg.get("c_channel_max", None)
    c_channel_max = None if c_channel_max is None else float(c_channel_max)
    min_area_m2 = float(
        source_cfg.get(
            "min_polygon_area_m2",
            cfg.get("outputs", {}).get("min_area_m2", 0.0),
        )
    )

    cell_rows: list[dict[str, Any]] = []
    polygon_rows: list[dict[str, Any]] = []
    c_band_cell_rows: list[dict[str, Any]] = []
    c_band_polygon_rows: list[dict[str, Any]] = []
    binary_paths: dict[str, Path] = {}
    polygon_paths: dict[str, Path] = {}
    c_band_binary_paths: dict[str, Path] = {}
    c_band_polygon_paths: dict[str, Path] = {}
    filter_label = "dod" if use_dod_filter else "no_dod"

    for top_percent in top_values:
        source, cutoff = source_mask_from_top_percent(
            pci,
            float(top_percent),
            erosion,
            valid,
        )
        label = _top_percent_label(top_percent)
        binary_path = write_binary_raster(
            out_dir / f"{label}_pci_{filter_label}_source_binary.tif",
            source,
            profile,
        )
        polygon_path = out_dir / f"{label}_pci_{filter_label}_source_polygons.gpkg"
        gdf = source_polygons_from_binary(
            source,
            profile,
            out_path=polygon_path,
            min_area_m2=min_area_m2,
            top_percent=float(top_percent),
            pci_cutoff=cutoff,
            erosion_threshold_m=threshold_m if use_dod_filter else None,
            method=f"pci_top_percent_{filter_label}",
        )

        pixels = int(source.sum())
        source_area_m2 = float(pixels * cell_area_m2)
        cell_rows.append(
            {
                "top_percent": float(top_percent),
                "pci_cutoff": cutoff,
                "source_pixels": pixels,
                "source_area_m2": source_area_m2,
                "source_area_km2": source_area_m2 / 1_000_000.0,
                "raster": str(binary_path),
            }
        )
        polygon_rows.append(
            {
                "top_percent": float(top_percent),
                "n_polygons": int(len(gdf)),
                "polygon_area_m2": float(gdf["area_m2"].sum()) if len(gdf) else 0.0,
                "polygon_area_km2": float(gdf["area_km2"].sum()) if len(gdf) else 0.0,
                "polygon_file": str(polygon_path),
            }
        )
        binary_paths[str(top_percent)] = binary_path
        polygon_paths[str(top_percent)] = polygon_path

    for c_min in c_min_values:
        c_min_float = float(c_min)
        source = source_mask_from_c_band(
            driving_index,
            c_min_float,
            c_max=c_channel_max,
            erosion_mask=erosion,
            valid_mask=valid,
        )
        label = _c_band_label(c_min_float, c_channel_max)
        binary_path = write_binary_raster(
            out_dir / f"{label}_{filter_label}_source_binary.tif",
            source,
            profile,
        )
        polygon_path = out_dir / f"{label}_{filter_label}_source_polygons.gpkg"
        gdf = source_polygons_from_binary(
            source,
            profile,
            out_path=polygon_path,
            min_area_m2=min_area_m2,
            c_min=c_min_float,
            c_max=c_channel_max,
            erosion_threshold_m=threshold_m if use_dod_filter else None,
            method=f"c_band_{filter_label}",
        )

        pixels = int(source.sum())
        source_area_m2 = float(pixels * cell_area_m2)
        c_band_cell_rows.append(
            {
                "c_min": c_min_float,
                "c_max": c_channel_max,
                "dod_filter": use_dod_filter,
                "dod_threshold_m": threshold_m if use_dod_filter else None,
                "source_pixels": pixels,
                "source_area_m2": source_area_m2,
                "source_area_km2": source_area_m2 / 1_000_000.0,
                "raster": str(binary_path),
            }
        )
        c_band_polygon_rows.append(
            {
                "c_min": c_min_float,
                "c_max": c_channel_max,
                "dod_filter": use_dod_filter,
                "dod_threshold_m": threshold_m if use_dod_filter else None,
                "n_polygons": int(len(gdf)),
                "polygon_area_m2": float(gdf["area_m2"].sum()) if len(gdf) else 0.0,
                "polygon_area_km2": float(gdf["area_km2"].sum()) if len(gdf) else 0.0,
                "polygon_file": str(polygon_path),
            }
        )
        c_band_binary_paths[str(c_min)] = binary_path
        c_band_polygon_paths[str(c_min)] = polygon_path

    cell_summary = out_dir / "source_cell_summary.csv"
    polygon_summary = out_dir / "source_polygon_summary.csv"
    c_band_cell_summary = out_dir / "c_band_source_cell_summary.csv"
    c_band_polygon_summary = out_dir / "c_band_source_polygon_summary.csv"
    write_rows_csv(cell_rows, cell_summary)
    write_rows_csv(polygon_rows, polygon_summary)
    write_rows_csv(c_band_cell_rows, c_band_cell_summary)
    write_rows_csv(c_band_polygon_rows, c_band_polygon_summary)

    return {
        "output_dir": out_dir,
        "c_samples": c_csv,
        "pci": pci_path,
        "erosion_mask": erosion_path,
        "source_binaries": binary_paths,
        "source_polygons": polygon_paths,
        "source_cell_summary": cell_summary,
        "source_polygon_summary": polygon_summary,
        "c_band_binaries": c_band_binary_paths,
        "c_band_polygons": c_band_polygon_paths,
        "c_band_cell_summary": c_band_cell_summary,
        "c_band_polygon_summary": c_band_polygon_summary,
        "cell_rows": cell_rows,
        "polygon_rows": polygon_rows,
        "c_band_cell_rows": c_band_cell_rows,
        "c_band_polygon_rows": c_band_polygon_rows,
    }


def write_c_sample_table(table: dict[str, np.ndarray], out_path: str | Path) -> Path:
    rows = [
        {
            "d50_mm": table["d50_mm"][i],
            "na": table["na"][i],
            "r_mm_hr": table["r_mm_hr"][i],
            "C": table["C"][i],
        }
        for i in range(len(table["C"]))
    ]
    return write_rows_csv(rows, out_path)


def write_rows_csv(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _as_scalar(value: Any, name: str) -> float:
    arr = np.asarray(value, dtype="float64")
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar for this source-area workflow.")
    return float(arr.reshape(-1)[0])


def _dem_diff_path(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths", {})
    source_cfg = cfg.get("source_area", {})
    if "dem_diff" in paths:
        return resolve_path(cfg, paths["dem_diff"])
    if "dem_diff" in source_cfg:
        return resolve_path(cfg, source_cfg["dem_diff"])
    raise ValueError("Set paths.dem_diff or source_area.dem_diff in the config.")


def _top_percent_label(value: Any) -> str:
    text = f"{float(value):g}".replace(".", "p")
    return f"top{text}"


def _c_band_label(c_min: float, c_max: float | None) -> str:
    min_text = _value_label(c_min)
    if c_max is None:
        return f"cmin{min_text}_g"
    return f"cmin{min_text}_lt_cmax{_value_label(c_max)}_g"


def _value_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus").replace(".", "p")


def _erosion_name(threshold_m: float) -> str:
    text = f"{abs(threshold_m):g}".replace(".", "p")
    sign = "minus" if threshold_m < 0 else "plus"
    return f"erosion_mask_dod_lt_{sign}_{text}m.tif"


def _optional_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"Mask shape {arr.shape} does not match reference shape {shape}.")
    return arr


def _area_m2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    if gdf.crs is not None and gdf.crs.is_geographic:
        projected_crs = gdf.estimate_utm_crs()
        if projected_crs is not None:
            return gdf.to_crs(projected_crs).geometry.area.to_numpy()
    return gdf.geometry.area.to_numpy()
