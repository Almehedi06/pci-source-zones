from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from pci_source_zones.config import load_config
from pci_source_zones.source_area import (
    build_c_sample_table,
    erosion_mask_from_dem_diff,
    pci_from_c_samples,
    run_source_area_workflow,
    source_mask_from_c_band,
    source_mask_from_top_percent,
    source_polygons_from_binary,
)


def test_pci_from_c_samples():
    g = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 10.0]])
    c = np.array([2.0, 4.0, 6.0, 8.0])
    pci = pci_from_c_samples(g, c)
    assert np.allclose(pci[np.isfinite(pci)], [0.0, 0.25, 0.25, 0.5, 1.0])
    assert np.isnan(pci[1, 0])


def test_source_mask_from_top_percent_uses_erosion_mask():
    pci = np.array([[0.1, 0.2], [0.7, 0.9]])
    erosion = np.array([[True, True], [True, False]])
    source, cutoff = source_mask_from_top_percent(pci, 50, erosion)
    assert np.isclose(cutoff, 0.45)
    assert source.tolist() == [[0, 0], [1, 0]]


def test_source_mask_from_c_band_uses_cmax_and_erosion_mask():
    g = np.array([[5.0, 10.0, 20.0], [30.0, 120.0, np.nan]])
    erosion = np.array([[True, False, True], [True, True, True]])
    source = source_mask_from_c_band(g, c_min=10, c_max=120, erosion_mask=erosion)
    assert source.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_erosion_mask_from_dem_diff():
    dem_diff = np.array([[-0.3, -0.1], [np.nan, -0.5]])
    mask = erosion_mask_from_dem_diff(dem_diff, -0.277, reference_shape=(2, 2))
    assert mask.tolist() == [[True, False], [False, True]]


def test_source_polygons_from_binary_small_raster():
    source = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype="uint8")
    profile = {
        "height": 3,
        "width": 3,
        "transform": from_origin(0, 3, 10, 10),
        "crs": "EPSG:32611",
    }
    gdf = source_polygons_from_binary(source, profile, min_area_m2=0)
    assert len(gdf) == 1
    assert int(gdf.iloc[0]["source_id"]) == 1
    assert np.isclose(gdf.iloc[0]["area_m2"], 300.0)


def test_c_sampling_is_deterministic():
    cfg = {
        "_repo_root": ".",
        "parameters": {},
        "monte_carlo": {"n_samples": 3, "seed": 5},
        "c_sampling": {
            "d50": {
                "mode": "distribution",
                "distribution": "lognormal",
                "mean": 2.0,
                "std": 0.48,
                "units": "mm",
            },
            "na": {
                "mode": "distribution",
                "distribution": "uniform",
                "min": 0.015,
                "max": 0.10,
            },
            "rainfall_excess": {
                "mode": "distribution",
                "distribution": "uniform",
                "min": 15.0,
                "max": 55.0,
                "units": "mm/hr",
            },
        },
    }
    first = build_c_sample_table(cfg)
    second = build_c_sample_table(cfg)
    assert np.allclose(first["C"], second["C"])


def test_source_area_workflow_smoke(tmp_path: Path):
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 40, 10, 10),
        "nodata": -9999.0,
    }

    g_path = tmp_path / "g.tif"
    dem_diff_path = tmp_path / "dem_diff.tif"
    g = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 20, 30],
            [40, 50, 60, 70],
        ],
        dtype="float32",
    )
    dem_diff = np.full((4, 4), -0.5, dtype="float32")
    with rasterio.open(g_path, "w", **profile) as dst:
        dst.write(g, 1)
    with rasterio.open(dem_diff_path, "w", **profile) as dst:
        dst.write(dem_diff, 1)

    config = {
        "paths": {
            "dem": str(g_path),
            "dem_diff": str(dem_diff_path),
            "processed_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        },
        "outputs": {
            "topographic_driving_index": str(g_path),
        },
        "monte_carlo": {"n_samples": 5, "seed": 1},
        "c_sampling": {
            "d50": {"mode": "constant", "value": 2.0, "units": "mm"},
            "na": {"mode": "constant", "value": 0.03},
            "rainfall_excess": {"mode": "constant", "value": 45.0, "units": "mm/hr"},
        },
        "source_area": {
            "output_subdir": "source_area_workflow",
            "erosion_threshold_m": -0.277,
            "top_percent_values": [25],
            "c_min_values": [10, 20],
            "c_channel_max": 50,
            "min_polygon_area_m2": 0.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_source_area_workflow(load_config(config_path))
    assert result["pci"].exists()
    assert result["source_cell_summary"].exists()
    assert result["source_polygon_summary"].exists()
    assert result["source_binaries"]["25"].exists()
    assert result["c_band_cell_summary"].exists()
    assert result["c_band_polygon_summary"].exists()
    assert result["c_band_binaries"]["10"].exists()
    assert result["c_band_polygons"]["10"].exists()
