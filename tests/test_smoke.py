from pathlib import Path

import importlib.util

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from pci_source_zones.pipeline import run_all


@pytest.mark.skipif(importlib.util.find_spec("landlab") is None, reason="landlab not installed")
def test_full_pipeline_smoke(tmp_path: Path):
    dem_path = tmp_path / "data" / "raw" / "dem.tif"
    dem_path.parent.mkdir(parents=True)
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 8, 10, 10),
        "nodata": -9999.0,
    }
    y, x = np.mgrid[0:8, 0:8]
    dem = (100 - y * 2 - x * 0.2).astype("float32")
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(dem, 1)

    cfg = {
        "paths": {
            "dem": str(dem_path),
            "processed_dir": "data/processed",
            "output_dir": "data/outputs",
        },
        "inputs": {
            "d50": {"mode": "constant", "value": 2.0, "units": "mm"},
            "na": {"mode": "constant", "value": 0.03, "units": "manning"},
            "rainfall_excess": {"mode": "constant", "value": 45.0, "units": "mm/hr"},
        },
        "monte_carlo": {"n_samples": 5, "seed": 1},
        "outputs": {"pci_threshold": 0.5, "min_area_m2": 0.0},
    }
    config_path = tmp_path / "config" / "test.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = run_all(config_path)
    assert result["rasters"]["pci"].exists()
    assert (tmp_path / "data" / "outputs" / "source_polygons.gpkg").exists()
