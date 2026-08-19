from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pci_source_zones.ml.data_contract import validate_data_contract


def _write_raster(
    path: Path,
    shape: tuple[int, int] = (5, 5),
    transform=None,
    crs: str = "EPSG:32611",
) -> None:
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform or from_origin(0, 50, 10, 10),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones(shape, dtype="float32"), 1)


def test_validate_data_contract_passes_when_aligned(tmp_path: Path):
    slope, g = tmp_path / "slope.tif", tmp_path / "g.tif"
    _write_raster(slope)
    _write_raster(g)

    cfg = {"ml": {"base_rasters": {"slope": str(slope), "g": str(g)}}}
    report = validate_data_contract(cfg)
    assert len(report.rasters) == 2


def test_validate_data_contract_raises_on_shape_mismatch(tmp_path: Path):
    slope, g = tmp_path / "slope.tif", tmp_path / "g.tif"
    _write_raster(slope, shape=(5, 5))
    _write_raster(g, shape=(6, 6))

    cfg = {"ml": {"base_rasters": {"slope": str(slope), "g": str(g)}}}
    with pytest.raises(ValueError, match="shape"):
        validate_data_contract(cfg)


def test_validate_data_contract_raises_on_missing_file(tmp_path: Path):
    slope = tmp_path / "slope.tif"
    _write_raster(slope)

    cfg = {
        "ml": {
            "base_rasters": {
                "slope": str(slope),
                "g": str(tmp_path / "missing.tif"),
            }
        }
    }
    with pytest.raises(ValueError, match="not found"):
        validate_data_contract(cfg)


def test_validate_data_contract_raises_on_transform_mismatch(tmp_path: Path):
    slope, g = tmp_path / "slope.tif", tmp_path / "g.tif"
    _write_raster(slope, transform=from_origin(0, 50, 10, 10))
    _write_raster(g, transform=from_origin(5, 50, 10, 10))  # same shape, shifted origin

    cfg = {"ml": {"base_rasters": {"slope": str(slope), "g": str(g)}}}
    with pytest.raises(ValueError, match="transform mismatch"):
        validate_data_contract(cfg)


def test_validate_data_contract_raises_on_crs_mismatch(tmp_path: Path):
    slope, g = tmp_path / "slope.tif", tmp_path / "g.tif"
    _write_raster(slope, crs="EPSG:32611")
    _write_raster(g, crs="EPSG:26911")

    cfg = {"ml": {"base_rasters": {"slope": str(slope), "g": str(g)}}}
    with pytest.raises(ValueError, match="CRS"):
        validate_data_contract(cfg)


def test_validate_data_contract_requires_at_least_one_raster():
    with pytest.raises(ValueError, match="nothing to validate"):
        validate_data_contract({"ml": {}})
