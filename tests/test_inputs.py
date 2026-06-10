from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pci_source_zones.inputs import InputProvider


def test_constant_input_unit_conversion():
    cfg = {"_repo_root": "."}
    provider = InputProvider({"mode": "constant", "value": 2.0, "units": "mm"}, "d50", cfg)
    assert np.isclose(provider.sample(np.random.default_rng(1)), 0.002)


def test_uniform_distribution_is_deterministic_with_seed():
    cfg = {"_repo_root": "."}
    spec = {"mode": "distribution", "distribution": "uniform", "min": 1.0, "max": 2.0}
    p1 = InputProvider(spec, "na", cfg)
    p2 = InputProvider(spec, "na", cfg)
    assert np.isclose(
        p1.sample(np.random.default_rng(7)),
        p2.sample(np.random.default_rng(7)),
    )


def test_raster_input_reads_and_converts(tmp_path: Path):
    raster_path = tmp_path / "d50.tif"
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999.0,
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(np.full((2, 2), 2.0, dtype="float32"), 1)

    cfg = {"_repo_root": str(tmp_path)}
    provider = InputProvider(
        {"mode": "raster", "path": "d50.tif", "units": "mm"},
        "d50",
        cfg,
        reference_shape=(2, 2),
    )
    assert np.allclose(provider.sample(np.random.default_rng(1)), 0.002)
