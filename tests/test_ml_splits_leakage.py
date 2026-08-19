from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from pci_source_zones.ml.splits import polygon_train_region_mask


def _write_train_test_polygons(path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[
            box(0, 50, 100, 100),  # train: top half (raster rows 0-4)
            box(0, 0, 100, 50),  # test: bottom half (raster rows 5-9)
        ],
        crs="EPSG:32611",
    )
    gdf.to_file(path, driver="GPKG")


def _profile() -> dict:
    return {
        "height": 10,
        "width": 10,
        "transform": from_origin(0, 100, 10, 10),
        "crs": "EPSG:32611",
    }


def test_polygon_train_region_mask_excludes_test_polygon(tmp_path: Path):
    poly_path = tmp_path / "train_test.gpkg"
    _write_train_test_polygons(poly_path)

    split_cfg = {
        "method": "polygons",
        "polygons": {
            "path": str(poly_path),
            "id_field": "id",
            "train_ids": [1],
            "test_ids": [2],
        },
    }

    mask = polygon_train_region_mask({}, _profile(), (10, 10), split_cfg)

    assert mask is not None
    assert mask.shape == (10, 10)
    # Train polygon covers rows 0-4 (top half) — must be entirely True.
    assert mask[0:5, :].all()
    # Test polygon covers rows 5-9 (bottom half) — must be entirely False,
    # i.e. no test-region pixel is ever treated as a valid train pixel.
    assert not mask[5:10, :].any()


def test_polygon_train_region_mask_none_for_random_split():
    # "random" scatters train/test pixels with no spatial locality — there is
    # no safe region to expand patch sampling into, so this must return None
    # rather than silently including test pixels.
    mask = polygon_train_region_mask({}, _profile(), (10, 10), {"method": "random"})
    assert mask is None


def test_polygon_train_region_mask_none_without_train_ids(tmp_path: Path):
    poly_path = tmp_path / "train_test.gpkg"
    _write_train_test_polygons(poly_path)

    split_cfg = {
        "method": "polygons",
        "polygons": {"path": str(poly_path), "id_field": "id"},
    }

    mask = polygon_train_region_mask({}, _profile(), (10, 10), split_cfg)
    assert mask is None
