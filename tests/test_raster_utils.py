"""Tests for the shared raster utilities.

Synthetic rasters only — no site data, no network.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from pci_source_zones import raster
from pci_source_zones.raster import GridMismatch


def _write(tmp_path, name, arr, nodata=-9999.0, transform=None, crs="EPSG:32610"):
    path = tmp_path / name
    transform = transform or from_origin(500000, 5000000, 10, 10)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", nodata=nodata, transform=transform, crs=crs)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.where(np.isfinite(arr), arr, nodata).astype("float32"), 1)
    return path


# ─────────────────────────────────────────────── read / write round trip

def test_read_float_maps_nodata_to_nan(tmp_path):
    a = np.array([[1.0, np.nan], [3.0, 4.0]], dtype="float32")
    p = _write(tmp_path, "a.tif", a)
    out, prof = raster.read_float(p)
    assert np.isnan(out[0, 1])
    assert out[0, 0] == pytest.approx(1.0)
    assert prof["height"] == 2


def test_write_float_round_trip(tmp_path):
    a = np.array([[1.5, np.nan], [3.5, 4.5]], dtype="float32")
    p = _write(tmp_path, "src.tif", a)
    _, prof = raster.read_float(p)
    out_path = raster.write_float(tmp_path / "out.tif", a, prof)
    back, _ = raster.read_float(out_path)
    assert np.allclose(back, a, equal_nan=True)


def test_read_binary_treats_ones_as_true(tmp_path):
    path = tmp_path / "mask.tif"
    arr = np.array([[1, 0], [1, 255]], dtype="uint8")
    prof = dict(driver="GTiff", height=2, width=2, count=1, dtype="uint8",
                nodata=255, transform=from_origin(0, 0, 1, 1), crs="EPSG:32610")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
    out, _ = raster.read_binary(path, nodata=255)
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0 and out[1, 0] == 1.0
    assert np.isnan(out[1, 1])


# ─────────────────────────────────────────────── block aggregation

def test_block_aggregate_mean():
    arr = np.arange(16, dtype="float32").reshape(4, 4)
    out = raster.block_aggregate(arr, 2, "mean")
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))


def test_block_aggregate_all_nan_block_returns_nan():
    arr = np.full((4, 4), np.nan, dtype="float32")
    arr[2:, 2:] = 1.0
    out = raster.block_aggregate(arr, 2, "mean")
    assert np.isnan(out[0, 0])
    assert out[1, 1] == pytest.approx(1.0)


def test_block_aggregate_sum_keeps_nan_for_empty_block():
    """Regression: np.nansum returns 0.0 for an all-NaN slice."""
    arr = np.full((2, 2), np.nan, dtype="float32")
    out = raster.block_aggregate(arr, 2, "sum")
    assert np.isnan(out[0, 0]), "all-NaN block must stay NaN, not become 0.0"


def test_block_aggregate_pads_uneven_shape():
    arr = np.ones((5, 7), dtype="float32")
    out = raster.block_aggregate(arr, 2, "mean")
    assert out.shape == (3, 4)
    assert out[0, 0] == pytest.approx(1.0)


def test_block_aggregate_rejects_bad_block():
    with pytest.raises(ValueError):
        raster.block_aggregate(np.ones((4, 4), dtype="float32"), 0, "mean")


def test_transform_for_block_scales_pixel_size():
    t = from_origin(500000, 5000000, 1, 1)
    t10 = raster.transform_for_block(t, 10)
    assert t10.a == pytest.approx(10.0)
    assert t10.e == pytest.approx(-10.0)
    assert t10.c == pytest.approx(t.c)   # origin unchanged


# ─────────────────────────────────────────────── nansum guard

def test_nansum_masked_preserves_nodata():
    """Regression for the MRMS coverage misread."""
    cube = np.full((3, 2, 2), np.nan, dtype="float32")
    cube[:, 0, 0] = [1.0, 2.0, 3.0]
    total = raster.nansum_masked(cube)
    assert total[0, 0] == pytest.approx(6.0)
    assert np.isnan(total[1, 1]), "cells with no valid layer must stay NaN"


# ─────────────────────────────────────────────── grid validation

def test_require_same_grid_passes_for_identical(tmp_path):
    a = np.ones((3, 3), dtype="float32")
    _, p1 = raster.read_float(_write(tmp_path, "a.tif", a))
    _, p2 = raster.read_float(_write(tmp_path, "b.tif", a))
    raster.require_same_grid(p1, {"b": p2})


def test_require_same_grid_raises_on_shape_mismatch(tmp_path):
    _, p1 = raster.read_float(_write(tmp_path, "a.tif", np.ones((3, 3), dtype="float32")))
    _, p2 = raster.read_float(_write(tmp_path, "b.tif", np.ones((4, 4), dtype="float32")))
    with pytest.raises(GridMismatch) as exc:
        raster.require_same_grid(p1, {"other": p2})
    assert "other" in str(exc.value)


def test_require_same_grid_raises_on_transform_mismatch(tmp_path):
    a = np.ones((3, 3), dtype="float32")
    _, p1 = raster.read_float(_write(tmp_path, "a.tif", a))
    _, p2 = raster.read_float(_write(tmp_path, "b.tif", a,
                                     transform=from_origin(999999, 5000000, 10, 10)))
    with pytest.raises(GridMismatch):
        raster.require_same_grid(p1, {"shifted": p2})


# ─────────────────────────────────────────────── stats

def test_raster_stats_empty_is_safe():
    s = raster.raster_stats(np.full((2, 2), np.nan))
    assert s["n_valid"] == 0
    assert np.isnan(s["mean"])


def test_raster_stats_values():
    s = raster.raster_stats(np.array([[1.0, 2.0], [3.0, np.nan]]))
    assert s["n_valid"] == 3
    assert s["mean"] == pytest.approx(2.0)
    assert s["max"] == pytest.approx(3.0)


# ───────────────────────────────── grid comparison tolerance

def test_transforms_match_tolerates_float_noise():
    """A COG rewrite can shift an origin by a fraction of a micron."""
    from affine import Affine
    from pci_source_zones.raster import transforms_match
    a = Affine(1.0, 0.0, 390000.0000005682, 0.0, -1.0, 3791250.0001043114)
    b = Affine(1.0, 0.0, 390000.0, 0.0, -1.0, 3791250.0001)
    assert transforms_match(a, b)


def test_transforms_match_rejects_half_pixel_offset():
    from affine import Affine
    from pci_source_zones.raster import transforms_match
    a = Affine(1.0, 0.0, 390000.0, 0.0, -1.0, 3791250.0)
    b = Affine(1.0, 0.0, 390000.5, 0.0, -1.0, 3791250.0)
    assert not transforms_match(a, b)


def test_transforms_match_rejects_different_pixel_size():
    from affine import Affine
    from pci_source_zones.raster import transforms_match
    a = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    b = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0)
    assert not transforms_match(a, b)


def test_transforms_match_boundary_is_one_hundredth_pixel():
    from affine import Affine
    from pci_source_zones.raster import transforms_match
    base = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    assert transforms_match(base, Affine(10.0, 0.0, 0.09, 0.0, -10.0, 0.0))
    assert not transforms_match(base, Affine(10.0, 0.0, 0.2, 0.0, -10.0, 0.0))
