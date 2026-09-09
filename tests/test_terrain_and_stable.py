"""Terrain (G, DeltaG) and stable-ground stages."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from pci_source_zones import evidence, runner
from pci_source_zones.physics.terrain import (
    TerrainParams,
    driving_index,
    terrain_deltas,
)
from pci_source_zones.site_config import ConfigError, SiteConfig

whitebox = pytest.importorskip("whitebox")


def _raster(path, arr, res=1.0, origin=(500000, 5000000), crs="EPSG:32610"):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", nodata=-9999.0, crs=crs,
                transform=from_origin(*origin, res, res))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def _dem_pair(tmp_path, n=60):
    """A tilted, gullied surface, and the same surface after incision."""
    yy, xx = np.mgrid[0:n, 0:n].astype("float64")
    pre = 100.0 - 0.30 * yy + 2.0 * np.sin(xx / 4.0)
    gully = 3.0 * np.exp(-((xx - n / 2) ** 2) / 18.0)
    post = pre - gully
    return (_raster(tmp_path / "dem_pre.tif", pre),
            _raster(tmp_path / "dem_post.tif", post), pre, post)


# ────────────────────────────────────────────── the G formula itself

def test_driving_index_matches_definition():
    a = np.array([100.0, 50.0])
    s = np.array([0.5, 0.25])
    assert np.allclose(driving_index(a, s, 1.167), a * s ** 1.167)


def test_driving_index_rejects_negative_and_nonfinite():
    g = driving_index(np.array([-5.0, np.nan, 10.0]), np.array([0.5, 0.5, 0.5]), 2.0)
    assert np.isnan(g[0]) and np.isnan(g[1]) and np.isfinite(g[2])


def test_alpha_changes_g_monotonically():
    a, s = np.array([100.0]), np.array([0.5])          # S < 1, so larger alpha -> smaller G
    assert driving_index(a, s, 2.0)[0] < driving_index(a, s, 1.167)[0]


def test_bad_alpha_rejected():
    with pytest.raises(ValueError, match="alpha"):
        TerrainParams(alpha=0)


# ────────────────────────────────────────────── terrain_deltas

def test_terrain_deltas_writes_products(tmp_path):
    pre, post, _, _ = _dem_pair(tmp_path)
    out = terrain_deltas(pre, post, tmp_path / "out", TerrainParams())
    for key in ("g_pre", "g_post", "delta_g"):
        assert (tmp_path / "out").joinpath(f"{out[key].split('/')[-1]}").exists()
    assert out["alpha"] == 1.167
    assert out["n_valid"] > 0


def test_incision_lowers_g_where_the_gully_deepens(tmp_path):
    """Physical sanity: deepening a gully changes G, and DeltaG is not all zero."""
    pre, post, _, _ = _dem_pair(tmp_path)
    terrain_deltas(pre, post, tmp_path / "out", TerrainParams())
    with rasterio.open(tmp_path / "out" / "DeltaG.tif") as src:
        dg = src.read(1)
        dg = np.where(dg == src.nodata, np.nan, dg)
    assert np.nanstd(dg) > 0, "DeltaG is constant — the DEM pair produced no signal"


def test_mismatched_dem_grids_rejected(tmp_path):
    pre, _, _, _ = _dem_pair(tmp_path)
    small = _raster(tmp_path / "small.tif", np.ones((20, 20), dtype="float32"))
    with pytest.raises(ValueError, match="different grids"):
        terrain_deltas(pre, small, tmp_path / "out", TerrainParams())


def test_alpha_propagates_from_config(tmp_path):
    from pci_source_zones.site_config import TerrainSpec
    assert TerrainSpec.parse({"alpha": 2.0}).params().alpha == 2.0


# ────────────────────────────────────────────── stable-ground core

def test_threshold_mask_keeps_nodata_as_nan():
    m = evidence.threshold_mask(np.array([0.1, 5.0, np.nan]), 1.0)
    assert m[0] == 1.0 and m[1] == 0.0 and np.isnan(m[2])


def test_range_mask_is_half_open():
    m = evidence.range_mask(np.array([4.9, 5.0, 9.9, 10.0]), 5.0, 10.0)
    assert list(m) == [0.0, 1.0, 1.0, 0.0]


def test_combine_all_requires_every_layer_valid():
    a = np.array([[1.0, 1.0]])
    b = np.array([[1.0, np.nan]])
    out = evidence.combine_masks(np.stack([a, b]), "all")
    assert out[0, 0] == 1.0 and np.isnan(out[0, 1])


def test_combine_any_accepts_one_valid_layer():
    a = np.array([[0.0, np.nan]])
    b = np.array([[1.0, 1.0]])
    out = evidence.combine_masks(np.stack([a, b]), "any")
    assert out[0, 0] == 1.0 and out[0, 1] == 1.0


def test_combine_rejects_bad_mode():
    with pytest.raises(ValueError, match="combine must be"):
        evidence.combine_masks(np.ones((1, 2, 2)), "most")


def test_check_representative_flags_a_tiny_sample():
    cov = np.arange(100.0).reshape(10, 10)
    mask = np.zeros((10, 10)); mask[0, 0] = 1.0
    rep = evidence.check_representative(mask, cov, "G", min_frac=0.05)
    assert "warning" in rep


# ────────────────────────────────────────────── stages

def _site(tmp_path, **extra):
    pre, post, _, _ = _dem_pair(tmp_path)
    dz = _raster(tmp_path / "dz.tif", np.full((60, 60), -0.4, dtype="float32"))
    d = {
        "site": "t",
        "grid": {"fine_res_m": 1.0, "coarse_res_m": 10.0},
        "inputs": {"mode": "single", "dem_pre": str(pre), "dem_post": str(post),
                   "dz": str(dz)},
        "outputs": {"root": str(tmp_path / "out")},
    }
    d.update(extra)
    return SiteConfig.parse(d)


def test_dem_pair_satisfies_the_input_requirement(tmp_path):
    """g_pre/delta_g may be omitted when the DEM pair is supplied."""
    cfg = _site(tmp_path)
    assert cfg.inputs.g_pre is None and cfg.inputs.dem_pre is not None


def test_terrain_stage_fails_without_dems_or_precomputed_g(tmp_path):
    cfg = SiteConfig.parse({"site": "t", "inputs": {"mode": "single", "dz": "x.tif"},
                            "outputs": {"root": str(tmp_path)}})
    res = runner.run(cfg, ["terrain"])[0]
    assert res.status == "failed" and "dem_pre" in res.error


def test_terrain_stage_feeds_the_next_stage(tmp_path):
    """The whole point: lod reads DeltaG the terrain stage produced."""
    cfg = _site(tmp_path)
    results = runner.run(cfg, ["terrain"])
    assert results[0].status == "ok"
    assert (cfg.output_dir() / "DeltaG.tif").exists()
    assert runner.resolve_input(cfg, "delta_g", None) == cfg.output_dir() / "DeltaG.tif"


def test_terrain_stage_skips_when_g_supplied_directly(tmp_path):
    g = _raster(tmp_path / "G.tif", np.ones((60, 60), dtype="float32"))
    cfg = SiteConfig.parse({
        "site": "t", "outputs": {"root": str(tmp_path / "o")},
        "inputs": {"mode": "single", "g_pre": str(g), "delta_g": str(g),
                   "dz": str(g)}})
    assert runner.run(cfg, ["terrain"])[0].status == "skipped"


def test_stable_stage_builds_a_mask_from_criteria(tmp_path):
    slope = _raster(tmp_path / "slope.tif",
                    np.tile(np.linspace(0, 60, 60), (60, 1)).astype("float32"))
    cfg = _site(tmp_path, stable={
        "method": "criteria",
        "layers": [{"name": "slope", "path": str(slope), "max": 30.0}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "ok"
    assert 0 < res.detail["frac_stable"] < 1
    assert (cfg.output_dir() / "stable_cells.tif").exists()


def test_stable_stage_flags_circular_threshold_method(tmp_path, capsys):
    dz = _raster(tmp_path / "d.tif",
                 np.random.default_rng(0).normal(0, 1, (60, 60)).astype("float32"))
    cfg = _site(tmp_path, stable={
        "method": "threshold",
        "layers": [{"name": "dz", "path": str(dz), "threshold": 1.0}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.detail["circular"] is True
    assert "biased low" in capsys.readouterr().out


def test_stable_stage_fails_when_mask_is_empty(tmp_path):
    slope = _raster(tmp_path / "s.tif", np.full((60, 60), 80.0, dtype="float32"))
    cfg = _site(tmp_path, stable={
        "layers": [{"name": "slope", "path": str(slope), "max": 30.0}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "failed" and "0 cells" in res.error


def test_stable_stage_fails_when_required_layer_is_missing(tmp_path):
    cfg = _site(tmp_path, stable={
        "layers": [{"name": "missing", "path": str(tmp_path / "missing.tif"),
                    "max": 30.0}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "failed" and "required stable layer" in res.error


def test_stable_stage_can_skip_optional_layer(tmp_path):
    slope = _raster(tmp_path / "slope.tif", np.zeros((60, 60), dtype="float32"))
    cfg = _site(tmp_path, stable={
        "layers": [{"name": "slope", "path": str(slope), "max": 30.0},
                   {"name": "optional", "path": str(tmp_path / "missing.tif"),
                    "max": 30.0, "required": False}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "ok"
    assert res.detail["layers_skipped"] == ["optional"]


def test_stable_stage_fails_on_grid_mismatch(tmp_path):
    a = _raster(tmp_path / "a.tif", np.zeros((60, 60), dtype="float32"))
    b = _raster(tmp_path / "b.tif", np.zeros((60, 60), dtype="float32"),
                origin=(500100, 5000000))
    cfg = _site(tmp_path, stable={
        "layers": [{"name": "a", "path": str(a), "max": 30.0},
                   {"name": "b", "path": str(b), "max": 30.0}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "failed" and "grid mismatch" in res.error


def test_stable_layer_needs_a_criterion(tmp_path):
    with pytest.raises(ConfigError, match="threshold, min, max"):
        _site(tmp_path, stable={"layers": [{"name": "x", "path": "x.tif"}]})


def test_stable_stage_skips_when_mask_supplied_directly(tmp_path):
    cfg = _site(tmp_path)
    assert runner.run(cfg, ["stable"])[0].status == "skipped"


def test_stage_order_puts_terrain_and_stable_before_lod():
    order = list(runner.STAGES)
    assert order.index("terrain") < order.index("stable") < order.index("lod")


def test_stable_layer_may_be_a_burn_perimeter_polygon(tmp_path):
    """The key independent criterion is usually a vector, not a raster."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    slope = _raster(tmp_path / "slope.tif", np.zeros((60, 60), dtype="float32"))
    poly = box(500000, 5000000 - 30, 500000 + 30, 5000000)     # upper-left quarter
    geopandas.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32610").to_file(
        tmp_path / "burn.shp")

    cfg = _site(tmp_path, stable={
        "layers": [{"name": "slope", "path": str(slope), "max": 30.0},
                   {"name": "burned", "path": str(tmp_path / "burn.shp"),
                    "invert": True}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "ok"
    # stable == outside the perimeter == three quarters of the grid
    assert 0.7 < res.detail["frac_stable"] < 0.8


def test_vector_stable_layer_needs_a_raster_for_the_grid(tmp_path):
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box
    poly = box(500000, 5000000 - 30, 500000 + 30, 5000000)
    geopandas.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32610").to_file(
        tmp_path / "b.shp")
    cfg = _site(tmp_path, stable={
        "layers": [{"name": "burned", "path": str(tmp_path / "b.shp"), "invert": True}]})
    res = runner.run(cfg, ["stable"])[0]
    assert res.status == "failed" and "no raster layer" in res.error


def test_terrain_work_dir_is_configurable(tmp_path):
    pre, post, _, _ = _dem_pair(tmp_path)
    scratch = tmp_path / "scratch"
    terrain_deltas(pre, post, tmp_path / "out",
                   TerrainParams(work_dir=str(scratch)))
    assert (scratch / "pre_filled.tif").exists()
    assert not (tmp_path / "out" / "_terrain_work").exists()
