"""End-to-end pipeline tests on synthetic rasters.

Builds a small fake site on disk, runs every stage through the runner, and checks
the products. Also covers the production behaviours lifted from the batch scripts:
resumability, per-area failure isolation, and run summaries.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from pci_source_zones import runner
from pci_source_zones.site_config import SiteConfig

SIZE = 200          # fine grid, divisible by the block size
BLOCK = 10


def _write_raster(path, arr, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", nodata=nodata, crs="EPSG:32610",
                transform=from_origin(500000, 5000000, 1, 1))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.where(np.isfinite(arr), arr, nodata).astype("float32"), 1)
    return path


def _write_mask(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="uint8", nodata=255, crs="EPSG:32610",
                transform=from_origin(500000, 5000000, 1, 1))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("uint8"), 1)
    return path


@pytest.fixture
def site(tmp_path):
    """A synthetic single-site dataset with a known erosion signal."""
    rng = np.random.default_rng(7)
    d = tmp_path / "data"

    # terrain forcing spanning the hillslope domain, with a channel stripe
    g = rng.uniform(6, 250, (SIZE, SIZE)).astype("float32")
    g[:, :20] = 500.0                       # channel, excluded by g_channel
    g[:, 20:30] = 1.0                        # flats, excluded by g_min

    # change signal: mostly noise, strong in one quadrant
    delta_g = rng.normal(0, 1, (SIZE, SIZE)).astype("float32")
    delta_g[:100, 100:] += 25.0

    # surface change: lowering in the same quadrant
    dz = rng.normal(0.05, 0.02, (SIZE, SIZE)).astype("float32")
    dz[:100, 100:] = -0.5

    stable = np.ones((SIZE, SIZE), dtype="uint8")
    stable[:100, 100:] = 0                   # the eroding quadrant is not stable

    dem = np.full((SIZE, SIZE), 800.0, dtype="float32")
    slope = np.full((SIZE, SIZE), 0.2, dtype="float32")

    paths = {
        "g_pre": _write_raster(d / "g_pre.tif", g),
        "s_pre": _write_raster(d / "s_pre.tif", slope),
        "delta_g": _write_raster(d / "delta_g.tif", delta_g),
        "dz": _write_raster(d / "dz.tif", dz),
        "stable_cells": _write_mask(d / "stable.tif", stable),
        "dem": _write_raster(d / "dem.tif", dem),
    }

    cfg_dict = {
        "site": "synthfire",
        "epochs": {"pre": 2020, "post": 2022},
        "grid": {"fine_res_m": 1, "coarse_res_m": BLOCK},
        "inputs": {"mode": "single", **{k: str(v) for k, v in paths.items()}},
        "outputs": {"root": str(tmp_path / "out")},
        "lod": {"scale": 2.0},
        "evidence": {"domain": {"g_min": 5, "g_channel": 300}},
        "ceff": {"elev_min": 500, "p_ceiling": 0.05},
    }
    cfg_path = tmp_path / "site.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    return SiteConfig.load(cfg_path)


# ────────────────────────────────────────────── full pipeline

def test_full_pipeline_runs_and_produces_products(site):
    results = runner.run(site, runner.STAGES)
    assert not [r for r in results if r.status == "failed"], [r.error for r in results]
    # this fixture declares no `align:` section, so that stage reports itself
    # as skipped rather than doing anything
    assert [(r.stage, r.status) for r in results][0] == ("align", "skipped")

    out = site.output_dir()
    for name in ("lod.json", "C2.tif", "C3.tif", "p_obs_intersection.tif",
                 "p_obs_union.tif", "G_mean.tif", "gamma_fit.json",
                 "C_p_obs_inter.tif", "C_p_obs_union.tif", "C_p050.tif",
                 "C_p050_log.tif"):
        assert (out / name).exists(), f"missing product {name}"


def test_lod_is_positive_and_recorded(site):
    runner.run(site, ["lod"])
    lod = json.loads((site.output_dir() / "lod.json").read_text())
    assert lod["delta_g_lod"] > 0
    assert lod["n_stable"] > 0
    assert lod["scale"] == 2.0


def test_p_obs_in_unit_interval_and_union_ge_intersection(site):
    runner.run(site, ["lod", "evidence", "p_obs"])
    out = site.output_dir()
    with rasterio.open(out / "p_obs_union.tif") as s:
        u = s.read(1); u[u == s.nodata] = np.nan
    with rasterio.open(out / "p_obs_intersection.tif") as s:
        i = s.read(1); i[i == s.nodata] = np.nan
    v = np.isfinite(u) & np.isfinite(i)
    assert u[v].min() >= 0.0 and u[v].max() <= 1.0
    assert np.all(u[v] >= i[v] - 1e-6), "union must be >= intersection"


def test_coarse_grid_shape_matches_block_factor(site):
    runner.run(site, ["lod", "evidence", "p_obs"])
    with rasterio.open(site.output_dir() / "p_obs_union.tif") as s:
        assert (s.height, s.width) == (SIZE // BLOCK, SIZE // BLOCK)
        assert s.transform.a == pytest.approx(BLOCK)


def test_domain_mask_excludes_channel_and_flats(site):
    """Channel and flat columns must not carry an erosion probability."""
    runner.run(site, ["lod", "evidence", "p_obs"])
    with rasterio.open(site.output_dir() / "G_mean.tif") as s:
        g = s.read(1); g[g == s.nodata] = np.nan
    v = g[np.isfinite(g)]
    assert v.min() >= 5.0 and v.max() < 300.0


def test_evidence_stage_supports_per_cell_lod_raster(site):
    sigma = np.full((SIZE, SIZE), 0.1, dtype="float32")
    sigma_path = _write_raster(site.source_path.parent / "data" / "sigma.tif", sigma)
    cfg = SiteConfig.load(site.source_path, overrides={
        "evidence": {"c3": {"lod": str(sigma_path), "scale": 1.96}}})
    results = runner.run(cfg, ["lod", "evidence"], overwrite=True)
    assert not [r for r in results if r.status == "failed"]
    assert results[-1].detail["c3_lod_source"].startswith("raster")
    assert abs(results[-1].detail["c3_lod"] - 0.196) < 1e-6


def test_evidence_stage_auto_lod_measures_from_stable_ground(site):
    """`lod: auto` carries no site-specific number — it measures one."""
    cfg = SiteConfig.load(site.source_path,
                          overrides={"evidence": {"c3": {"lod": "auto"}}})
    results = runner.run(cfg, ["lod", "evidence"], overwrite=True)
    assert not [r for r in results if r.status == "failed"]
    lod = json.loads((site.output_dir() / "lod.json").read_text())
    assert lod["dz_lod"] > 0 and lod["dz_lod"] == pytest.approx(
        lod["nmad_dz"] * lod["dz_scale"])
    assert results[-1].detail["c3_lod_source"] == "auto"


def test_auto_lod_without_dz_is_rejected(site):
    """Fail loudly rather than silently falling back to a threshold of zero."""
    cfg = SiteConfig.load(site.source_path,
                          overrides={"evidence": {"c3": {"lod": "auto"}}})
    cfg.inputs.dz = None
    results = runner.run(cfg, ["lod"], overwrite=True)
    assert results[0].status == "failed" and "dz" in results[0].error


def test_p_obs_applies_configured_elevation_and_slope_domain(site):
    cfg = SiteConfig.load(site.source_path, overrides={
        "evidence": {"domain": {"g_min": 5, "g_channel": 300,
                                "elev_min": 700, "slope_max_deg": 45}}
    })
    results = runner.run(cfg, ["lod", "evidence", "p_obs"], overwrite=True)
    assert not [r for r in results if r.status == "failed"]
    assert results[-1].detail["elev_min"] == 700
    assert results[-1].detail["slope_max_deg"] == 45


def test_threshold_inversion_produces_positive_values(site):
    runner.run(site, runner.STAGES)
    with rasterio.open(site.output_dir() / "C_p_obs_union.tif") as s:
        c = s.read(1); c[c == s.nodata] = np.nan
    v = c[np.isfinite(c)]
    assert v.size > 0 and v.min() > 0


def test_upper_bound_covers_more_cells_than_inversion(site):
    """C at a fixed p is defined everywhere in the domain; C(p_obs) is not."""
    runner.run(site, runner.STAGES)
    out = site.output_dir()

    def n_valid(name):
        with rasterio.open(out / name) as s:
            a = s.read(1)
            return int((a != s.nodata).sum())

    assert n_valid("C_p050.tif") >= n_valid("C_p_obs_union.tif")


# ────────────────────────────────────────────── production behaviours

def test_second_run_skips_existing_outputs(site):
    runner.run(site, runner.STAGES)
    again = runner.run(site, runner.STAGES)
    assert all(r.status == "skipped" for r in again)   # align included: not configured


def test_overwrite_forces_recompute(site):
    runner.run(site, ["lod"])
    again = runner.run(site, ["lod"], overwrite=True)
    assert all(r.status == "ok" for r in again)


def test_missing_prerequisite_is_isolated_not_raised(site):
    """Running a downstream stage alone fails that stage, without crashing."""
    results = runner.run(site, ["ceff"])
    assert results[0].status == "failed"
    assert "not found" in results[0].error


def test_fail_fast_raises(site):
    with pytest.raises(FileNotFoundError):
        runner.run(site, ["ceff"], fail_fast=True)


def test_unknown_stage_rejected(site):
    with pytest.raises(ValueError, match="unknown stage"):
        runner.run(site, ["bogus"])


def test_summary_and_manifest_written(site):
    results = runner.run(site, ["lod", "evidence"])
    paths = runner.write_summary(site, results, ["lod", "evidence"])
    assert paths["summary"].exists() and paths["manifest"].exists()
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["site"] == "synthfire"
    assert manifest["counts"]["ok"] == 2
    assert manifest["config"]["grid"]["block_size"] == BLOCK


def test_single_site_discovers_one_area(site):
    assert runner.discover_areas(site) == [None]


def test_batch_discovery_keeps_only_complete_input_sets(tmp_path):
    """Batch mode should not launch an ID missing a required input raster."""
    pre = tmp_path / "pre"
    post = tmp_path / "post"
    dz = tmp_path / "dz"
    for root in (pre, post, dz):
        root.mkdir()

    # ID 1 is complete. ID 2 has no post DEM and must not be discovered.
    (pre / "1_Pre.tif").touch()
    (pre / "2_Pre.tif").touch()
    (post / "1_Post.tif").touch()
    (dz / "1_DoD.tif").touch()
    (dz / "2_DoD.tif").touch()

    cfg = SiteConfig.parse({
        "site": "batch_synth",
        "inputs": {
            "mode": "batch",
            "roots": {"dem_pre": str(pre), "dem_post": str(post), "dz": str(dz)},
            "templates": {
                "dem_pre": "{id}_Pre.tif",
                "dem_post": "{id}_Post.tif",
                "dz": "{id}_DoD.tif",
            },
        },
        "outputs": {"root": str(tmp_path / "out")},
        "stable": {
            "layers": [{"name": "dummy", "path": str(pre / "1_Pre.tif"), "max": 5.0}]
        },
    })

    assert runner.discover_areas(cfg) == ["1"]


def test_detail_cannot_overwrite_identity_columns():
    """Several stages put their own 'area_id' in detail; it must not win."""
    r = runner.AreaResult("area_7", "terrain", "ok",
                          {"area_id": None, "stage": "wrong", "n_valid": 5})
    row = r.row()
    assert row["area_id"] == "area_7"
    assert row["stage"] == "terrain"
    assert row["n_valid"] == 5
