"""The input contract — what the pipeline demands of its inputs, checked up front."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pci_source_zones.contract import (
    ContractError,
    check_inputs,
    format_issues,
    require_contract,
)
from pci_source_zones.site_config import SiteConfig

ALL_STAGES = ["terrain", "stable", "lod", "evidence", "p_obs", "ceff"]


def _raster(path, arr, res=(1.0, 1.0), origin=(0.0, 100.0), crs="EPSG:32610",
            nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32", crs=crs,
                       nodata=nodata,
                       transform=from_origin(origin[0], origin[1], res[0], res[1])) as d:
        d.write(arr.astype("float32"), 1)
    return path


@pytest.fixture
def good(tmp_path):
    """A config whose inputs satisfy the contract."""
    a = np.full((40, 40), 100.0, dtype="float32")
    paths = {
        "dem_pre": _raster(tmp_path / "pre.tif", a),
        "dem_post": _raster(tmp_path / "post.tif", a - 1),
        "dz": _raster(tmp_path / "dz.tif", np.full((40, 40), -0.5, dtype="float32")),
        "stable_cells": _raster(tmp_path / "stable.tif",
                                np.ones((40, 40), dtype="float32")),
    }
    return SiteConfig.parse({
        "site": "t",
        "inputs": {"mode": "single", **{k: str(v) for k, v in paths.items()}},
        "outputs": {"root": str(tmp_path / "out")}}), tmp_path


def _names(issues):
    return {(i.name, i.level) for i in issues}


# ───────────────────────────────────────────── the contract holds

def test_clean_inputs_pass(good):
    cfg, _ = good
    assert check_inputs(cfg, stages=ALL_STAGES) == []
    assert format_issues([]) == "input contract: OK"
    require_contract(cfg, stages=ALL_STAGES)          # does not raise


# ───────────────────────────────────────────── the four requirements

def test_non_square_pixels_are_caught(good, tmp_path):
    """Never checked anywhere before: they did not fail, they made slope wrong."""
    cfg, _ = good
    cfg.inputs.dz = str(_raster(tmp_path / "ns.tif",
                                np.zeros((40, 40), dtype="float32"),
                                res=(1.17, 1.88)))
    issues = check_inputs(cfg, stages=ALL_STAGES)
    assert any("not square" in i.problem for i in issues)
    assert all(i.level == "error" for i in issues if "not square" in i.problem)


def test_mismatched_grid_is_caught(good, tmp_path):
    cfg, _ = good
    cfg.inputs.dz = str(_raster(tmp_path / "small.tif",
                                np.zeros((20, 20), dtype="float32")))
    assert any("grid differs" in i.problem
               for i in check_inputs(cfg, stages=ALL_STAGES))


def test_sub_micron_offset_is_not_a_mismatch(good, tmp_path):
    """A COG rewrite shifts the origin a fraction of a micron. That is not a bug."""
    cfg, _ = good
    cfg.inputs.dz = str(_raster(tmp_path / "shift.tif",
                                np.zeros((40, 40), dtype="float32"),
                                origin=(5.68e-7, 100.0)))
    assert not any("grid differs" in i.problem
                   for i in check_inputs(cfg, stages=ALL_STAGES))


def test_undeclared_fill_is_caught(good, tmp_path):
    cfg, _ = good
    arr = np.random.default_rng(0).normal(500, 10, (60, 60)).astype("float32")
    arr[:30] = 0.0
    cfg.inputs.dz = str(_raster(tmp_path / "fill.tif", arr, nodata=None))
    issues = check_inputs(cfg, stages=ALL_STAGES)
    assert any("no nodata declared" in i.problem and "exactly 0" in i.problem
               for i in issues)


def test_binary_mask_is_not_mistaken_for_fill(good, tmp_path):
    """A stable-ground mask is mostly zeros by design, not by omission."""
    cfg, _ = good
    mask = np.zeros((60, 60), dtype="float32")
    mask[:12] = 1.0
    cfg.inputs.stable_cells = str(_raster(tmp_path / "m.tif", mask, nodata=None))
    issues = check_inputs(cfg, stages=ALL_STAGES)
    fills = [i for i in issues if "exactly" in i.problem]
    assert not fills, f"binary mask flagged as fill: {fills}"


def test_missing_crs_is_caught(good, tmp_path):
    cfg, _ = good
    cfg.inputs.dz = str(_raster(tmp_path / "nocrs.tif",
                                np.zeros((40, 40), dtype="float32"), crs=None))
    assert ("dz", "error") in _names(check_inputs(cfg, stages=ALL_STAGES))


# ───────────────────────────────────────────── missing sources

def test_no_source_for_g_is_caught(tmp_path):
    cfg = SiteConfig.parse({"site": "t", "inputs": {"mode": "single", "dz": "x.tif"},
                            "outputs": {"root": str(tmp_path)}})
    assert any(i.problem == "no source for G"
               for i in check_inputs(cfg, stages=["terrain"]))


def test_dem_pair_counts_as_a_source_for_g(good):
    cfg, _ = good
    assert not any("no source for G" in i.problem
                   for i in check_inputs(cfg, stages=ALL_STAGES))


def test_derived_products_are_not_reported_missing(good):
    """g_pre/delta_g do not exist yet when the terrain stage will make them."""
    cfg, _ = good
    assert not any(i.name in ("g_pre", "delta_g")
                   for i in check_inputs(cfg, stages=ALL_STAGES))


def test_missing_stable_source_is_caught(tmp_path):
    a = np.full((10, 10), 1.0, dtype="float32")
    cfg = SiteConfig.parse({
        "site": "t",
        "inputs": {"mode": "single", "dz": str(_raster(tmp_path / "z.tif", a)),
                   "g_pre": str(_raster(tmp_path / "g.tif", a)),
                   "delta_g": str(_raster(tmp_path / "d.tif", a))},
        "outputs": {"root": str(tmp_path / "o")}})
    assert any(i.name == "stable_cells" for i in check_inputs(cfg, stages=["lod"]))


def test_requirements_are_scoped_to_requested_stages(tmp_path):
    """A partial run must not be blocked by inputs it never touches."""
    cfg = SiteConfig.parse({"site": "t", "inputs": {"mode": "single", "dz": "x.tif"},
                            "outputs": {"root": str(tmp_path)}})
    assert check_inputs(cfg, stages=["ceff"]) == []
    assert check_inputs(cfg, stages=["terrain"]) != []


# ───────────────────────────────────────────── reporting behaviour

def test_all_problems_reported_at_once(good, tmp_path):
    """One failure per run would mean N runs to find N problems."""
    cfg, _ = good
    cfg.inputs.dz = str(_raster(tmp_path / "bad.tif",
                                np.zeros((20, 20), dtype="float32"),
                                res=(1.17, 1.88)))
    issues = check_inputs(cfg, stages=ALL_STAGES)
    assert len(issues) >= 2, "expected both the shape and the squareness problem"


def test_missing_file_is_a_warning_for_validate_and_an_error_for_run(tmp_path):
    a = np.full((10, 10), 1.0, dtype="float32")
    cfg = SiteConfig.parse({
        "site": "t",
        "inputs": {"mode": "single", "dz": str(_raster(tmp_path / "z.tif", a)),
                   "g_pre": str(_raster(tmp_path / "g.tif", a)),
                   "delta_g": str(_raster(tmp_path / "d.tif", a)),
                   "stable_cells": str(tmp_path / "gone.tif")},
        "outputs": {"root": str(tmp_path / "o")}})
    lenient = check_inputs(cfg, missing_is_error=False, stages=ALL_STAGES)
    strict = check_inputs(cfg, missing_is_error=True, stages=ALL_STAGES)
    assert ("stable_cells", "warning") in _names(lenient)
    assert ("stable_cells", "error") in _names(strict)


def test_require_contract_raises_with_the_full_report(good, tmp_path):
    cfg, _ = good
    cfg.inputs.dz = str(_raster(tmp_path / "b.tif",
                                np.zeros((20, 20), dtype="float32")))
    with pytest.raises(ContractError, match="grid differs"):
        require_contract(cfg, stages=ALL_STAGES)


def test_report_separates_errors_from_warnings():
    from pci_source_zones.contract import Issue
    text = format_issues([Issue("a", "error", "broken"),
                          Issue("b", "warning", "odd")])
    assert "1 FAILURE" in text and "1 warning" in text
