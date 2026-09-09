"""Input preparation: fill masking and regridding onto one analysis grid."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from pci_source_zones.prep import (
    PrepConfig,
    PrepError,
    RasterSpec,
    TargetGrid,
    run_prep,
    scan_fills,
)


def _raster(path, arr, res=(1.0, 1.0), origin=(0.0, 100.0), crs="EPSG:32610",
            nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs=crs, nodata=nodata,
                transform=from_origin(origin[0], origin[1], res[0], res[1]))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


@pytest.fixture
def scene(tmp_path):
    """Two DEMs and a difference, on three different non-square grids."""
    rng = np.random.default_rng(0)
    pre = 500.0 + rng.normal(0, 5, (60, 80)).astype("float32")
    post = pre - 1.0
    dz = (post - pre).astype("float32")
    pre[0:10, 0:10] = 0.0            # background fill
    pre[50:55, 0:5] = -999.0         # second, different fill
    post[0:10, 0:10] = 0.0
    dz[0:10, 0:10] = 0.0             # difference: exact zero is footprint fill
    dz[20:30, 20:30] = -2.5          # erosion — must survive

    return {
        "pre": _raster(tmp_path / "pre.tif", pre, res=(1.17, 1.88)),
        "post": _raster(tmp_path / "post.tif", post, res=(1.19, 1.86),
                        origin=(1.0, 99.0)),
        "dz": _raster(tmp_path / "dz.tif", dz, res=(1.16, 1.89), origin=(0.5, 99.5)),
        "dir": tmp_path,
    }


def _cfg(scene, **overrides):
    d = {
        "target": {"res": 2.0, "extent": "intersection", "nodata": -9999.0},
        "output_dir": str(scene["dir"] / "prep"),
        "rasters": [
            {"name": "dem_pre", "path": str(scene["pre"]), "kind": "dem",
             "fill": [0, -999], "valid_min": 0},
            {"name": "dem_post", "path": str(scene["post"]), "kind": "dem",
             "fill": [0, -999], "valid_min": 0},
            {"name": "dz", "path": str(scene["dz"]), "kind": "diff",
             "fill": [0, -999]},
        ],
    }
    d.update(overrides)
    return d


# ───────────────────────────────────────────── the two mistakes this prevents

def test_valid_min_on_a_difference_is_rejected():
    """Negatives in a dDEM are the erosion signal, not fill.

    Applying an elevation-style lower bound here deletes every eroded cell and
    leaves an all-positive difference raster, which looks plausible and is not.
    """
    with pytest.raises(PrepError, match="valid_min is not allowed"):
        RasterSpec.parse({"name": "dz", "path": "x.tif", "kind": "diff",
                          "valid_min": 0})


def test_erosion_survives_preparation(scene):
    """The whole point: negative dz must still be there afterwards."""
    run_prep(PrepConfig.parse(_cfg(scene)))
    with rasterio.open(scene["dir"] / "prep" / "dz.tif") as src:
        a = src.read(1).astype("float64")
        a[a == src.nodata] = np.nan
    assert np.nanmin(a) < -2.0, "erosion was masked away"
    assert (a[np.isfinite(a)] < 0).any()


def test_fill_is_masked_before_resampling(tmp_path):
    """A resampler must never blend a fill value into real data.

    Masking after the warp leaves interpolated values between the fill and the
    real surface — e.g. -654 between -999 and 1000 — which no later threshold
    can identify.
    """
    arr = np.full((40, 40), 1000.0, dtype="float32")
    arr[:, :20] = -999.0
    src = _raster(tmp_path / "s.tif", arr, res=(1.0, 1.0))
    cfg = PrepConfig.parse({
        "target": {"res": 2.0, "nodata": -9999.0},
        "output_dir": str(tmp_path / "out"),
        "rasters": [{"name": "d", "path": str(src), "kind": "dem",
                     "fill": [-999], "resampling": "bilinear"}]})
    run_prep(cfg)
    with rasterio.open(tmp_path / "out" / "d.tif") as s:
        a = s.read(1).astype("float64")
        a[a == s.nodata] = np.nan
    real = a[np.isfinite(a)]
    assert real.size > 0
    assert real.min() > 999.0, f"fill bled into the data: min={real.min()}"


# ───────────────────────────────────────────── grid resolution

def test_outputs_share_one_square_grid(scene):
    run_prep(PrepConfig.parse(_cfg(scene)))
    grids = []
    for name in ("dem_pre", "dem_post", "dz"):
        with rasterio.open(scene["dir"] / "prep" / f"{name}.tif") as s:
            grids.append((s.width, s.height, s.transform, str(s.crs)))
            assert abs(s.res[0] - s.res[1]) < 1e-9, "pixels are not square"
    assert all(g == grids[0] for g in grids)


def test_intersection_is_inside_every_input(scene):
    cfg = PrepConfig.parse(_cfg(scene))
    grid = cfg.target.resolve(cfg.rasters)
    left, bottom, right, top = grid["bounds"]
    for spec in cfg.rasters:
        with rasterio.open(spec.path) as s:
            b = s.bounds
        assert left >= b.left - 2.0 and right <= b.right + 2.0


def test_union_is_larger_than_intersection(scene):
    inter = PrepConfig.parse(_cfg(scene)).target.resolve(
        PrepConfig.parse(_cfg(scene)).rasters)
    d = _cfg(scene)
    d["target"]["extent"] = "union"
    cfg = PrepConfig.parse(d)
    union = cfg.target.resolve(cfg.rasters)
    assert union["width"] * union["height"] > inter["width"] * inter["height"]


def test_explicit_bbox_is_honoured(scene):
    d = _cfg(scene)
    d["target"]["extent"] = [10.0, 40.0, 30.0, 60.0]
    cfg = PrepConfig.parse(d)
    grid = cfg.target.resolve(cfg.rasters)
    assert grid["width"] == 10 and grid["height"] == 10


def test_non_overlapping_inputs_are_rejected(tmp_path):
    a = _raster(tmp_path / "a.tif", np.ones((10, 10), dtype="float32"))
    b = _raster(tmp_path / "b.tif", np.ones((10, 10), dtype="float32"),
                origin=(10_000.0, 10_000.0))
    cfg = PrepConfig.parse({
        "target": {"res": 1.0}, "output_dir": str(tmp_path / "o"),
        "rasters": [{"name": "a", "path": str(a)}, {"name": "b", "path": str(b)}]})
    with pytest.raises(PrepError, match="do not overlap"):
        cfg.target.resolve(cfg.rasters)


def test_crs_defaults_to_the_first_raster(scene):
    cfg = PrepConfig.parse(_cfg(scene))
    assert "32610" in str(cfg.target.resolve(cfg.rasters)["crs"])


# ───────────────────────────────────────────── config validation

def test_missing_res_rejected(scene):
    d = _cfg(scene); d["target"].pop("res")
    with pytest.raises(PrepError, match="'res' is required"):
        PrepConfig.parse(d)


def test_bad_kind_rejected(scene):
    d = _cfg(scene); d["rasters"][0]["kind"] = "raster"
    with pytest.raises(PrepError, match="kind must be one of"):
        PrepConfig.parse(d)


def test_unknown_key_rejected(scene):
    with pytest.raises(PrepError, match="unknown key"):
        PrepConfig.parse(_cfg(scene, typo=1))


def test_duplicate_names_rejected(scene):
    d = _cfg(scene); d["rasters"].append(dict(d["rasters"][0]))
    with pytest.raises(PrepError, match="duplicate"):
        PrepConfig.parse(d)


def test_empty_raster_list_rejected(scene):
    d = _cfg(scene); d["rasters"] = []
    with pytest.raises(PrepError, match="nothing to prepare"):
        PrepConfig.parse(d)


def test_missing_input_file_reported_by_name(scene):
    d = _cfg(scene); d["rasters"][0]["path"] = str(scene["dir"] / "gone.tif")
    with pytest.raises(PrepError, match="dem_pre.*not found"):
        run_prep(PrepConfig.parse(d))


def test_load_from_yaml(tmp_path, scene):
    p = tmp_path / "prep.yaml"
    p.write_text(yaml.safe_dump(_cfg(scene)))
    assert len(PrepConfig.load(p).rasters) == 3


# ───────────────────────────────────────────── fill discovery and reruns

def test_scan_finds_an_undeclared_fill(scene):
    info = scan_fills(scene["pre"])
    assert info["declared_nodata"] is None
    values = [e["value"] for e in info["most_common"]]
    assert 0.0 in values, "the 0 background should surface as a candidate fill"


def test_scan_only_writes_nothing(scene):
    cfg = PrepConfig.parse(_cfg(scene))
    out = run_prep(cfg, scan_only=True)
    assert "scan" in out
    assert not (scene["dir"] / "prep").exists()


def test_second_run_skips_existing(scene):
    cfg = PrepConfig.parse(_cfg(scene))
    run_prep(cfg)
    again = run_prep(cfg)
    assert again["written"] == [] and len(again["skipped"]) == 3


def test_overwrite_rewrites(scene):
    cfg = PrepConfig.parse(_cfg(scene))
    run_prep(cfg)
    again = run_prep(cfg, overwrite=True)
    assert len(again["written"]) == 3


def test_declared_nodata_is_honoured(tmp_path):
    arr = np.full((20, 20), 5.0, dtype="float32")
    arr[0:5, :] = -32768.0
    src = _raster(tmp_path / "n.tif", arr, nodata=-32768.0)
    cfg = PrepConfig.parse({
        "target": {"res": 1.0}, "output_dir": str(tmp_path / "o"),
        "rasters": [{"name": "n", "path": str(src), "kind": "dem"}]})
    out = run_prep(cfg)
    assert out["rasters"]["n"]["n_fill_masked"] == 100


def test_prepared_output_feeds_the_pipeline(scene):
    """Prep output must satisfy the pipeline's own grid check."""
    from pci_source_zones.raster import read_float, require_same_grid

    run_prep(PrepConfig.parse(_cfg(scene)))
    _, ref = read_float(scene["dir"] / "prep" / "dem_pre.tif")
    others = {}
    for name in ("dem_post", "dz"):
        _, prof = read_float(scene["dir"] / "prep" / f"{name}.tif")
        others[name] = prof
    require_same_grid(ref, others)      # raises GridMismatch if not identical
