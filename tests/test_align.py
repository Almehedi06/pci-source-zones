"""Tests for config-driven raster alignment."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from pci_source_zones.align import AlignConfig, AlignError, run_alignment


def _raster(path, arr, res=10, origin=(500000, 5000000), crs="EPSG:32610"):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", nodata=-9999.0, crs=crs,
                transform=from_origin(*origin, res, res))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


@pytest.fixture
def scene(tmp_path):
    ref = _raster(tmp_path / "ref.tif", np.ones((20, 20), dtype="float32"))
    # a finer source on the same origin
    fine = _raster(tmp_path / "fine.tif",
                   np.arange(1600, dtype="float32").reshape(40, 40), res=5)
    cfg = {
        "reference": str(ref),
        "output_dir": str(tmp_path / "aligned"),
        "layers": [{"name": "fine", "path": str(fine), "resampling": "bilinear"}],
    }
    return tmp_path, cfg


# ───────────────────────────────────────────── parsing

def test_parse_minimal(scene):
    _, cfg = scene
    c = AlignConfig.parse(cfg)
    assert len(c.layers) == 1
    assert c.nodata == -9999.0
    assert c.suffix == "_aligned"


def test_unknown_key_rejected(scene):
    _, cfg = scene
    with pytest.raises(AlignError, match="unknown key"):
        AlignConfig.parse({**cfg, "typo": 1})


def test_missing_required_key_rejected(scene):
    _, cfg = scene
    d = dict(cfg); d.pop("reference")
    with pytest.raises(AlignError, match="reference"):
        AlignConfig.parse(d)


def test_bad_resampling_rejected(scene):
    _, cfg = scene
    cfg["layers"][0]["resampling"] = "magic"
    with pytest.raises(AlignError, match="unknown resampling"):
        AlignConfig.parse(cfg)


def test_duplicate_layer_names_rejected(scene):
    _, cfg = scene
    cfg["layers"].append(dict(cfg["layers"][0]))
    with pytest.raises(AlignError, match="duplicate"):
        AlignConfig.parse(cfg)


def test_categorical_layer_with_interpolation_warns(scene, capsys):
    """Interpolating class codes is almost always a mistake."""
    _, cfg = scene
    cfg["layers"][0]["name"] = "burn_severity"
    AlignConfig.parse(cfg)
    assert "looks categorical" in capsys.readouterr().out


def test_load_from_yaml(tmp_path, scene):
    _, cfg = scene
    p = tmp_path / "align.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert len(AlignConfig.load(p).layers) == 1


# ───────────────────────────────────────────── execution

def test_alignment_produces_reference_grid(scene):
    tmp, cfg = scene
    c = AlignConfig.parse(cfg)
    run_alignment(c)
    out = tmp / "aligned" / "fine_aligned.tif"
    assert out.exists()
    with rasterio.open(out) as s, rasterio.open(c.reference) as r:
        assert (s.height, s.width) == (r.height, r.width)
        assert s.transform == r.transform
        assert s.crs == r.crs


def test_missing_optional_layer_is_skipped(scene):
    tmp, cfg = scene
    cfg["layers"].append({"name": "absent", "path": str(tmp / "nope.tif")})
    summary = run_alignment(AlignConfig.parse(cfg))
    assert summary["missing"] == ["absent"]
    assert summary["n_written"] == 1


def test_missing_required_layer_raises(scene):
    tmp, cfg = scene
    cfg["layers"].append({"name": "absent", "path": str(tmp / "nope.tif"),
                          "required": True})
    with pytest.raises(AlignError, match="required layer"):
        run_alignment(AlignConfig.parse(cfg))


def test_second_run_skips_existing(scene):
    _, cfg = scene
    c = AlignConfig.parse(cfg)
    run_alignment(c)
    again = run_alignment(c)
    assert again["n_written"] == 0 and again["n_skipped"] == 1


def test_overwrite_rewrites(scene):
    _, cfg = scene
    c = AlignConfig.parse(cfg)
    run_alignment(c)
    again = run_alignment(c, overwrite=True)
    assert again["n_written"] == 1


def test_missing_reference_raises(scene, tmp_path):
    _, cfg = scene
    cfg["reference"] = str(tmp_path / "gone.tif")
    with pytest.raises(AlignError, match="reference grid not found"):
        run_alignment(AlignConfig.parse(cfg))


def test_aoi_clip_masks_outside(tmp_path, scene):
    """Cells outside the AOI must become nodata."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    tmp, cfg = scene
    # AOI covering only the upper-left quarter of the 20x20 @10m reference
    poly = box(500000, 5000000 - 100, 500000 + 100, 5000000)
    gdf = geopandas.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:32610")
    aoi = tmp / "aoi.shp"
    gdf.to_file(aoi)
    cfg["aoi"] = str(aoi)

    run_alignment(AlignConfig.parse(cfg))
    with rasterio.open(tmp / "aligned" / "fine_aligned.tif") as s:
        a = s.read(1)
        nod = a == s.nodata
    assert nod.any(), "some cells should fall outside the AOI"
    assert not nod.all(), "some cells should fall inside the AOI"


# ───────────────────────────────────────── alignment as a site-config stage

def _site_cfg(tmp_path, align):
    """Minimal valid site config with a given `align` section."""
    g = _raster(tmp_path / "in" / "G_pre.tif", np.ones((20, 20), dtype="float32"))
    return {
        "site": "t", "inputs": {"mode": "single", "g_pre": str(g),
                                "delta_g": str(g), "dz": str(g)},
        "outputs": {"root": str(tmp_path / "out")},
        "align": align,
    }


def test_site_config_accepts_inline_align(tmp_path, scene):
    from pci_source_zones.site_config import SiteConfig
    _, acfg = scene
    cfg = SiteConfig.parse(_site_cfg(tmp_path, acfg))
    assert len(cfg.align_config().layers) == 1


def test_site_config_accepts_align_path_relative_to_itself(tmp_path, scene):
    """A bare filename resolves next to the site config, not the cwd."""
    from pci_source_zones.site_config import SiteConfig
    _, acfg = scene
    sub = tmp_path / "sites"
    sub.mkdir()
    (sub / "a.yaml").write_text(yaml.safe_dump(acfg))
    site = sub / "site.yaml"
    site.write_text(yaml.safe_dump(_site_cfg(tmp_path, "a.yaml")))
    assert len(SiteConfig.load(site).align_config().layers) == 1


def test_site_config_rejects_bad_align_section(tmp_path, scene):
    from pci_source_zones.site_config import ConfigError, SiteConfig
    _, acfg = scene
    with pytest.raises(ConfigError, match="align"):
        SiteConfig.parse(_site_cfg(tmp_path, {**acfg, "typo": 1}))


def test_align_absent_is_none(tmp_path, scene):
    from pci_source_zones.site_config import SiteConfig
    d = _site_cfg(tmp_path, None)
    d.pop("align")
    assert SiteConfig.parse(d).align_config() is None


def test_align_is_a_stage_and_runs_once(tmp_path, scene):
    """The align stage is site-level: one result, not one per analysis area."""
    from pci_source_zones import runner
    from pci_source_zones.site_config import SiteConfig
    _, acfg = scene
    cfg = SiteConfig.parse(_site_cfg(tmp_path, acfg))
    results = runner.run(cfg, ["align"])
    assert [r.stage for r in results] == ["align"]
    assert results[0].status == "ok"
    assert results[0].detail["n_written"] == 1


def test_align_stage_skips_when_not_configured(tmp_path, scene):
    from pci_source_zones import runner
    from pci_source_zones.site_config import SiteConfig
    d = _site_cfg(tmp_path, None)
    d.pop("align")
    results = runner.run(SiteConfig.parse(d), ["align"])
    assert results[0].status == "skipped"


def test_align_listed_first_in_stages():
    from pci_source_zones import runner
    assert runner.STAGES[0] == "align"
    assert "align" in runner.SITE_STAGES
