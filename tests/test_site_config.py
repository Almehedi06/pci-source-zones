"""Tests for the canonical site configuration schema."""
from __future__ import annotations

import textwrap

import pytest
import yaml

from pci_source_zones.site_config import (
    ConfigError,
    SiteConfig,
    deep_merge,
)

MINIMAL = {
    "site": "testfire",
    "epochs": {"pre": 2016, "post": 2025},
    "inputs": {"mode": "single", "g_pre": "/g.tif",
               "delta_g": "/dg.tif", "dz": "/dz.tif"},
    "outputs": {"root": "/out"},
}


def cfg(**patch):
    d = deep_merge(MINIMAL, patch)
    return SiteConfig.parse(d)


# ─────────────────────────────────────────── defaults and accessors

def test_minimal_config_parses_with_defaults():
    c = cfg()
    assert c.site == "testfire"
    assert c.block_size == 10               # 10 m / 1 m
    assert c.evidence.domain.g_min == 5.0
    assert c.evidence.domain.g_channel == 300.0
    assert c.ceff.p_ceiling == 0.05
    assert c.lod.scale == 2.0
    assert not c.is_batch


def test_block_size_derived_from_resolutions():
    assert cfg(grid={"fine_res_m": 1, "coarse_res_m": 30}).block_size == 30
    assert cfg(grid={"fine_res_m": 2, "coarse_res_m": 10}).block_size == 5


def test_non_integer_block_ratio_rejected():
    with pytest.raises(ConfigError, match="integer multiple"):
        cfg(grid={"fine_res_m": 3, "coarse_res_m": 10})


def test_single_mode_input_path():
    c = cfg()
    assert str(c.input_path("delta_g")) == "/dg.tif"
    assert c.input_path("aoi") is None


def test_output_dir_single_and_batch():
    c = cfg()
    assert str(c.output_dir()) == "/out"
    b = cfg(inputs={"mode": "batch", "roots": {"delta_g": "/r"},
                    "templates": {"delta_g": "{id}/dg.tif"}},
            outputs={"root": "/out", "layout": "area_{id}"})
    assert str(b.output_dir("7")) == "/out/area_7"


def test_batch_input_path_templating():
    b = cfg(inputs={"mode": "batch",
                    "roots": {"delta_g": "/data"},
                    "templates": {"delta_g": "{id}/DeltaG.tif"}})
    assert str(b.input_path("delta_g", "12")) == "/data/12/DeltaG.tif"


def test_batch_input_path_without_id_raises():
    b = cfg(inputs={"mode": "batch", "roots": {"delta_g": "/d"},
                    "templates": {"delta_g": "{id}/x.tif"}})
    with pytest.raises(ConfigError, match="requires an area id"):
        b.input_path("delta_g")


# ─────────────────────────────────────────── strict validation

def test_unknown_root_key_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        SiteConfig.parse({**MINIMAL, "typo_section": {}})


def test_unknown_nested_key_rejected():
    """A mistyped threshold must fail loudly, not be silently ignored."""
    with pytest.raises(ConfigError, match="unknown key"):
        cfg(evidence={"domain": {"g_minn": 5}})


def test_missing_required_key_rejected():
    d = dict(MINIMAL)
    d.pop("outputs")
    with pytest.raises(ConfigError, match="outputs"):
        SiteConfig.parse(d)


def test_single_mode_allows_stage_specific_inputs():
    c = SiteConfig.parse({**MINIMAL,
                          "inputs": {"mode": "single", "dem_pre": "/pre.tif",
                                     "dem_post": "/post.tif"}})
    assert c.inputs.dz is None


def test_single_mode_requires_some_input_path():
    with pytest.raises(ConfigError, match="at least one input"):
        SiteConfig.parse({**MINIMAL, "inputs": {"mode": "single"}})


def test_batch_template_must_contain_id():
    with pytest.raises(ConfigError, match=r"\{id\}"):
        cfg(inputs={"mode": "batch", "roots": {"delta_g": "/d"},
                    "templates": {"delta_g": "fixed_name.tif"}})


def test_bad_domain_bounds_rejected():
    with pytest.raises(ConfigError, match="must be <"):
        cfg(evidence={"domain": {"g_min": 400, "g_channel": 300}})


def test_bad_p_ceiling_rejected():
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ConfigError, match="p_ceiling"):
            cfg(ceff={"p_ceiling": bad})


def test_bad_lod_scale_rejected():
    with pytest.raises(ConfigError, match="scale"):
        cfg(lod={"scale": 0})


def test_unknown_p_obs_method_rejected():
    with pytest.raises(ConfigError, match="p_obs_methods"):
        cfg(ceff={"p_obs_methods": ["intersection", "bogus"]})


def test_c3_lod_defaults_to_auto():
    assert cfg().evidence.c3.is_auto


def test_c3_lod_accepts_all_three_forms():
    assert cfg(evidence={"c3": {"lod": 0}}).evidence.c3.lod == 0
    assert cfg(evidence={"c3": {"lod": 0.175}}).evidence.c3.lod == 0.175
    c = cfg(evidence={"c3": {"lod": "/s.tif"}}).evidence.c3
    assert c.raster_path == "/s.tif" and not c.is_auto


def test_c3_negative_lod_rejected():
    with pytest.raises(ConfigError, match="lod must be >= 0"):
        cfg(evidence={"c3": {"lod": -1}})


def test_c3_nonsense_lod_rejected():
    with pytest.raises(ConfigError, match="number, 'auto', or a raster path"):
        cfg(evidence={"c3": {"lod": "sometimes"}})


def test_c3_old_mode_key_is_a_hard_error():
    """The retired `mode` key must fail loudly, never be silently ignored."""
    with pytest.raises(ConfigError, match="unknown key"):
        cfg(evidence={"c3": {"mode": "raw_threshold"}})


def test_bad_c3_scale_rejected():
    with pytest.raises(ConfigError, match="scale must be > 0"):
        cfg(evidence={"c3": {"lod": 0, "scale": 0}})


def test_bad_domain_slope_rejected():
    with pytest.raises(ConfigError, match="slope_max_deg"):
        cfg(evidence={"domain": {"slope_max_deg": 95}})


# ─────────────────────────────────────────── file loading

def test_load_from_yaml(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump(MINIMAL))
    c = SiteConfig.load(p)
    assert c.site == "testfire"
    assert c.source_path == p


def test_load_with_overrides(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump(MINIMAL))
    c = SiteConfig.load(p, overrides={"evidence": {"domain": {"g_channel": 120}}})
    assert c.evidence.domain.g_channel == 120
    assert c.evidence.domain.g_min == 5.0      # untouched


def test_deep_merge_preserves_siblings():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = deep_merge(base, {"a": {"y": 99}})
    assert out == {"a": {"x": 1, "y": 99}, "b": 3}


def test_as_dict_includes_block_size():
    d = cfg().as_dict()
    assert d["grid"]["block_size"] == 10
    assert "source_path" not in d
