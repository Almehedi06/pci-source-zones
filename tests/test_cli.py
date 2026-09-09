"""CLI surface. Thin, but untested plumbing is how a stale attribute ships."""
from __future__ import annotations

import pytest
import yaml

from pci_source_zones import cli

REPO_CONFIGS = ["montecito_2016_2018", "eaton_2016_2025", "eagle_creek_2018_2021"]


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({
        "site": "t",
        "inputs": {"mode": "single", "g_pre": "g.tif", "delta_g": "d.tif",
                   "dz": "z.tif"},
        "outputs": {"root": str(tmp_path / "out")},
    }))
    return p


def test_validate_runs_and_reports(cfg_path, capsys):
    assert cli.main(["validate", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    for line in ("config OK", "grid", "c3 lod", "terrain", "output root"):
        assert line in out


@pytest.mark.parametrize("name", REPO_CONFIGS)
def test_every_shipped_site_config_validates(name, capsys):
    """A config in config/sites/ that does not parse is a broken release."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "config" / "sites" / f"{name}.yaml"
    assert p.exists(), f"{p} is missing"
    assert cli.main(["validate", "--config", str(p)]) == 0


def test_list_reports_a_single_site(cfg_path, capsys):
    assert cli.main(["list", "--config", str(cfg_path)]) == 0
    assert "single site" in capsys.readouterr().out


def test_bad_config_exits_two_not_traceback(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"site": "t", "typo": 1,
                                 "inputs": {"mode": "single", "dz": "z.tif"},
                                 "outputs": {"root": str(tmp_path)}}))
    assert cli.main(["validate", "--config", str(p)]) == 2
    assert "config error" in capsys.readouterr().err


def test_unknown_stage_is_rejected(cfg_path):
    with pytest.raises(ValueError, match="unknown stage"):
        cli.main(["run", "--config", str(cfg_path), "--stages", "bogus"])


def test_align_is_a_stage_not_a_verb():
    """Regression: alignment must not reappear as its own command."""
    import argparse
    with pytest.raises(SystemExit):
        cli.main(["align", "--config", "x.yaml"])
