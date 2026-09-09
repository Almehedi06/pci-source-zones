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


def test_every_script_config_default_exists():
    """A --config default pointing at a moved file breaks only at runtime.

    Relocating the ML configs once already broke eight scripts this way, so the
    defaults are now checked here rather than discovered by a failed run.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    missing = []
    for script in sorted((root / "scripts").glob("*.py")):
        for ref in re.findall(r"config/[\w/]+\.yaml", script.read_text()):
            if not (root / ref).exists():
                missing.append(f"{script.name} -> {ref}")
    assert not missing, "scripts reference configs that do not exist:\n  " + \
        "\n  ".join(missing)


def test_prep_scan_writes_nothing(tmp_path, capsys):
    """--scan must report and stop, never touch the output directory."""
    import numpy as np, rasterio
    from rasterio.transform import from_origin

    src = tmp_path / "a.tif"
    arr = np.full((30, 30), 100.0, dtype="float32"); arr[:10] = 0.0
    with rasterio.open(src, "w", driver="GTiff", height=30, width=30, count=1,
                       dtype="float32", crs="EPSG:32610",
                       transform=from_origin(0, 30, 1, 1)) as d:
        d.write(arr, 1)

    cfg = tmp_path / "prep.yaml"
    cfg.write_text(yaml.safe_dump({
        "target": {"res": 2.0}, "output_dir": str(tmp_path / "out"),
        "rasters": [{"name": "a", "path": str(src), "kind": "dem"}]}))

    assert cli.main(["prep", "--config", str(cfg), "--scan"]) == 0
    assert "likely fill" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_prep_bad_config_exits_two(tmp_path, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"target": {"res": 2.0},
                                   "output_dir": str(tmp_path),
                                   "rasters": [{"name": "d", "path": "x.tif",
                                                "kind": "diff", "valid_min": 0}]}))
    assert cli.main(["prep", "--config", str(cfg)]) == 2
    assert "valid_min is not allowed" in capsys.readouterr().err
