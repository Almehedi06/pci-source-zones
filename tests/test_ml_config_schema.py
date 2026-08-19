from pathlib import Path

import pytest

from pci_source_zones.config import load_config
from pci_source_zones.ml.config_schema import validate_ml_config

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_CONFIGS = sorted((REPO_ROOT / "config").glob("ml_*.yaml"))


def _base_cfg() -> dict:
    return {
        "ml": {
            "base_rasters": {"slope": "slope.tif"},
            "target": {
                "type": "physics_dod",
                "c_min": 5,
                "c_channel": 120,
                "erosion_threshold_m": -0.55,
            },
            "features": {"numeric": ["slope"], "categorical": []},
            "split": {"method": "random", "seed": 42, "test_size": 0.2},
            "model": {"type": "random_forest"},
        }
    }


def test_valid_config_passes():
    out = validate_ml_config(_base_cfg())
    assert out["ml"]["target"]["type"] == "physics_dod"
    assert out["ml"]["model"]["type"] == "random_forest"


def test_missing_ml_section_raises():
    with pytest.raises(ValueError):
        validate_ml_config({})


def test_unknown_key_typo_raises():
    cfg = _base_cfg()
    cfg["ml"]["taget"] = {"type": "physics_dod"}  # typo of "target"
    with pytest.raises(ValueError):
        validate_ml_config(cfg)


def test_raster_continuous_requires_path():
    cfg = _base_cfg()
    cfg["ml"]["target"] = {"type": "raster_continuous"}
    with pytest.raises(ValueError, match="requires ml.target.path"):
        validate_ml_config(cfg)


def test_physics_dod_requires_c_min():
    cfg = _base_cfg()
    cfg["ml"]["target"] = {"type": "physics_dod"}
    with pytest.raises(ValueError, match="c_min"):
        validate_ml_config(cfg)


def test_empty_features_raises():
    cfg = _base_cfg()
    cfg["ml"]["features"] = {"numeric": [], "categorical": []}
    with pytest.raises(ValueError):
        validate_ml_config(cfg)


def test_missing_features_raises():
    cfg = _base_cfg()
    del cfg["ml"]["features"]
    with pytest.raises(ValueError):
        validate_ml_config(cfg)


def test_polygons_split_requires_train_ids():
    cfg = _base_cfg()
    cfg["ml"]["split"] = {
        "method": "polygons",
        "polygons": {"path": "x.gpkg", "id_field": "id"},
    }
    with pytest.raises(ValueError, match="train_ids"):
        validate_ml_config(cfg)


def test_polygons_split_valid():
    cfg = _base_cfg()
    cfg["ml"]["split"] = {
        "method": "polygons",
        "polygons": {"path": "x.gpkg", "id_field": "id", "train_ids": [1], "test_ids": [2]},
    }
    out = validate_ml_config(cfg)
    assert out["ml"]["split"]["polygons"]["train_ids"] == [1]


@pytest.mark.parametrize("config_path", ML_CONFIGS, ids=lambda p: p.name)
def test_real_ml_configs_pass_schema(config_path: Path):
    cfg = load_config(config_path)
    validate_ml_config(cfg)  # must not raise
