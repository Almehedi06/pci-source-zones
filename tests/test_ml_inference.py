import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pci_source_zones.config import load_config
from pci_source_zones.ml.inference import (
    ModelBundle,
    _check_feature_match,
    _sliding_window_cfg,
    load_model_bundle,
    run_inference,
)
from pci_source_zones.ml.workflow import run_ml_workflow

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn is not installed",
)


def _profile(shape=(10, 10)):
    return {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 100, 10, 10),
        "nodata": -9999.0,
    }


def _write_site(root: Path, seed: int = 0) -> dict[str, Path]:
    """Write a small synthetic 'fire' with the rasters the config needs."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:10, 0:10]

    arrays = {
        "g": (xx * 5 + yy * 4 + 1).astype("float32"),
        "slope": (0.1 + yy / 20 + rng.normal(0, 0.01, (10, 10))).astype("float32"),
        "a": (10 + xx * 2).astype("float32"),
        "dem_diff": np.where((xx * 5 + yy * 4 + 1) >= 20, -1.0, -0.1).astype("float32"),
    }
    paths = {}
    for name, arr in arrays.items():
        p = root / f"{name}.tif"
        with rasterio.open(p, "w", **_profile()) as dst:
            dst.write(arr, 1)
        paths[name] = p
    return paths


def _train_config(tmp_path: Path, site: dict[str, Path]) -> Path:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        f"""
paths:
  output_dir: {tmp_path / "out"}
ml:
  output_subdir: ml/rf
  base_rasters:
    g: {site["g"]}
    slope: {site["slope"]}
    specific_area: {site["a"]}
    dem_diff: {site["dem_diff"]}
  target:
    type: physics_dod
    c_min: 20
    c_channel: 500
    erosion_threshold_m: -0.55
    nodata: 255
  features:
    numeric:
      - slope
      - drainage_area
  split:
    method: random
    seed: 2
    test_size: 0.2
  model:
    type: random_forest
    n_estimators: 5
    random_state: 2
    n_jobs: 1
  prediction:
    probability_threshold: 0.5
    exclude_channels: false
""",
        encoding="utf-8",
    )
    return cfg_path


def _inference_config(tmp_path: Path, site: dict[str, Path]) -> Path:
    """Same features, no target/split — what a new fire's config looks like."""
    cfg_path = tmp_path / "predict.yaml"
    cfg_path.write_text(
        f"""
paths:
  output_dir: {tmp_path / "out"}
ml:
  base_rasters:
    g: {site["g"]}
    slope: {site["slope"]}
    specific_area: {site["a"]}
  features:
    numeric:
      - slope
      - drainage_area
  prediction:
    probability_threshold: 0.5
""",
        encoding="utf-8",
    )
    return cfg_path


def test_predict_new_site_without_retraining(tmp_path: Path):
    train_site = _write_site(tmp_path / "fire_a", seed=0)
    trained = run_ml_workflow(load_config(_train_config(tmp_path, train_site)))

    # A different "fire": same feature definitions, different pixel values,
    # and crucially no target raster at all.
    new_site = _write_site(tmp_path / "fire_b", seed=99)
    out_dir = tmp_path / "predictions"

    result = run_inference(
        load_config(_inference_config(tmp_path, new_site)),
        trained["output_dir"],
        out_dir,
    )

    assert result["train_run_id"] == trained["run_id"]
    assert Path(result["probability"]).exists()
    assert Path(result["class"]).exists()
    assert Path(result["manifest"]).exists()

    with rasterio.open(result["probability"]) as src:
        probs = src.read(1, masked=True)
    assert probs.count() > 0
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_load_model_bundle_from_run_dir(tmp_path: Path):
    site = _write_site(tmp_path / "fire_a", seed=1)
    trained = run_ml_workflow(load_config(_train_config(tmp_path, site)))

    bundle = load_model_bundle(trained["output_dir"])
    assert bundle.kind == "sklearn"
    assert bundle.feature_names == ["slope", "drainage_area"]
    assert bundle.train_run_id == trained["run_id"]


def test_load_model_bundle_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "nope.joblib")


def _bundle(names: list[str]) -> ModelBundle:
    return ModelBundle(
        kind="sklearn", model=None, feature_names=names, is_regression=False, model_name="m"
    )


def test_feature_match_accepts_identical_list():
    _check_feature_match(_bundle(["slope", "G"]), ["slope", "G"])


def test_feature_match_rejects_missing_feature():
    with pytest.raises(ValueError, match="missing from config"):
        _check_feature_match(_bundle(["slope", "G"]), ["slope"])


def test_feature_match_rejects_extra_feature():
    with pytest.raises(ValueError, match="not seen during training"):
        _check_feature_match(_bundle(["slope"]), ["slope", "G"])


def test_feature_match_rejects_reordering():
    # Same set, different order — would silently map columns to wrong variables.
    with pytest.raises(ValueError, match="different order"):
        _check_feature_match(_bundle(["slope", "G"]), ["G", "slope"])


def _unet_bundle(inference_cfg):
    return ModelBundle(
        kind="unet",
        model=None,
        feature_names=["a"],
        is_regression=True,
        model_name="unet",
        inference_cfg=inference_cfg,
    )


def test_sliding_window_uses_trained_geometry_over_config(capsys):
    # A model trained at overlap=0.75 must not be run at the config's 0.5:
    # same weights, different window averaging, different output.
    cfg = {"ml": {"unet": {"patch_size": 128, "overlap": 0.5, "device": "cpu"}}}
    unet = _sliding_window_cfg(_unet_bundle({"patch_size": 128, "overlap": 0.75}), cfg)["ml"]["unet"]

    assert unet["overlap"] == 0.75
    assert unet["device"] == "cpu"  # non-geometry settings still honored
    assert "WARNING" in capsys.readouterr().out


def test_sliding_window_applies_trained_geometry_when_config_silent():
    unet = _sliding_window_cfg(_unet_bundle({"patch_size": 256, "overlap": 0.9}), {"ml": {}})["ml"]["unet"]
    assert unet == {"patch_size": 256, "overlap": 0.9}


def test_sliding_window_warns_for_legacy_bundle_without_geometry(capsys):
    cfg = {"ml": {"unet": {"patch_size": 64, "overlap": 0.25}}}
    unet = _sliding_window_cfg(_unet_bundle(None), cfg)["ml"]["unet"]

    assert unet["overlap"] == 0.25  # falls back to config
    assert "WARNING" in capsys.readouterr().out
