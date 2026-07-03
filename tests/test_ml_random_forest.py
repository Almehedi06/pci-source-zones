from pathlib import Path

import importlib.util

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pci_source_zones.config import load_config
from pci_source_zones.ml.tuning import run_tuning_workflow
from pci_source_zones.ml.workflow import run_ml_workflow


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn is not installed",
)


def test_random_forest_pipeline_smoke(tmp_path: Path):
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 100, 10, 10),
        "nodata": -9999.0,
    }
    yy, xx = np.mgrid[0:10, 0:10]
    g = (xx * 5 + yy * 4 + 1).astype("float32")
    slope = (0.1 + yy / 20).astype("float32")
    a = (10 + xx * 2).astype("float32")
    dem_diff = np.where((g >= 10) & (g < 60), -1.0, -0.1).astype("float32")
    soil = (xx % 3).astype("float32")

    paths = {}
    for name, arr in {
        "g": g,
        "slope": slope,
        "a": a,
        "dem_diff": dem_diff,
        "soil": soil,
    }.items():
        path = tmp_path / f"{name}.tif"
        paths[name] = path
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
paths:
  output_dir: {tmp_path / "out"}
ml:
  output_subdir: ml/rf
  base_rasters:
    g: {paths["g"]}
    slope: {paths["slope"]}
    specific_area: {paths["a"]}
    dem_diff: {paths["dem_diff"]}
  target:
    type: physics_dod
    c_min: 10
    c_channel: 60
    erosion_threshold_m: -0.55
    nodata: 255
  feature_paths:
    soil_texture: {paths["soil"]}
  features:
    numeric:
      - slope
      - drainage_area
    categorical:
      - soil_texture
  split:
    method: random
    seed: 2
    test_size: 0.2
    val_size: 0.1
    negative_to_positive_ratio: 2
  model:
    n_estimators: 5
    random_state: 2
    n_jobs: 1
  prediction:
    probability_threshold: 0.5
    exclude_channels: true
""",
        encoding="utf-8",
    )

    result = run_ml_workflow(load_config(cfg_path), model_name="random_forest")
    assert result["probability"].exists()
    assert result["class"].exists()
    assert result["metrics"].exists()
    assert result["feature_scores"].exists()


def test_logistic_regression_pipeline_smoke(tmp_path: Path):
    profile = {
        "driver": "GTiff",
        "height": 12,
        "width": 12,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 120, 10, 10),
        "nodata": -9999.0,
    }
    yy, xx = np.mgrid[0:12, 0:12]
    g = (xx * 4 + yy * 3 + 2).astype("float32")
    slope = (0.05 + yy / 25).astype("float32")
    a = (8 + xx).astype("float32")
    dem_diff = np.where((g >= 12) & (g < 55), -1.0, -0.1).astype("float32")
    soil = (xx % 2).astype("float32")

    paths = {}
    for name, arr in {
        "g": g,
        "slope": slope,
        "a": a,
        "dem_diff": dem_diff,
        "soil": soil,
    }.items():
        path = tmp_path / f"{name}.tif"
        paths[name] = path
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
paths:
  output_dir: {tmp_path / "out"}
ml:
  output_subdir: ml/{{model}}
  base_rasters:
    g: {paths["g"]}
    slope: {paths["slope"]}
    specific_area: {paths["a"]}
    dem_diff: {paths["dem_diff"]}
  target:
    type: physics_dod
    c_min: 12
    c_channel: 55
    erosion_threshold_m: -0.55
    nodata: 255
  feature_paths:
    soil_texture: {paths["soil"]}
  features:
    numeric:
      - slope
      - drainage_area
    categorical:
      - soil_texture
  split:
    method: random
    seed: 2
    test_size: 0.2
    val_size: 0.1
    negative_to_positive_ratio: 2
  model:
    type: logistic_regression
    max_iter: 200
    class_weight: balanced
  prediction:
    probability_threshold: 0.5
    exclude_channels: true
""",
        encoding="utf-8",
    )

    result = run_ml_workflow(load_config(cfg_path))
    assert result["model_name"] == "logistic_regression"
    assert result["probability"].exists()
    assert result["class"].exists()
    assert result["metrics"].exists()
    assert result["feature_scores"].exists()


def test_random_forest_tuning_spatial_blocks_smoke(tmp_path: Path):
    profile = {
        "driver": "GTiff",
        "height": 20,
        "width": 20,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 200, 10, 10),
        "nodata": -9999.0,
    }
    yy, xx = np.mgrid[0:20, 0:20]
    g = (xx * 4 + yy * 3 + 1).astype("float32")
    slope = (0.1 + yy / 30).astype("float32")
    a = (10 + xx).astype("float32")
    dem_diff = np.where((g >= 12) & (g < 70), -1.0, -0.1).astype("float32")
    soil = (xx % 3).astype("float32")

    paths = {}
    for name, arr in {
        "g": g,
        "slope": slope,
        "a": a,
        "dem_diff": dem_diff,
        "soil": soil,
    }.items():
        path = tmp_path / f"{name}.tif"
        paths[name] = path
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)

    cfg_path = tmp_path / "tune_config.yaml"
    cfg_path.write_text(
        f"""
paths:
  output_dir: {tmp_path / "out"}
ml:
  output_subdir: ml/{{model}}
  base_rasters:
    g: {paths["g"]}
    slope: {paths["slope"]}
    specific_area: {paths["a"]}
    dem_diff: {paths["dem_diff"]}
  target:
    type: physics_dod
    c_min: 12
    c_channel: 70
    erosion_threshold_m: -0.55
    nodata: 255
  feature_paths:
    soil_texture: {paths["soil"]}
  features:
    numeric:
      - slope
      - drainage_area
    categorical:
      - soil_texture
  split:
    method: random
    seed: 3
    test_size: 0.2
    val_size: 0.0
    negative_to_positive_ratio: 2
  model:
    type: random_forest
    random_state: 3
    n_jobs: 1
  tuning:
    search_method: random_search
    scoring: mcc
    n_iter: 2
    random_state: 3
    n_jobs: 1
    cv:
      method: spatial_blocks
      n_splits: 2
      block_size_m: 40
      seed: 3
    search_space:
      n_estimators: [5, 8]
      max_depth: [null, 4]
      min_samples_leaf: [1, 2]
  prediction:
    probability_threshold: 0.5
    exclude_channels: true
""",
        encoding="utf-8",
    )

    result = run_tuning_workflow(load_config(cfg_path))
    assert result["probability"].exists()
    assert result["class"].exists()
    assert result["metrics"].exists()
    assert result["best_params"].exists()
    assert result["cv_results"].exists()
