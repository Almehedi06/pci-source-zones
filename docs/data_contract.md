# Data Contract — ML Input Layers

This repo is a **consumer** of harmonized geospatial layers produced by upstream repos
(data-prep, Landlab/physics, p_obs computation). Everything below must exist in `data_dir`
before running any ML script.

This contract is enforced automatically: every `run_ml_workflow` / `run_tuning_workflow` /
`run_unet_workflow` call runs `ml.preflight.run_preflight()` first, which validates the
config against a schema (`ml.config_schema`) and then checks every raster the config
actually references — existence, CRS, resolution, and pixel grid alignment against a
reference raster — via `ml.data_contract.validate_data_contract()`, failing fast with a
single consolidated report instead of surfacing as a warning or a mid-training error.
Each run's output directory is timestamped (`.../ml/{model}/{run_id}/`) and contains a
`run_manifest.json` recording the exact config, git commit, and input-file fingerprints
that produced it.

## Validation splits

Both the tabular models and the UNet validate on spatially separated data, because
neighbouring pixels are strongly autocorrelated and a random split would leak:

- **RF / XGBoost** — `ml.tuning.cv.method: polygon_groups` gives leave-one-polygon-out CV
  over the train polygons (`cv.py`).
- **UNet** — `ml.unet.val_method` (default `auto`) holds out one whole train polygon's
  patches for validation. Because a patch is much wider than a pixel, any patch that
  merely *touches* the held-out polygon is dropped from training too, rather than being
  allowed to leak validation pixels in. Set `ml.unet.val_polygon_id` to choose which
  polygon is held out (default: whichever contains the most patches); `val_method: random`
  restores the old, spatially leaky behaviour and is only appropriate for non-spatial splits.

The test polygons are never touched by either mechanism — they are for the final
evaluation only.

## Hyperparameter tuning

| Models | Entry point | Search | CV |
|---|---|---|---|
| RF / LR / XGBoost | `scripts/08_tune_ml.py` | grid / random / optuna (`tuning.py`) | `polygon_groups`, `spatial_blocks`, k-fold |
| UNet | `scripts/14_tune_unet.py` | grid / random (`unet_tuning.py`) | leave-one-polygon-out |

The UNet tuner deliberately uses an explicit grid rather than a Bayesian optimiser: a
trial costs minutes on a GPU, and only a few hyperparameters matter at this data size.
Raster I/O and the patch pixel mask are computed once and reused across all trials;
only the patch datasets and the model are rebuilt per trial. `ml.tuning.max_epochs`
caps epochs during the sweep — retrain the winner uncapped with `scripts/09_run_unet.py`
using the emitted `unet_best.yaml`. Cost scales as `n_trials x cv.n_splits`, so rank
cheaply with `n_splits: 1` and re-run the shortlist with more folds.

## How paths resolve

All relative filenames in the config are resolved against `paths.data_dir`.
Absolute paths are used as-is (backward compatible).

Set `data_dir` in the YAML:
```yaml
paths:
  data_dir: /path/to/event/aligned
```
or override at runtime:
```bash
python scripts/07_run_ml.py  --config config/ml_random_forest.yaml  --data-dir /path/to/event/aligned
python scripts/09_run_unet.py --config config/ml_unet_regression.yaml --data-dir /path/to/event/aligned
```

---

## Required rasters (place in `data_dir`)

All rasters must share the same CRS, resolution (10 m), extent, and nodata value.
Reference grid: match slope/drainage_area output from terrain preprocessing.

| Filename | Description | Units / Range | Nodata |
|---|---|---|---|
| `slope_aligned.tif` | Terrain slope | m/m (0–∞) | NaN |
| `drainage_area_aligned.tif` | Specific catchment area | m | NaN |
| `G_aligned.tif` | Topographic driving index G = SCA × S² | m | NaN |
| `curvature_aligned.tif` | Plan curvature | 1/m | NaN |
| `aspect_aligned.tif` | Aspect | degrees 0–360 | NaN |
| `elevation_aligned.tif` | Elevation | m a.s.l. | NaN |
| `burn_severity_aligned.tif` | dNBR or RdNBR burn severity | dimensionless | NaN |
| `et_diff_aligned.tif` | Pre–post ET difference | mm/day | NaN |
| `ndvi_diff_aligned.tif` | Pre–post NDVI difference | dimensionless | NaN |
| `soil_thickness_aligned.tif` | Soil depth | m | NaN |
| `sand_total_aligned.tif` | Sand fraction | % | NaN |
| `silt_total_aligned.tif` | Silt fraction | % | NaN |
| `clay_total_aligned.tif` | Clay fraction | % | NaN |
| `porosity_aligned.tif` | Soil porosity | fraction 0–1 | NaN |
| `ksat_aligned.tif` | Saturated hydraulic conductivity | mm/hr | NaN |
| `ph_aligned.tif` | Soil pH | pH units | NaN |
| `field_capacity_aligned.tif` | Field capacity | fraction 0–1 | NaN |
| `precip_jan9_aligned.tif` | Storm precipitation (Jan 9) | mm | NaN |
| `precip_jan10_aligned.tif` | Storm precipitation (Jan 10) | mm | NaN |

### Regression targets (choose one per run)

| Filename | Description | Range | Nodata |
|---|---|---|---|
| `p_obs_union_aligned.tif` | Fraction of 1m cells with C2 OR C3 per 10m block | 0–1 | NaN |
| `p_obs_inter_aligned.tif` | Fraction of 1m cells with C2 AND C3 per 10m block | 0–1 | NaN |
| `C_p_obs_union_log_aligned.tif` | log(C_eff) from Gamma inversion of p_obs_union | log(Pa) | NaN |

### Tobit auxiliary layers (UNet tobit mode only)

| Filename | Description | Range | Nodata |
|---|---|---|---|
| `censored_aligned.tif` | Binary mask: 1 = p_obs=0 hillslope pixel (left-censored) | 0/1 | NaN |
| `logC_bound_aligned.tif` | Lower bound on log(C_eff) for censored pixels | log(Pa) | NaN |

### Train/test split polygon

| Filename | Description |
|---|---|
| `train_test_poly1.gpkg` | GeoPackage with integer `id` field; polygon IDs [1–6]=train, [7–8]=test |

This file can live in `data_dir` or be specified as an absolute path in the config.

---

## CRS and grid requirements

- **CRS**: match the DEM CRS for the event (typically UTM or state plane)
- **Resolution**: 10 m × 10 m
- **Nodata**: NaN (float rasters) or event-specific integer (label rasters)
- **Grid alignment**: all layers must snap to the same pixel grid (same transform origin).
  Run `scripts/10_align_features.py` from the data-prep repo to enforce this.

---

## Checklist for a new fire event

```
[ ] Run terrain preprocessing → slope_aligned.tif, drainage_area_aligned.tif, G_aligned.tif
[ ] Run p_obs computation    → p_obs_union_aligned.tif, p_obs_inter_aligned.tif
[ ] Run C_eff inversion      → C_p_obs_union_log_aligned.tif
[ ] Run tobit prep           → censored_aligned.tif, logC_bound_aligned.tif  (UNet only)
[ ] Align all feature layers → curvature, aspect, elevation, burn_severity, ...
[ ] Draw train/test polygons → train_test_poly1.gpkg
[ ] Copy config/*.yaml, set paths.data_dir to new event's aligned/ folder
[ ] python scripts/07_run_ml.py  --config config/ml_random_forest.yaml
[ ] python scripts/09_run_unet.py --config config/ml_unet_regression.yaml
```
