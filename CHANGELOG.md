# Changelog

## Unreleased — pipeline consolidation

One command, one config schema, one definition of every quantity.

### Added
- `pci-zones` console entry point (`run`, `validate`, `prep`, `list`).
- Input contract check (`contract.py`): one grid, square pixels, declared
  nodata, and a source for every quantity the requested stages need. Reported by
  `validate`, enforced by `run`, all problems at once. Non-square pixels were
  previously checked nowhere — they never failed, they just made slope wrong.
- `prep` command: masks undeclared fill values and puts delivered rasters
  on one square grid. Rejects an elevation-style `valid_min` on a
  difference raster, and `--scan` reports candidate fills before writing.
- Canonical site config schema (`config/templates/site.yaml`); unknown keys are
  a hard error.
- Stage runner with resumability, per-area failure isolation, and a run manifest
  recording config and git SHA.
- Stages `terrain` (G, DeltaG from the DEM pair) and `stable` (stable-ground
  mask), so the pipeline runs from pre-DEM / post-DEM / dDEM.
- `physics/`, `evidence/`, `ml/` subpackages over a flat foundation.
- Invariant checks (`checks.py`) targeting silent-failure classes.
- CI on every push; 240 tests.

### Changed
- A config that both computes and supplies the same quantity (the stable
  mask, or terrain forcing) is now rejected. Previously the computed one
  silently won and the supplied file was ignored.
- **C3 detection limit is now one key.** `evidence.c3.lod` takes `auto`, a
  constant in metres, or a per-cell sigma raster, replacing the former `mode` /
  `threshold` / `sigma_path` / `sigma_scale` block. `auto` is the default and
  measures the LoD from stable ground, so C2 and C3 share one rule.
- NMAD is now median-centred, so a residual dDEM bias no longer inflates it.
- Flow routing is WhiteboxTools D-infinity everywhere; the Landlab D8 path and
  the `landlab` dependency are gone.
- Alignment is a pipeline stage, not a separate command or config file.
- `scipy` and `whitebox` are now declared dependencies (they were always
  required); `torch` moved to the `dl` extra.
- ML configs moved to `config/ml/`; the legacy physics config to `config/legacy/`.
- LoD output renamed `dg_lod.json` -> `lod.json` (it now holds both thresholds).

### Removed
- `compute_p_obs.py`, `compute_ceff.py`, `compute_dg_lod.py`,
  `compute_channel_mask.py`, `batch_*.py` — replaced by `pci-zones run`.
- `compute_terrain_deltas_1m.py` and `compute_stable_cells.py` are deprecation
  shims forwarding to the `terrain` and `stable` stages.
- `raster.read_match` (unused).


All notable changes to the pci-source-zones pipeline are recorded here.
Format: `[YYYY-MM-DD] — summary`

---

## [2026-08-04] — Source zone evidence pipeline completed

### Added
- `scripts/compute_dg_lod.py` — NMAD-based ΔG limit of detection from stable cells
  - Method: LoD = 2 × 1.4826 × median(|ΔG|) in geodetically stable cells (xDEM)
  - Result (Thomas/Montecito): **ΔG_LoD = 5.51** (n = 9,179,593 stable pixels)
  - Saves: `source_zone_labels/dg_lod.json`
- `scripts/compute_p_obs.py` — observed rill initiation probability at 10m
  - C2: ΔG > ΔG_LoD (flow network reorganized)
  - C3: dDEM < 0 (surface lowered, LoD-filtered)
  - Methods: intersection P(C2 ∩ C3) and union P(C2) + P(C3) − P(C2 ∩ C3)
  - Saves: `p_obs_intersection.tif`, `p_obs_union.tif`, `G_mean_10m.tif`, `G_star_10m.tif`
- `scripts/compute_ceff.py` — spatially variable soil resistance via Gamma inversion
  - Fits Gamma(k, scale) to G_mean on hillslopes (elev > 600m, G < 200)
  - Inverts p_obs cell-by-cell to recover local C_eff (Istanbulluoglu et al. 2002)
  - Result (Thomas/Montecito, p_obs_union): **k = 2.44, scale = 10.68, median C_eff = 41.9**
  - Saves: `C_eff_mean_10m.tif`, `C_eff_scale_10m.tif`, `ceff_fit.json`
- `config/source_zones.yaml` — unified config for all three scripts
- `config/experiments/exp01_intersection.yaml` — experiment override: intersection method
- `config/experiments/exp02_union.yaml` — experiment override: union method

### Changed
- A config that both computes and supplies the same quantity (the stable
  mask, or terrain forcing) is now rejected. Previously the computed one
  silently won and the supplied file was ignored.
- `compute_dg_lod.py`: replaced percentile method with NMAD (fixes mismatch with actual LoD used)
- `config/source_zones.yaml`: renamed `dg_lod_percentile` → `dg_lod_method: nmad`; added `dem_pre` input and `ceff:` section

---

## [2026-08-04] — UNet classify pipeline added

### Added
- `scripts/10_align_features.py` — align feature rasters to reference grid
- `scripts/11_make_binary_label.py` — threshold p_obs to binary label for UNet
- `scripts/12_run_unet_classify.py` — run trained UNet on full AOI
- `src/pci_source_zones/ml/unet_classify_workflow.py` — modular classify workflow

### Changed
- A config that both computes and supplies the same quantity (the stable
  mask, or terrain forcing) is now rejected. Previously the computed one
  silently won and the supplied file was ignored.
- `src/pci_source_zones/ml/unet_train.py` — updated training loop
- `src/pci_source_zones/ml/unet_workflow.py` — updated workflow integration
- `config/ml_unet.yaml`, `config/ml_random_forest.yaml` — config updates

---

## [2026-08-04] — FIS notebook updated (G_mean inversion)

### Changed
- A config that both computes and supplies the same quantity (the stable
  mask, or terrain forcing) is now rejected. Previously the computed one
  silently won and the supplied file was ignored.
- `notebooks/FIS_v1.ipynb` — switched from G_star (max) to G_mean (mean per 10m block)
  per Erkan's feedback; Gamma CDF plot now correctly shows terrain distribution (P→1),
  not empirical Pr; C_eff inversion uses p_obs_union directly

---

## [2026-07-31] — Initial ML pipeline

### Added
- `scripts/01–09`: terrain prep, C model, PCI, source extraction, validation, ML
- `src/pci_source_zones/ml/`: RF, LR, XGBoost, UNet model modules
- `config/ml_*.yaml`: ML experiment configs
- `topo_catalog/`: fire perimeter cataloging and LiDAR overlap scripts
- `dem_lidar_dl/`: USGS TNM and OpenTopography DEM/LiDAR download tools

---

## Run Log (key parameter records)

| Date | Script | Site | Key Result | Output |
|------|--------|------|-----------|--------|
| 2026-08-03 | compute_dg_lod.py | Thomas/Montecito | ΔG_LoD = 5.51, n_stable = 9,179,593 | dg_lod.json |
| 2026-08-03 | compute_p_obs.py | Thomas/Montecito | p_obs_union, G_mean_10m produced | source_zone_labels/ |
| 2026-08-04 | compute_ceff.py | Thomas/Montecito | k=2.44, scale=10.68, median C_eff=41.9 | C_eff_mean_10m.tif |
