# Random Forest Hyperparameter Tuning — Technical Reference

## Overview

Goal: tune RF regressor HPs to maximize CV R² on `log1p(C_eff)` prediction across 6 train polygons (leave-one-polygon-out CV).

**Machine:** i9-13950HX, 32 cores, 62 GB RAM (WSL2 on Windows)

---

## Key Files

| File | Role |
|------|------|
| `config/ml_random_forest_tune.yaml` | Tuning config (search space, CV, paths) |
| `scripts/08_tune_ml.py` | Tuning entry point (NOT `07_run_ml.py`) |
| `src/pci_source_zones/ml/tuning.py` | Core tuning logic |
| `tuning.log` | Runtime log (written by nohup run) |
| Output: `tuning_results.json` | Best params + best CV score |
| Output: `tuning_cv_results.csv` | All combinations ranked |

Output dir: `/mnt/c/Users/amehedi/Downloads/source_area_workflow/ml/random_forest_regressor_tuned/{run_id}/`
(each run gets its own timestamped subdirectory now — see `docs/data_contract.md` — so
rerunning never overwrites a previous tuning run's results)

---

## Config Summary (`ml_random_forest_tune.yaml`)

```yaml
model:
  type: random_forest_regressor
  random_state: 42
  n_jobs: 1                  # per-model cores — 1 to maximize parallel trials

tuning:
  search_method: random_search
  scoring: r2
  n_iter: 100                # number of HP combinations to try
  random_state: 42
  n_jobs: 32                 # parallel trials (= all cores)
  dataset_cache: true

  cv:
    method: polygon_groups   # leave-one-polygon-out
    n_splits: 6              # 6 train polygons → 6 folds

  search_space:
    n_estimators:     [20, 30, 50, 80, 100, 150, 200, 250, 300, 500]
    max_depth:        [null, 8, 15, 25, 40]
    min_samples_leaf: [1, 2, 5, 10, 20]
    max_features:     [0.2, 0.3, 0.4, 0.5, sqrt]
    criterion:        [squared_error, absolute_error, poisson]
    bootstrap:        [true, false]
    max_samples:      [null, 0.6, 0.7, 0.8, 0.9]
```

**Core allocation logic:**
- `model n_jobs=1` × `tuning n_jobs=32` = 32 cores total (machine maxed)
- Recommended over `n_jobs=4` × `n_jobs=8` for RF because parallelizing trials beats parallelizing within a single tree

---

## CV Strategy

- Method: polygon leave-one-out (6 folds)
- Each fold: train on 5 polygons, validate on 1
- All 100 HP combinations are evaluated across all 6 folds = 600 RF fits total
- CV R² is the mean across 6 folds — honest estimate, matches held-out test performance (polys 7–8)
- Validation set (10%) excluded from tuning (`include_validation_in_tuning: false`)

---

## bootstrap / max_samples Constraint

**Problem:** `RandomizedSearchCV` samples HPs independently — it can draw `bootstrap=False` with `max_samples=0.8`, which is invalid (sklearn raises error).

**Solution implemented in `tuning.py` → `_build_constrained_param_list()`:**

1. Use `ParameterSampler` to draw `n_iter × 5` raw combinations
2. For any combination where `bootstrap=False`, force `max_samples=None`
3. Deduplicate, keep first `n_iter` valid combinations
4. Wrap each scalar value in a list: `{k: [v] for k, v in p.items()}`
5. Pass the list of dicts to `GridSearchCV` (not `RandomizedSearchCV`)

Key: `GridSearchCV` requires list-wrapped values — scalars cause `TypeError`.

---

## Features (13 numeric, 0 categorical)

```
G, slope, drainage_area, curvature, aspect, elevation,
et_diff, ndvi_diff, soil_thickness, sand_total, silt_total,
ksat, ph, precip_jan9
```

Excluded (comments in config): `burn_severity` (zero importance), `clay_total` (collinear), `porosity` / `field_capacity` (4 discrete values only), `precip_jan10` (r=1.00 with jan9).

**Target:** `log1p(C_eff)` at 10m — Gamma-inverted soil resistance from 1m LiDAR dDEM.

---

## How to Run

```bash
# From ~/pci-source-zones (critical — must be in correct directory)
nohup conda run -n ml_debris python -u scripts/08_tune_ml.py \
  --config config/ml_random_forest_tune.yaml > tuning.log 2>&1 &
```

**Monitor:**
```bash
tail -f tuning.log
ps aux | grep 08_tune | grep -v grep
```

**Prevent sleep (Windows):** Settings → Power → Sleep → Never  
**Or WSL:** wrap command with `systemd-inhibit --what=sleep`

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `unrecognized arguments: --tune` | Used `07_run_ml.py` instead of `08_tune_ml.py` | Use correct script |
| `TypeError: Parameter grid for 'n_estimators' needs to be a list` | `GridSearchCV` received scalar values | Wrap with `{k: [v] for k, v in p.items()}` |
| Process dies overnight, no results | PC went to sleep | Use `nohup` + set sleep to Never |
| Log shows only `nohup: ignoring input` | Normal nohup message + Python output buffered | Use `python -u` flag for unbuffered output |
| Two duplicate processes | Ran command twice | `ps aux | grep 08_tune`, kill older PID |

---

## After Tuning Completes

1. Open `tuning_results.json` — note `best_params` and `best_score` (CV R²)
2. Update `config/ml_random_forest.yaml` with best HPs
3. Re-run full pipeline: `python scripts/07_run_ml.py --config config/ml_random_forest.yaml`
4. Compare test R² (polys 7–8) against previous best (0.078)

---

## Current Baseline Performance

| Model | Test R² | CV R² |
|-------|---------|-------|
| RF (untuned) | 0.078 | ~0.06 |
| XGBoost | 0.055 | — |
| UNet | −0.095 | — |

RF is best. Tuning goal: push test R² > 0.10. Fundamental ceiling limited by 6 train polygons — more polygons will help more than HP tuning.

---

## Potential Next Steps

- **Optuna** (`search_method: optuna`) — Bayesian search, smarter than random, fewer iterations needed
- **SHAP / permutation importance** — understand which features drive predictions
- **Residual kriging** — spatial interpolation of RF residuals
- **More training polygons** — fundamental R² fix (currently only 6)
- **Alternative targets** — `frac_c2c3` (confirmed rill initiation label) instead of `log1p(C_eff)`
