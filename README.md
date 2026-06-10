# pci-source-zones

Small, config-driven tools for postfire channel-initiation probability (PCI)
and candidate source polygons for Landlab MWR-style workflows.

The core calculation is intentionally simple:

```text
G = a S^alpha
PCI = P(C <= G)
```

where `G` is terrain forcing and `C` is the modeled resistance threshold from
grain size, roughness, and rainfall-excess inputs.

## Install

Use the existing `ml_debris` environment on this machine:

```bash
conda activate ml_debris
pip install -e .
```

Or run scripts directly from the repo root by setting `PYTHONPATH=src`.

## Quick Start

1. Put a DEM at `data/raw/dem_pre.tif` or edit `config/montecito.yaml`.
2. Edit the simple constants in `config/montecito.yaml`.
3. Run the full workflow:

```bash
conda run -n ml_debris python scripts/run_all.py --config config/montecito.yaml
```

Step-by-step:

```bash
conda run -n ml_debris python scripts/01_prepare_terrain.py --config config/montecito.yaml
conda run -n ml_debris python scripts/02_compute_c.py --config config/montecito.yaml
conda run -n ml_debris python scripts/03_compute_pci.py --config config/montecito.yaml
conda run -n ml_debris python scripts/04_extract_sources.py --config config/montecito.yaml
conda run -n ml_debris python scripts/05_validate.py --config config/montecito.yaml
```

Focused source-area workflow:

```bash
conda run -n ml_debris python scripts/06_make_source_areas.py --config config/montecito.yaml
```

This workflow computes `PCI = P(C <= G)`, filters candidate cells by observed
erosion from `paths.dem_diff`, then exports top-percent source-cell rasters and
GeoPackage polygons. It also exports deterministic C-band source candidates:

```text
candidate = G >= Cmin
candidate = G >= Cmin and G < Cmax
candidate = G >= Cmin and G < Cmax and DoD < erosion_threshold_m
```

These controls live in `config/montecito.yaml`:

```yaml
source_area:
  use_dod_filter: true
  erosion_threshold_m: -0.277
  top_percent_values: [10, 5]
  c_min_values: [5, 10, 20, 30, 50, 70]
  c_channel_max: 120
```

## Outputs

- `data/processed/slope.tif`
- `data/processed/specific_catchment_area.tif`
- `data/processed/topographic_driving_index.tif`
- `data/outputs/c_model_mean.tif`
- `data/outputs/pci.tif`
- `data/outputs/source_polygons.gpkg`
- `<output_dir>/source_area_workflow/pci_from_c_cdf.tif`
- `<output_dir>/source_area_workflow/top5_pci_dod_source_binary.tif`
- `<output_dir>/source_area_workflow/top5_pci_dod_source_polygons.gpkg`
- `<output_dir>/source_area_workflow/cmin20_lt_cmax120_g_dod_source_binary.tif`
- `<output_dir>/source_area_workflow/cmin20_lt_cmax120_g_dod_source_polygons.gpkg`
- `<output_dir>/source_area_workflow/c_band_source_cell_summary.csv`
- `<output_dir>/source_area_workflow/c_band_source_polygon_summary.csv`

## Input Modes

`d50`, `na`, and `rainfall_excess` support:

- `constant`
- `distribution`
- `raster`

Point/table interpolation is intentionally left out of v1.

## Notes

- `d50` can be supplied in `mm` or `m`.
- `rainfall_excess` can be supplied in `mm/hr` or `m/s`.
- `na` is Manning roughness and is used as supplied.
- For a deterministic run, set all inputs to `constant`.
- For PCI, use at least one `distribution` input.
