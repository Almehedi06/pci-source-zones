# pci-source-zones

Config-driven pipeline for postfire debris-flow source zones: LiDAR change
evidence, observed erosion probability, channel-initiation threshold, and the
machine-learning models trained on them.

**A new fire is a config file, not a code change.**

## Install

```bash
conda activate ml_debris
pip install -e .
```

Or run from the repo root with `PYTHONPATH=src`.

## Quick start

```bash
# check a config without running anything
pci-zones validate --config config/sites/montecito_2016_2018.yaml

# run the whole evidence pipeline
pci-zones run --config config/sites/montecito_2016_2018.yaml

# selected stages, selected analysis areas, forcing recomputation
pci-zones run --config config/sites/eagle_creek_2018_2021.yaml \
    --stages evidence,p_obs --ids 1-5,12 --overwrite

# just the alignment stage
pci-zones run --config config/sites/montecito_2016_2018.yaml \
    --stages align
```

`--ids` is the only difference between single-site and batch. Single-site is a
batch of one, through the same code path.

## Stages

| Stage | Produces |
|---|---|
| `align` | predictor rasters on a common reference grid (site-level, optional) |
| `terrain` | G for both epochs and DeltaG, from the pre/post DEM pair |
| `stable` | the stable-ground mask the detection threshold is estimated from |
| `lod` | detection threshold for the change signal, from stable ground |
| `evidence` | C2 / C3 binary evidence layers at the fine resolution |
| `p_obs` | observed erosion probability and terrain forcing at the modelling resolution |
| `ceff` | Gamma fit and per-cell initiation threshold |

Each stage skips work whose outputs already exist, isolates failures per
analysis area, and contributes to a run summary and manifest.

The pipeline's real inputs are **three rasters: a pre-event DEM, a post-event
DEM, and a dDEM**. Everything else — G, DeltaG, the stable mask, tau — is derived
by a stage. A config that already has G and DeltaG can supply them directly under
`inputs`, and the `terrain` stage then reports itself as not configured and skips.

Two things about the derived stages are worth knowing:

- **`evidence.c3.lod`** is the detection limit for surface lowering, and takes
  `auto` (measure it from stable ground — the default, and the only choice with
  no site-specific number in it), a constant in metres, or a per-cell sigma
  raster. C2 and C3 then share one rule for "bigger than the noise".
- **`stable.method`** defaults to `criteria`, which builds the mask from
  predictors independent of the change signal. `threshold` reproduces the older
  rule and is circular: it thresholds the same change signals whose detection
  threshold it feeds, biasing tau low. The stage says so when you select it.

Alignment is a stage rather than a separate tool. It runs once per site, before
the per-area stages, and does nothing when a config has no `align:` section.
Point it at an alignment YAML beside the site config, or write the same keys
inline:

```yaml
align: montecito_align.yaml
```

`pci_source_zones.align` also stands alone — `AlignConfig.load(path)` and
`run_alignment(cfg)` are importable from a notebook or another repo without a
site config.

## Configuration

One canonical schema for every site. Start from `config/templates/site.yaml`,
which documents every key at its default.

Unknown keys are a **hard error** — a mistyped threshold fails loudly instead of
being silently ignored.

```
config/
  templates/site.yaml          annotated template
  sites/<site>_<pre>_<post>.yaml   one per fire and epoch pair
  ml_*.yaml                    ML experiment configs
  legacy/montecito_physics.yaml legacy physics/PCI config for scripts/01-06
```

## Package layout

```
src/pci_source_zones/
  # foundation — imported by every track, so deliberately flat
  config.py        config loading and path resolution
  inputs.py        raster readers and writers for the physics track
  cli.py           the `pci-zones` command
  raster.py        raster I/O, aggregation, grid validation
  geometry.py      AOI rasterisation, ID discovery and parsing
  checks.py        pipeline invariant assertions
  site_config.py   canonical site schema
  align.py         alignment to a reference grid (a stage, and a portable utility)
  runner.py        stage execution, resumability, failure isolation, manifests

  # tracks
  physics/         terrain forcing, resistance model, PCI, source areas
  evidence/        detection threshold, evidence layers, p_obs, inversion
  ml/              models, splits, tuning, inference
```

Three tracks, one foundation. `config.py` and `inputs.py` stay flat because
physics, evidence and ML all import them — moving them under `physics/` would
have made the other two tracks depend on a package they have nothing to do with.

## Invariants

Every defect found in this pipeline has been *silent* — a plausible wrong number
rather than a crash. `checks.py` makes that class loud:

- nodata must propagate through comparisons (`NaN > x` is `False`, not `NaN`)
- a configured filter that removes **zero** cells fails, rather than appearing to work
- probabilities are asserted to lie in [0, 1]
- all-NaN slices must not collapse to `0.0` in a sum

Call them liberally. They are cheap and they target the failure mode that has
cost the most time.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Synthetic rasters only — no site data, no network. CI runs on every push.

## Data acquisition

Not in this repo. Fetching, reprojection and harmonisation live upstream in
`landslide-data-prep`; this repo consumes analysis-ready rasters. Acquisition
code previously here is staged in `../pci-source-zones_bin/` pending that
integration.
