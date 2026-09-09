# pci-source-zones

Postfire debris-flow source zones. Two tracks over one foundation:

- **physics** — mechanistic. G = a·S^α vs a modeled resistance C.
- **evidence** — observational. LiDAR change → p_obs → C_eff by Gamma inversion.
- **ml** — learns C_eff / p_obs from predictors.

## Run it

```bash
pip install -e ".[dev]"
pci-zones validate --config config/sites/<site>.yaml
pci-zones run      --config config/sites/<site>.yaml
pytest -q
```

Stages: `align → terrain → stable → lod → evidence → p_obs → ceff`.
`prep` is a separate command that runs before all of it, only when the delivered
rasters are not already one square grid with real nodata.
Real inputs are **pre-DEM, post-DEM, dDEM**; everything else is derived.

## Conventions that matter

**Input preparation is a command, not a script per fire.** `prep` exists so
nobody hand-writes another regrid script with a hardcoded extent. Two rules in
it are silent when broken: mask fill values *before* resampling, and never apply
an elevation-style `valid_min` to a difference raster — negatives there are the
erosion signal, and removing them yields an all-positive dDEM that looks fine.

**A new site is a config file, never a code change.** If you find yourself
editing a module to onboard a fire, the config schema is missing a key.

**Unknown config keys are a hard error.** A silently ignored typo in a threshold
is the failure class this repo has lost the most time to. Never add a lenient
fallback, and never keep a retired key working as an alias — that is exactly how
two flow-routing implementations drifted apart.

**Every defect found here has been silent** — a plausible wrong number, not a
crash. Prefer a loud failure to a sensible default. `checks.py` exists for this:

- `NaN > x` is `False`, not `NaN` — propagate nodata explicitly after comparisons
- `np.nansum` of an all-NaN slice is `0.0` — use `raster.nansum_masked`
- a filter that removes **zero** cells is a bug, not a no-op
- a glob that matches nothing silently deletes test coverage — assert non-empty

**One definition per quantity.** `driving_index()` is the only G. `nmad()` is the
only noise estimate, shared by C2 and C3. `evidence/core.py` holds the science;
`runner.py` only does I/O and orchestration. Keep pure functions pure — they are
what the tests can actually pin down.

**Foundation stays flat.** `config.py`, `inputs.py`, `raster.py`, `geometry.py`,
`checks.py`, `site_config.py`, `align.py`, `runner.py` are imported by every
track. Only track-specific modules belong in `physics/`, `evidence/`, `ml/`.

## Science you should not change casually

`α = 7/6` in `physics/terrain.py` and `m1 = 0.6` in `physics/c_model.py` are the
same Manning exponent. They derive together:

```
τ = ρ g h S,  h = (n q / √S)^0.6  ⇒  τ ∝ n^0.6 q^0.6 S^0.7
set τ = τ_c, raise to 1/0.6      ⇒  a · S^(7/6) = C
```

Change one and you must change the other. Any doc claiming α = 2 is wrong.

`evidence.c3.lod` defaults to `auto` on purpose: a hard-coded LoD is a
site-specific number wearing a default's clothing.

## Known open items

- `fit_gamma` fits shape `k` to the spread of `G_mean` *across* cells;
  `invert_threshold` then uses that `k` as the shape of the threshold
  distribution *within* a cell. The algebra is right, the identification is an
  assumption. Weakest link in the evidence track.
- WhiteboxTools writes its own nodata sentinel while declaring the source
  nodata, so DEMs with nodata holes produce contaminated G. Deferred by
  decision — do not run sites with nodata gaps until it is addressed.
- The new pipeline has not been diffed against the retired `compute_*.py`
  outputs on Montecito.
- `ml/` does not use `SiteConfig`; it still has its own config world.
