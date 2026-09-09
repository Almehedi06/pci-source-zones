"""Stage runner: executes pipeline stages for one site or many analysis areas.

Single-site is a batch of one. There is exactly one code path, so behaviour
cannot drift between the two modes — which it previously did, with the batch
scripts carrying resumability and error isolation that the single scripts lacked.

Production behaviours carried from the batch implementation:
    * skip work whose outputs already exist, unless --overwrite
    * isolate failures per area, so one bad ID does not kill a run
    * write a run summary (CSV + JSON) and a manifest with config and git state

Stages:
    align     put predictor rasters on the reference grid (site-level, optional)
    terrain   G and DeltaG from the pre/post DEM pair
    stable    stable-ground mask, the sample the LoD is estimated from
    lod       detection threshold from stable ground
    evidence  C2 / C3 binary layers
    p_obs     coarse observed erosion probability and terrain forcing
    ceff      Gamma fit and per-cell threshold inversion
"""
from __future__ import annotations

import csv
import json
import subprocess
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import checks, evidence
from .align import run_alignment
from .contract import require_contract
from .geometry import rasterize_aoi, sort_ids

VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json", ".gdb", ".fgb"}
from .raster import (
    grids_match,
    read_binary,
    read_float,
    require_same_grid,
    reproject_to_grid,
    raster_stats,
    transform_for_block,
    write_float,
)
from .site_config import SiteConfig

STAGES = ("align", "terrain", "stable", "lod", "evidence", "p_obs", "ceff")

# Alignment prepares the grid every analysis area shares, so it runs once per
# site rather than once per area. Keeping it in STAGES is what stops it being a
# separate task with its own command and its own config file.
SITE_STAGES = ("align",)


# ────────────────────────────────────────────────────────── results

@dataclass
class AreaResult:
    """Outcome of one stage for one analysis area."""
    area_id: str
    stage: str
    status: str                       # ok | skipped | failed
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # Detail dicts are stage-authored, and several carry their own "area_id".
    # Without this guard they overwrite the identity columns and the summary
    # loses track of which area a row belongs to.
    RESERVED = ("area_id", "stage", "status", "error")

    def row(self) -> dict[str, Any]:
        r = {"area_id": self.area_id, "stage": self.stage, "status": self.status}
        for k, v in self.detail.items():
            if k in self.RESERVED:
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                r[k] = v
        if self.error:
            r["error"] = self.error
        return r


# ────────────────────────────────────────────────────────── discovery

def discover_areas(cfg: SiteConfig, only: list[str] | None = None) -> list[str | None]:
    """Analysis-area IDs to process.

    Single mode returns [None] — one unnamed area. Batch mode discovers IDs from
    the first configured input and keeps only IDs whose other configured inputs
    also exist. That avoids launching areas that have a pre DEM but no post DEM
    or dDEM.
    """
    if not cfg.is_batch:
        return [None]

    key = next(iter(cfg.inputs.templates), None)
    if key is None:
        raise ValueError("batch mode requires inputs.templates")

    ids = _discover_ids_for_template(
        Path(cfg.inputs.roots[key]).expanduser(),
        cfg.inputs.templates[key],
        cfg.inputs.id_glob,
    )
    for other_key, template in cfg.inputs.templates.items():
        if other_key == key:
            continue
        root = Path(cfg.inputs.roots[other_key]).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"input root not found for {other_key}: {root}")
        ids = {i for i in ids if (root / template.format(id=i)).exists()}

    out = sort_ids(ids)
    if only:
        wanted = set(only)
        missing = wanted - set(out)
        if missing:
            print(f"  WARNING: requested IDs not found on disk: {sorted(missing)}")
        out = [i for i in out if i in wanted]
    return out


def _discover_ids_for_template(root: Path, template: str,
                               glob_pattern: str | None = None) -> set[str]:
    """IDs present under one batch root/template pair."""
    if not root.exists():
        raise FileNotFoundError(f"input root not found: {root}")

    glob_pattern = glob_pattern or template.replace("{id}", "*")
    prefix, _, suffix = template.partition("{id}")

    found: set[str] = set()
    for path in root.glob(glob_pattern):
        rel = str(path.relative_to(root))
        if rel.startswith(prefix) and rel.endswith(suffix):
            area = rel[len(prefix):len(rel) - len(suffix)] if suffix else rel[len(prefix):]
            if area:
                found.add(area)
    return found


# ────────────────────────────────────────────────────────── stages

def _paths(cfg: SiteConfig, area: str | None) -> dict[str, Path]:
    out = cfg.output_dir(area)
    return {
        "dir": out,
        "lod": out / "lod.json",
        "c2": out / "C2.tif",
        "c3": out / "C3.tif",
        "p_inter": out / "p_obs_intersection.tif",
        "p_union": out / "p_obs_union.tif",
        "p_c2": out / "p_c2.tif",
        "p_c3": out / "p_c3.tif",
        "g_mean": out / "G_mean.tif",
        "g_star": out / "G_star.tif",
        "gamma": out / "gamma_fit.json",
        # terrain / stable stage products, resolved ahead of configured inputs
        "g_pre": out / "G_pre.tif",
        "g_post": out / "G_post.tif",
        "s_pre": out / "S_pre.tif",
        "delta_g": out / "DeltaG.tif",
        "terrain_meta": out / "terrain.json",
        "stable_cells": out / "stable_cells.tif",
    }


def resolve_input(cfg: SiteConfig, key: str, area: str | None) -> Path | None:
    """Prefer a product this pipeline produced over an externally configured path.

    Lets `terrain` and `stable` feed `lod`/`evidence`/`p_obs` without the config
    naming their outputs twice, while a config that supplies G, DeltaG or a
    stable mask directly still wins when the stage has not been run.
    """
    produced = _paths(cfg, area).get(key)
    if produced is not None and produced.exists():
        return produced
    return cfg.input_path(key, area)


def stage_align(cfg: SiteConfig, area: str | None = None, *,
                overwrite: bool = False) -> dict[str, Any]:
    """Put predictor rasters on the reference grid. Site-level; no-op if unset."""
    acfg = cfg.align_config()
    if acfg is None:
        return {"configured": False}
    summary = run_alignment(acfg, overwrite=overwrite)
    return {"configured": True, **{k: v for k, v in summary.items()
                                   if not isinstance(v, list)}}


def stage_terrain(cfg: SiteConfig, area: str | None) -> dict[str, Any]:
    """G for both epochs and DeltaG, from the pre/post DEM pair."""
    from .physics.terrain import terrain_deltas

    p = _paths(cfg, area)
    dem_pre = cfg.input_path("dem_pre", area)
    dem_post = cfg.input_path("dem_post", area)
    if not dem_pre and not dem_post:
        if cfg.input_path("g_pre", area) and cfg.input_path("delta_g", area):
            return {"configured": False}      # G supplied directly; nothing to derive
        raise ValueError(
            "terrain stage needs inputs.dem_pre and inputs.dem_post, or "
            "precomputed inputs.g_pre and inputs.delta_g")
    if not dem_pre or not dem_post:
        raise ValueError(
            "terrain stage needs both inputs.dem_pre and inputs.dem_post; "
            f"got dem_pre={dem_pre}, dem_post={dem_post}")

    p["dir"].mkdir(parents=True, exist_ok=True)
    result = terrain_deltas(dem_pre, dem_post, p["dir"], cfg.terrain.params())
    result["area_id"] = area
    p["terrain_meta"].write_text(json.dumps(result, indent=2))
    return result


def stage_stable(cfg: SiteConfig, area: str | None) -> dict[str, Any]:
    """Stable-ground mask — the sample the detection threshold is estimated from."""
    spec = cfg.stable
    if spec is None:
        return {"configured": False}          # mask supplied directly via inputs

    p = _paths(cfg, area)
    masks: dict[str, np.ndarray] = {}
    ref_prof: dict[str, Any] | None = None
    skipped: list[str] = []

    # Rasters first, so a vector layer has a grid to be rasterised onto.
    ordered = sorted(spec.layers, key=lambda x: Path(x.path).suffix.lower() in VECTOR_SUFFIXES)

    for layer in ordered:
        path = Path(layer.path.format(id=area) if area else layer.path).expanduser()
        if not path.exists():
            if layer.required:
                raise FileNotFoundError(
                    f"required stable layer '{layer.name}' not found: {path}")
            skipped.append(layer.name)
            print(f"    SKIP optional {layer.name} (not found: {path})")
            continue

        if path.suffix.lower() in VECTOR_SUFFIXES:
            if ref_prof is None:
                raise ValueError(
                    f"stable layer '{layer.name}' is a vector but no raster layer "
                    "has established the grid — add at least one raster layer")
            inside = rasterize_aoi(path, (ref_prof["height"], ref_prof["width"]),
                                   ref_prof["transform"], ref_prof.get("crs"))
            arr, prof = inside.astype("float64"), ref_prof
        else:
            arr, prof = read_float(path)
        if ref_prof is None:
            ref_prof = prof
        else:
            require_same_grid(ref_prof, {layer.name: prof})
        if layer.threshold is not None:
            m = evidence.threshold_mask(arr, layer.threshold)
        elif layer.min is not None or layer.max is not None:
            m = evidence.range_mask(arr, layer.min, layer.max)
        else:
            m = arr.astype("float64")      # vector: already 1 inside, 0 outside
        if layer.invert:
            m = np.where(np.isfinite(m), 1.0 - m, np.nan)
        masks[layer.name] = m
        write_float(p["dir"] / f"stable_{layer.name}.tif", m, prof)

    if not masks:
        raise FileNotFoundError(
            f"no stable-mask layers found on disk (tried {[x.name for x in spec.layers]})")

    combined = evidence.combine_masks(np.stack(list(masks.values())), spec.combine)
    write_float(p["stable_cells"], combined, ref_prof)

    stats = evidence.stable_stats(combined)
    checks.check_range(combined, 0.0, 1.0, "stable mask")
    if stats["n_stable"] == 0:
        raise checks.InvariantError(
            "stable mask selected 0 cells — every LoD downstream would be undefined")
    if stats["frac_stable"] < spec.min_frac:
        print(f"    WARNING: stable sample is {100 * stats['frac_stable']:.2f}% of "
              f"valid cells (< {100 * spec.min_frac:.1f}%); NMAD may be unrepresentative")

    out = {"method": spec.method, "combine": spec.combine,
           "layers_used": list(masks), "layers_skipped": skipped, **stats}
    if spec.method == "threshold":
        print("    NOTE: method 'threshold' derives stable ground from the same "
              "change signals whose detection threshold it feeds — tau is biased low")
        out["circular"] = True

    # is the sample representative of the terrain erosion actually happens on?
    g_path = resolve_input(cfg, "g_pre", area)
    if g_path is not None and Path(g_path).exists():
        g, _ = read_float(g_path)
        if g.shape == combined.shape:
            rep = evidence.check_representative(combined, g, "G", min_frac=spec.min_frac)
            out["representativeness"] = rep
            if "warning" in rep:
                print(f"    WARNING: {rep['warning']}")
    return out


def stage_lod(cfg: SiteConfig, area: str | None) -> dict[str, Any]:
    """Detection threshold for the change signal, from stable ground."""
    p = _paths(cfg, area)
    dg_path = resolve_input(cfg, "delta_g", area)
    stable_path = resolve_input(cfg, "stable_cells", area)
    if dg_path is None:
        raise ValueError("lod stage needs inputs.delta_g or a terrain stage DeltaG.tif")
    if stable_path is None:
        raise ValueError("lod stage needs inputs.stable_cells")

    delta_g, prof = read_float(dg_path)
    stable_arr, stable_prof = read_binary(stable_path)
    require_same_grid(prof, {"stable_cells": stable_prof})
    stable = stable_arr == 1.0

    if cfg.lod.exclude_channels:
        g_path = resolve_input(cfg, "g_pre", area)
        g, g_prof = read_float(g_path)
        require_same_grid(prof, {"g_pre": g_prof})
        cutoff = cfg.lod.g_channel or cfg.evidence.domain.g_channel
        before = np.where(stable, 1.0, np.nan)
        stable &= g < cutoff
        checks.check_mask_effect(before, np.where(stable, 1.0, np.nan),
                                 "channel exclusion from stable set", warn=True)

    dg = evidence.detection_threshold(delta_g, stable, cfg.lod.scale, name="DeltaG")
    result: dict[str, Any] = {
        "method": cfg.lod.method, "area_id": area, "n_stable": dg["n_stable"],
        "nmad_delta_g": dg["nmad"], "delta_g_lod": dg["lod"], "scale": dg["scale"],
    }

    # Same rule, same stable sample, for the dz signal — only needed when C3 is
    # set to derive its own threshold.
    if cfg.evidence.c3.is_auto:
        dz_path = resolve_input(cfg, "dz", area)
        if dz_path is None:
            raise ValueError("evidence.c3.lod is 'auto' but inputs.dz is not set")
        dz, dz_prof = read_float(dz_path, nodata=cfg.inputs.dz_nodata)
        require_same_grid(prof, {"dz": dz_prof})
        dzr = evidence.detection_threshold(dz, stable, cfg.evidence.c3.scale, name="dz")
        result.update(nmad_dz=dzr["nmad"], dz_lod=dzr["lod"],
                      dz_median_stable=dzr["median"], dz_scale=dzr["scale"])

    p["dir"].mkdir(parents=True, exist_ok=True)
    p["lod"].write_text(json.dumps(result, indent=2))
    return result


def resolve_c3_lod(cfg: SiteConfig, area: str | None, lod_json: dict[str, Any],
                   reference: dict[str, Any]) -> tuple[float | np.ndarray, str]:
    """Turn `evidence.c3.lod` into metres: a number, a raster, or the measured value."""
    spec = cfg.evidence.c3
    if spec.is_auto:
        if "dz_lod" not in lod_json:
            raise ValueError(
                "evidence.c3.lod is 'auto' but the lod stage recorded no dz_lod — "
                "rerun the lod stage")
        return float(lod_json["dz_lod"]), "auto"
    if spec.raster_path is not None:
        path = Path(spec.raster_path.format(id=area) if area else spec.raster_path)
        sigma, sigma_prof = read_float(path.expanduser())
        require_same_grid(reference, {"c3 lod raster": sigma_prof})
        return sigma * spec.scale, f"raster {path.name} x {spec.scale}"
    return float(spec.lod), "constant"


def stage_evidence(cfg: SiteConfig, area: str | None) -> dict[str, Any]:
    """C2 and C3 binary evidence layers at the fine resolution."""
    p = _paths(cfg, area)
    if not p["lod"].exists():
        raise FileNotFoundError(f"{p['lod']} not found — run the lod stage first")
    tau = float(json.loads(p["lod"].read_text())["delta_g_lod"])

    dg_path = resolve_input(cfg, "delta_g", area)
    dz_path = resolve_input(cfg, "dz", area)
    if dg_path is None or dz_path is None:
        raise ValueError("evidence stage needs inputs.delta_g and inputs.dz")
    delta_g, prof = read_float(dg_path)
    dz, dz_prof = read_float(dz_path, nodata=cfg.inputs.dz_nodata)
    require_same_grid(prof, {"dz": dz_prof})

    lod_json = json.loads(p["lod"].read_text())
    lod, how = resolve_c3_lod(cfg, area, lod_json, prof)
    c2 = evidence.c2_layer(delta_g, tau)
    c3 = evidence.c3_layer(dz, lod)
    detail: dict[str, Any] = {
        "tau_dg": tau, "c3_lod_source": how,
        "c3_lod": float(np.nanmean(lod)) if np.ndim(lod) else float(lod),
    }

    write_float(p["c2"], c2, prof)
    write_float(p["c3"], c3, prof)
    return {**detail, "n_c2_true": int(np.nansum(c2)),
            "n_c3_true": int(np.nansum(c3)),
            "n_valid_fine": int((np.isfinite(c2) & np.isfinite(c3)).sum())}


def stage_p_obs(cfg: SiteConfig, area: str | None) -> dict[str, Any]:
    """Aggregate evidence to coarse observed erosion probability."""
    p = _paths(cfg, area)
    for key in ("c2", "c3"):
        if not p[key].exists():
            raise FileNotFoundError(f"{p[key]} not found — run the evidence stage first")

    c2, prof = read_float(p["c2"])
    c3, c3_prof = read_float(p["c3"])
    g, g_prof = read_float(resolve_input(cfg, "g_pre", area))
    require_same_grid(prof, {"C3": c3_prof, "G_pre": g_prof})

    coverage = {"raw": c2.copy()}
    mask_stats: dict[str, Any] = {}
    if cfg.evidence.apply_domain:
        dom = cfg.evidence.domain
        elev = None
        if dom.elev_min is not None:
            dem_path = cfg.input_path("dem", area) or cfg.input_path("dem_pre", area)
            if dem_path is None:
                raise ValueError(
                    "evidence.domain.elev_min needs inputs.dem or inputs.dem_pre")
            elev = _read_or_reproject(dem_path, prof)

        slope = None
        if dom.slope_max_deg is not None:
            slope_path = resolve_input(cfg, "s_pre", area)
            if slope_path is None:
                raise ValueError(
                    "evidence.domain.slope_max_deg needs inputs.s_pre or the terrain stage output")
            slope = _read_or_reproject(slope_path, prof)

        masked, mask_stats = evidence.apply_domain_mask(
            {"C2": c2, "C3": c3, "G": g}, g, dom.g_min, dom.g_channel,
            elev=elev, elev_min=dom.elev_min,
            slope=slope, slope_max_deg=dom.slope_max_deg)
        c2, c3, g = masked["C2"], masked["C3"], masked["G"]
        coverage["after domain mask"] = c2

    result = evidence.aggregate_p_obs(c2, c3, g, cfg.block_size)

    # AOI clip, if configured
    aoi_path = cfg.input_path("aoi", area)
    t_coarse = transform_for_block(prof["transform"], cfg.block_size)
    if aoi_path is not None:
        aoi = rasterize_aoi(aoi_path, result.p_union.shape, t_coarse, prof.get("crs"))
        for arr in (result.p_c2, result.p_c3, result.p_intersection,
                    result.p_union, result.g_mean, result.g_star):
            arr[~aoi] = np.nan
        mask_stats["n_inside_aoi"] = int(aoi.sum())

    print(checks.summarize_coverage(coverage))
    write_float(p["p_c2"], result.p_c2, prof, t_coarse)
    write_float(p["p_c3"], result.p_c3, prof, t_coarse)
    write_float(p["p_inter"], result.p_intersection, prof, t_coarse)
    write_float(p["p_union"], result.p_union, prof, t_coarse)
    write_float(p["g_mean"], result.g_mean, prof, t_coarse)
    write_float(p["g_star"], result.g_star, prof, t_coarse)
    return {**result.stats(), **mask_stats}


def stage_ceff(cfg: SiteConfig, area: str | None) -> dict[str, Any]:
    """Fit the Gamma and invert observed probability to a per-cell threshold."""
    p = _paths(cfg, area)
    if not p["g_mean"].exists():
        raise FileNotFoundError(f"{p['g_mean']} not found — run the p_obs stage first")

    g_mean, prof = read_float(p["g_mean"])
    domain = np.isfinite(g_mean)

    if cfg.ceff.elev_min is not None:
        dem_path = cfg.input_path("dem", area) or cfg.input_path("dem_pre", area)
        if dem_path is None:
            raise ValueError("ceff.elev_min needs inputs.dem or inputs.dem_pre")
        elev = reproject_to_grid(dem_path, prof["transform"], prof.get("crs"),
                                 prof["height"], prof["width"])
        domain &= np.isfinite(elev) & (elev > cfg.ceff.elev_min)
    domain &= np.isfinite(g_mean) & (g_mean > 0) & (g_mean < cfg.ceff.g_hillslope_max)
    checks.check_min_valid(np.where(domain, g_mean, np.nan),
                       cfg.ceff.min_domain_cells, "ceff domain")

    fit = evidence.fit_gamma(g_mean, domain)
    p["gamma"].write_text(json.dumps({**fit.as_dict(), "area_id": area}, indent=2))

    out: dict[str, Any] = {**fit.as_dict()}
    invert_domain = domain if cfg.ceff.elev_filter_outputs else np.isfinite(g_mean)

    for method in cfg.ceff.p_obs_methods:
        src = p["p_inter"] if method == "intersection" else p["p_union"]
        if not src.exists():
            raise FileNotFoundError(f"{src} not found — run the p_obs stage first")
        p_obs, _ = read_float(src)
        c = evidence.invert_threshold(p_obs, g_mean, fit.k, invert_domain)
        tag = method[:5]
        write_float(p["dir"] / f"C_p_obs_{tag}.tif", c, prof)
        write_float(p["dir"] / f"C_p_obs_{tag}_log.tif",
                    np.log(np.clip(c, 1e-6, None)), prof)
        out[f"n_C_{tag}"] = int(np.isfinite(c).sum())
        out[f"C_{tag}_median"] = raster_stats(c)["median"]

    bound = evidence.threshold_at_probability(g_mean, fit.k, cfg.ceff.p_ceiling, invert_domain)
    tag = f"p{int(cfg.ceff.p_ceiling * 1000):03d}"
    write_float(p["dir"] / f"C_{tag}.tif", bound, prof)
    write_float(p["dir"] / f"C_{tag}_log.tif",
                np.log(np.clip(bound, 1e-6, None)), prof)
    out[f"n_C_{tag}"] = int(np.isfinite(bound).sum())
    return out


STAGE_FUNCS: dict[str, Callable[[SiteConfig, str | None], dict[str, Any]]] = {
    "align": stage_align,
    "terrain": stage_terrain,
    "stable": stage_stable,
    "lod": stage_lod,
    "evidence": stage_evidence,
    "p_obs": stage_p_obs,
    "ceff": stage_ceff,
}

STAGE_OUTPUT_KEY = {"terrain": "delta_g", "stable": "stable_cells", "lod": "lod", "evidence": "c2", "p_obs": "p_union"}
# "align" is absent deliberately: run_alignment does its own per-layer
# skip-if-exists, so there is no single marker file to test.


def _read_or_reproject(path: str | Path, reference: dict[str, Any]) -> np.ndarray:
    """Read a raster on `reference` grid, reprojecting only when necessary."""
    arr, prof = read_float(path)
    if grids_match(reference, prof):
        return arr
    return reproject_to_grid(path, reference["transform"], reference.get("crs"),
                             reference["height"], reference["width"])


def _ceff_outputs(cfg: SiteConfig, paths: dict[str, Path]) -> list[Path]:
    """All files that mean the ceff stage really completed."""
    out = [paths["gamma"]]
    for method in cfg.ceff.p_obs_methods:
        tag = method[:5]
        out.extend([
            paths["dir"] / f"C_p_obs_{tag}.tif",
            paths["dir"] / f"C_p_obs_{tag}_log.tif",
        ])
    tag = f"p{int(cfg.ceff.p_ceiling * 1000):03d}"
    out.extend([paths["dir"] / f"C_{tag}.tif", paths["dir"] / f"C_{tag}_log.tif"])
    return out


def _stage_complete(cfg: SiteConfig, stage: str, paths: dict[str, Path]) -> bool:
    if stage == "ceff":
        return all(path.exists() for path in _ceff_outputs(cfg, paths))
    return paths[STAGE_OUTPUT_KEY[stage]].exists()


# ────────────────────────────────────────────────────────── driver

def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def run(cfg: SiteConfig, stages: Iterable[str], areas: list[str] | None = None,
        *, overwrite: bool = False, fail_fast: bool = False) -> list[AreaResult]:
    """Run stages over one site or many analysis areas."""
    stages = list(stages)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; valid: {list(STAGES)}")

    ids = discover_areas(cfg, areas)
    label = "area" if cfg.is_batch else "site"
    print(f"{cfg.site}: {len(ids)} {label}(s), stages {stages}")

    # Refuse at the door rather than mid-run: a grid mismatch used to surface
    # inside `lod`, after the terrain stage had already routed both DEMs.
    require_contract(cfg, ids[0] if ids else None, stages)

    results: list[AreaResult] = []

    for stage in [s for s in stages if s in SITE_STAGES]:
        try:
            detail = STAGE_FUNCS[stage](cfg, None, overwrite=overwrite)
            status = "ok" if detail.get("configured", True) else "skipped"
            results.append(AreaResult(cfg.site, stage, status, detail))
            note = "" if status == "ok" else " (not configured)"
            print(f"  [{cfg.site}] {stage}: {status}{note}")
        except Exception as exc:
            results.append(AreaResult(cfg.site, stage, "failed",
                                      error=f"{type(exc).__name__}: {exc}"))
            print(f"  [{cfg.site}] {stage}: FAILED  {type(exc).__name__}: {exc}")
            if fail_fast:
                raise
            traceback.print_exc(limit=1)

    stages = [s for s in stages if s not in SITE_STAGES]
    for area in ids:
        name = area if area is not None else cfg.site
        for stage in stages:
            paths = _paths(cfg, area)
            if _stage_complete(cfg, stage, paths) and not overwrite:
                results.append(AreaResult(name, stage, "skipped"))
                print(f"  [{name}] {stage}: skipped (output exists)")
                continue
            try:
                detail = STAGE_FUNCS[stage](cfg, area)
                status = "ok" if detail.get("configured", True) else "skipped"
                results.append(AreaResult(name, stage, status, detail))
                note = "" if status == "ok" else " (not configured)"
                print(f"  [{name}] {stage}: {status}{note}")
            except Exception as exc:                       # isolate per area
                results.append(AreaResult(name, stage, "failed",
                                          error=f"{type(exc).__name__}: {exc}"))
                print(f"  [{name}] {stage}: FAILED  {type(exc).__name__}: {exc}")
                if fail_fast:
                    raise
                traceback.print_exc(limit=1)
                break                                      # skip dependent stages
    return results


def write_summary(cfg: SiteConfig, results: list[AreaResult], stages: list[str],
                  out_dir: Path | None = None) -> dict[str, Path]:
    """Write run summary (CSV) and manifest (JSON)."""
    out_dir = out_dir or cfg.output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows = [r.row() for r in results]
    csv_path = out_dir / f"run_summary_{stamp}.csv"
    if rows:
        cols: list[str] = []
        for r in rows:
            cols.extend(k for k in r if k not in cols)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    manifest = {
        "site": cfg.site, "epochs": cfg.epochs, "stages": stages,
        "run_utc": stamp, "git_sha": _git_sha(),
        "config_path": str(cfg.source_path) if cfg.source_path else None,
        "config": cfg.as_dict(),
        "counts": counts,
        "failures": [{"area": r.area_id, "stage": r.stage, "error": r.error}
                     for r in results if r.status == "failed"],
    }
    json_path = out_dir / f"run_manifest_{stamp}.json"
    json_path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n{counts}")
    print(f"summary  : {csv_path}")
    print(f"manifest : {json_path}")
    return {"summary": csv_path, "manifest": json_path}
