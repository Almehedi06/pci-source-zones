"""The input contract: what the pipeline requires of its inputs, checked up front.

Four requirements, and until this module existed none of them were verified
before compute started:

1. every input on ONE grid — same CRS, transform, shape
2. square pixels — flow routing and slope assume it
3. declared nodata — no undeclared fill masquerading as data
4. the files actually exist and carry a CRS

`validate` used to read only the YAML, so a config pointing at three mismatched
rasters reported "config OK". The mismatch then surfaced inside `lod`, after the
terrain stage had already routed both DEMs — half an hour to learn something
readable from a file header. Non-square pixels were checked nowhere at all: they
never failed, they just made every slope wrong.

This module only reads. It opens headers, samples a few blocks and returns a
verdict; it never modifies a file. That is deliberate — it keeps the check from
growing into a data-preparation tool. Fixing belongs in `prep` or upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .raster import transforms_match
from .site_config import SiteConfig

# Inputs that must share the analysis grid. `dem` is excluded because it is
# reprojected onto the coarse grid when the elevation floor is applied, and
# `aoi` because it is a vector.
GRID_KEYS = ("dem_pre", "dem_post", "dz", "g_pre", "delta_g", "stable_cells")
EXISTENCE_ONLY = ("dem", "aoi")

# Which stages actually read each input. A partial run must not be blocked by
# a file it never opens.
KEY_STAGES = {
    "dem_pre": {"terrain"}, "dem_post": {"terrain"},
    "dz": {"lod", "evidence"},
    "g_pre": {"stable", "lod", "p_obs"},
    "delta_g": {"lod", "evidence"},
    "stable_cells": {"lod"},
    "dem": {"ceff"}, "aoi": {"p_obs"},
}

SQUARE_TOL = 1e-6          # relative difference between x and y pixel size
FILL_SUSPECT = 0.02        # a single value on this share of a surface is a fill
CATEGORICAL_MAX_VALUES = 4 # at or below this, the raster is a mask, not a surface


@dataclass
class Issue:
    """One contract violation."""
    name: str
    level: str              # "error" | "warning"
    problem: str
    hint: str = ""

    def line(self) -> str:
        tail = f"\n             -> {self.hint}" if self.hint else ""
        return f"  {self.name:<12} {self.problem}{tail}"


def _suspect_fill(src, sample_blocks: int = 40) -> tuple[float, float] | None:
    """The most repeated value and its share, when that share looks like fill."""
    windows = [w for _, w in src.block_windows(1)]
    if not windows:
        return None
    rng = np.random.default_rng(0)
    n = min(sample_blocks, len(windows))
    picks = [windows[i] for i in rng.choice(len(windows), n, replace=False)]
    arr = np.concatenate([src.read(1, window=w).astype("float64").ravel()
                          for w in picks])
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    values, counts = np.unique(finite, return_counts=True)
    # A binary or categorical raster — a stable-ground mask, a burn severity
    # class — repeats its values by design. Only a continuous surface makes a
    # repeated value suspicious.
    if values.size <= CATEGORICAL_MAX_VALUES:
        return None
    i = int(np.argmax(counts))
    share = counts[i] / arr.size
    return (float(values[i]), float(share)) if share >= FILL_SUSPECT else None


# What each stage needs a source for. The schema stays permissive so a partial
# run (`--stages ceff` over existing products) is possible; the requirement is
# enforced here, against the stages actually requested.
STAGE_NEEDS = {
    "terrain": ("g_source",),
    "lod": ("g_source", "stable_source"),
    "evidence": ("g_source", "dz"),
    "p_obs": ("g_source",),
}


def _missing_sources(cfg: SiteConfig, stages: list[str] | None) -> list[Issue]:
    """Inputs a requested stage has no source for at all."""
    if stages is None:
        return []
    needed = {n for s in stages for n in STAGE_NEEDS.get(s, ())}
    out: list[Issue] = []
    if "g_source" in needed and not cfg.is_batch:
        has_dems = bool(cfg.inputs.dem_pre and cfg.inputs.dem_post)
        has_g = bool(cfg.inputs.g_pre and cfg.inputs.delta_g)
        if not (has_dems or has_g):
            out.append(Issue("inputs", "error", "no source for G",
                             "supply dem_pre + dem_post, or g_pre + delta_g"))
    if "dz" in needed and not cfg.is_batch and not cfg.inputs.dz:
        out.append(Issue("dz", "error", "not configured",
                         "the evidence stage cannot build C3 without it"))
    if "stable_source" in needed and not cfg.is_batch:
        if not cfg.inputs.stable_cells and cfg.stable is None:
            out.append(Issue("stable_cells", "error", "no source",
                             "supply inputs.stable_cells or a `stable:` section"))
    return out


def check_inputs(cfg: SiteConfig, area: str | None = None, *,
                 sample_fills: bool = True,
                 missing_is_error: bool = True,
                 stages: list[str] | None = None) -> list[Issue]:
    """Verify the input contract. Returns every problem found, not just the first.

    `missing_is_error` is False for `validate`, which is a config check that
    should work on a machine where the data is not mounted, and True for `run`,
    which cannot proceed without it.
    """
    absent = "error" if missing_is_error else "warning"
    issues: list[Issue] = _missing_sources(cfg, stages)
    reference: tuple[str, dict[str, Any]] | None = None

    def wanted(key: str) -> bool:
        return stages is None or bool(KEY_STAGES.get(key, set()) & set(stages))

    for key in EXISTENCE_ONLY:
        if not wanted(key):
            continue
        path = cfg.input_path(key, area)
        if path is not None and not Path(path).exists():
            issues.append(Issue(key, absent, f"file not found: {path}"))

    for key in GRID_KEYS:
        if not wanted(key):
            continue
        path = cfg.input_path(key, area)
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            # A stage may be configured to produce it; absence is only an error
            # when nothing upstream will.
            if key in ("g_pre", "delta_g") and cfg.input_path("dem_pre", area):
                continue
            if key == "stable_cells" and cfg.stable is not None:
                continue
            issues.append(Issue(key, absent, f"file not found: {path}"))
            continue

        try:
            src = rasterio.open(path)
        except Exception as exc:                       # unreadable / not a raster
            issues.append(Issue(key, "error", f"cannot open: {type(exc).__name__}"))
            continue

        with src:
            if src.crs is None:
                issues.append(Issue(key, "error", "no CRS",
                                    "assign one before running"))

            rx, ry = src.res
            if abs(rx - ry) > SQUARE_TOL * max(rx, ry):
                issues.append(Issue(
                    key, "error", f"pixels are not square ({rx:.6g} x {ry:.6g})",
                    "flow routing and slope assume square cells — run `pci-zones prep`"))

            profile = {"height": src.height, "width": src.width,
                       "transform": src.transform, "crs": src.crs}
            if reference is None:
                reference = (key, profile)
            elif not (profile["height"] == reference[1]["height"]
                      and profile["width"] == reference[1]["width"]
                      and transforms_match(profile["transform"],
                                           reference[1]["transform"])
                      and profile["crs"] == reference[1]["crs"]):
                ref_name, ref = reference
                issues.append(Issue(
                    key, "error",
                    f"grid differs from {ref_name} "
                    f"({src.width}x{src.height} vs {ref['width']}x{ref['height']})",
                    "every input must share one grid — run `pci-zones prep`"))

            if src.nodata is None:
                suspect = _suspect_fill(src) if sample_fills else None
                if suspect is not None:
                    value, share = suspect
                    issues.append(Issue(
                        key, "error",
                        f"no nodata declared; {100 * share:.1f}% of sampled cells "
                        f"are exactly {value:g}",
                        "that value will be read as data — run `pci-zones prep`"))
                else:
                    issues.append(Issue(key, "warning", "no nodata declared"))

    return issues


def format_issues(issues: list[Issue]) -> str:
    """Render a report. Empty string when the contract holds."""
    if not issues:
        return "input contract: OK"
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    head = "input contract: "
    parts = []
    if errors:
        parts.append(f"{len(errors)} FAILURE{'S' if len(errors) != 1 else ''}")
    if warnings:
        parts.append(f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''}")
    lines = [head + ", ".join(parts)]
    lines += [i.line() for i in errors + warnings]
    return "\n".join(lines)


class ContractError(ValueError):
    """The inputs do not satisfy the pipeline's requirements."""


def require_contract(cfg: SiteConfig, area: str | None = None,
                     stages: list[str] | None = None) -> list[Issue]:
    """Raise unless the inputs satisfy the contract. Warnings are printed, not raised."""
    issues = check_inputs(cfg, area, stages=stages)
    report = format_issues(issues)
    if any(i.level == "error" for i in issues):
        raise ContractError(report)
    if issues:
        print(report)
    return issues
