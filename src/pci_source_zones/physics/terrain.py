"""Terrain forcing G = a * S^alpha, and its change between two epochs.

    S      slope, m/m
    a      specific catchment area, m
    alpha  7/6, from the shear-stress derivation (see below)
    G      a * S^alpha

Flow routing is WhiteboxTools D-infinity throughout: flow is split between
downslope neighbours rather than forced into one of eight directions, which
matters on hillslopes, where this model is applied. There is deliberately only
one router — two implementations of G is how the pre-refactor repo drifted.

Why alpha = 7/6:

    tau  = rho g h S                          shear stress
    h    = (n q / sqrt(S))^0.6                Manning, solved for depth
    tau  = rho g n^0.6 q^0.6 S^0.7            substituting, with q = r * a
    set tau = tau_c and raise to 1/0.6:
    a * S^(7/6) = C

The 0.6 is the Manning exponent `m1` in `c_model.py`. alpha and m1 are the same
number wearing two hats and must be changed together.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import output_path, resolve_path
from ..raster import read_float, write_float


@dataclass
class TerrainParams:
    """Everything that changes the numbers G comes out with."""
    alpha: float = 1.167          # 7/6
    fill_sinks: bool = True
    curvature: bool = False       # also write plan curvature
    work_dir: str | None = None   # router scratch; defaults beside the products.
                                  # Point at fast local disk for large grids —
                                  # the router writes several DEM-sized files.

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")


def _whitebox():
    try:
        import whitebox
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "WhiteboxTools is required for terrain routing: pip install whitebox"
        ) from exc
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    return wbt


def route(dem_path: str | Path, work_dir: str | Path, tag: str,
          fill_sinks: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill, then derive slope (m/m) and specific catchment area (m) by D-infinity."""
    dem_path, work_dir = Path(dem_path), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wbt = _whitebox()

    routed = work_dir / f"{tag}_filled.tif"
    if fill_sinks:
        wbt.fill_depressions_wang_and_liu(str(dem_path), str(routed), fix_flats=True)
    else:
        routed = dem_path

    slope_pct = work_dir / f"S_{tag}_percent.tif"
    sca_path = work_dir / f"a_{tag}_sca.tif"
    wbt.slope(str(routed), str(slope_pct), units="percent")
    wbt.d_inf_flow_accumulation(str(routed), str(sca_path), out_type="sca")

    s_pct, profile = read_float(slope_pct)
    sca, _ = read_float(sca_path)
    return s_pct / 100.0, sca, profile


def driving_index(sca: np.ndarray, slope: np.ndarray, alpha: float) -> np.ndarray:
    """G = a * S^alpha. The single definition, shared by every caller."""
    with np.errstate(invalid="ignore"):
        g = sca * np.power(slope, alpha)
    g = np.where(np.isfinite(g), g, np.nan)
    return np.where(g < 0, np.nan, g)


def compute_g(dem_path: str | Path, out_dir: str | Path, tag: str,
              params: TerrainParams) -> tuple[np.ndarray, dict[str, Any]]:
    """Route one DEM and write its slope, specific catchment area and G."""
    dem_path, out_dir = Path(dem_path), Path(out_dir)
    # router scratch is kept out of the product directory so the outputs a user
    # sees are only the real products
    work = Path(params.work_dir) if params.work_dir else out_dir / "_terrain_work"
    slope, sca, profile = route(dem_path, work, tag, params.fill_sinks)
    g = driving_index(sca, slope, params.alpha)

    write_float(out_dir / f"S_{tag}.tif", slope, profile)
    write_float(out_dir / f"a_{tag}.tif", sca, profile)
    write_float(out_dir / f"G_{tag}.tif", g, profile)
    print(f"  G_{tag}: mean={np.nanmean(g):.4f}  max={np.nanmax(g):.4f}  "
          f"(alpha={params.alpha})")
    return g, profile


def plan_curvature(dem_path: str | Path, out_dir: str | Path,
                   tag: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Plan curvature on the raw DEM — a surface derivative, so no fill needed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"curvature_{tag}.tif"
    _whitebox().plan_curvature(str(dem_path), str(path))
    return read_float(path)


def terrain_deltas(dem_pre: str | Path, dem_post: str | Path,
                   out_dir: str | Path,
                   params: TerrainParams | None = None) -> dict[str, Any]:
    """G for both epochs and their difference. The evidence pipeline's entry point.

    Writes G_pre.tif, G_post.tif, DeltaG.tif (and curvature products when asked)
    into `out_dir`, and returns paths plus summary statistics.
    """
    params = params or TerrainParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g_pre, profile = compute_g(dem_pre, out_dir, "pre", params)
    g_post, _ = compute_g(dem_post, out_dir, "post", params)
    if g_post.shape != g_pre.shape:
        raise ValueError(
            f"pre/post DEMs give different grids: {g_pre.shape} vs {g_post.shape}. "
            "Put both epochs on one grid before computing DeltaG.")

    delta_g = g_post - g_pre
    write_float(out_dir / "DeltaG.tif", delta_g, profile)
    out: dict[str, Any] = {
        "alpha": params.alpha,
        "g_pre": str(out_dir / "G_pre.tif"),
        "g_post": str(out_dir / "G_post.tif"),
        "delta_g": str(out_dir / "DeltaG.tif"),
        "mean_g_pre": float(np.nanmean(g_pre)),
        "mean_delta_g": float(np.nanmean(delta_g)),
        "n_valid": int(np.isfinite(delta_g).sum()),
    }

    if params.curvature:
        c_pre, c_profile = plan_curvature(dem_pre, out_dir, "pre")
        c_post, _ = plan_curvature(dem_post, out_dir, "post")
        write_float(out_dir / "DeltaC.tif", c_post - c_pre, c_profile)
        out["delta_c"] = str(out_dir / "DeltaC.tif")
    return out


# ─────────────────────────────────────────────────── legacy physics-track entry

def prepare_terrain(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute S, a and G from a single DEM, for the old physics-track config.

    Kept for `scripts/01_prepare_terrain.py` and the source-area workflow. Now
    routes with D-infinity like everything else, so the physics and evidence
    tracks finally share one definition of G.
    """
    terrain_cfg = cfg.get("terrain", {})
    params = TerrainParams(
        alpha=float(cfg.get("parameters", {}).get("alpha", 1.167)),
        fill_sinks=bool(terrain_cfg.get("fill_sinks", True)),
    )

    dem_path = resolve_path(cfg, cfg["paths"]["dem"])
    slope_path = output_path(cfg, "slope", "slope.tif")
    area_path = output_path(cfg, "specific_catchment_area", "specific_catchment_area.tif")
    g_path = output_path(cfg, "topographic_driving_index", "topographic_driving_index.tif")

    slope, sca, profile = route(dem_path, Path(g_path).parent / "_terrain_work",
                                "prepare", params.fill_sinks)
    g = driving_index(sca, slope, params.alpha)

    write_float(slope_path, slope, profile)
    write_float(area_path, sca, profile)
    write_float(g_path, g, profile)
    return {
        "slope": slope_path,
        "specific_catchment_area": area_path,
        "topographic_driving_index": g_path,
    }
