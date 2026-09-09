"""Core evidence computations: detection threshold, evidence layers, p_obs, inversion.

Pure functions over arrays. No I/O, no config dicts, no site knowledge — so they
can be tested with synthetic rasters and reused by both the single-site and batch
entry points.

Pipeline order:

    delta_g_lod       ->  tau_dG from stable ground
    evidence_layers   ->  C2, C3 binary layers at fine resolution
    aggregate_p_obs   ->  p_obs (intersection and union) at coarse resolution
    fit_gamma         ->  k, theta on the hillslope domain
    invert_threshold  ->  per-cell initiation threshold C
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.stats import gamma as gamma_dist

from .. import checks
from ..raster import block_aggregate


# ──────────────────────────────────────────────── detection threshold

def nmad(values: np.ndarray) -> float:
    """Normalized median absolute deviation: a robust estimate of noise sigma.

        NMAD = 1.4826 * median(|x - median(x)|)

    Median-centred, so a residual vertical bias in a dDEM does not inflate it.
    The 1.4826 makes it match the standard deviation for Gaussian noise.
    """
    values = np.asarray(values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values to estimate NMAD from")
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def detection_threshold(values: np.ndarray, stable: np.ndarray, scale: float = 2.0,
                        name: str = "signal") -> dict[str, float]:
    """Detection threshold for one change signal, measured on stable ground.

        LoD = scale * NMAD(values on stable ground)

    The same rule is used for DeltaG and for dz, so C2 and C3 share one
    definition of "bigger than the noise". scale=2.0 is roughly 95%.
    """
    sample = values[stable & np.isfinite(values)]
    if sample.size == 0:
        raise ValueError(f"no valid {name} values in the stable mask")
    sigma = nmad(sample)
    return {"nmad": sigma, "lod": float(scale * sigma),
            "median": float(np.median(sample)),
            "n_stable": int(sample.size), "scale": float(scale)}


def delta_g_lod(delta_g: np.ndarray, stable: np.ndarray, scale: float = 2.0) -> dict[str, float]:
    """Detection threshold for DeltaG. Thin wrapper over `detection_threshold`."""
    out = detection_threshold(delta_g, stable, scale, name="DeltaG")
    out["delta_g_lod"] = out["lod"]
    return out


# ──────────────────────────────────────────────── evidence layers

def c2_layer(delta_g: np.ndarray, tau_dg: float, *, validate: bool = True) -> np.ndarray:
    """C2 binary evidence: 1 where DeltaG exceeds its detection threshold."""
    c2 = (delta_g > tau_dg).astype("float32")
    c2[~np.isfinite(delta_g)] = np.nan
    if validate:
        checks.check_nodata_propagated(delta_g, c2, "C2")
    return c2


def c3_layer(dz: np.ndarray, lod: float | np.ndarray, *,
             validate: bool = True) -> np.ndarray:
    """C3 binary evidence: 1 where the surface dropped by more than the LoD.

        C3 = 1[ dz < -lod ]

    `lod` is a single number or a per-cell raster of the same shape. It is the
    detection threshold in metres, already scaled — see `resolve_c3_lod`.
    A lod of 0 means "any lowering counts", which is what an already
    LoD-filtered dDEM wants.

    NumPy comparisons return False for NaN, so nodata is propagated explicitly:
    without this, a nodata cell silently becomes a negative observation.
    """
    lod = np.asarray(lod, dtype="float64")
    if lod.ndim and lod.shape != dz.shape:
        raise ValueError(
            f"lod raster shape {lod.shape} does not match dz shape {dz.shape}")
    if np.any(lod[np.isfinite(lod)] < 0):
        raise ValueError("lod must be >= 0 (it is subtracted from zero)")

    valid = np.isfinite(dz) & (np.isfinite(lod) if lod.ndim else np.isfinite(lod).all())
    c3 = np.full(dz.shape, np.nan, dtype="float32")
    c3[valid] = (dz[valid] < -(lod[valid] if lod.ndim else float(lod))).astype("float32")
    if validate:
        checks.check_nodata_propagated(np.where(valid, dz, np.nan), c3, "C3")
    return c3


def evidence_layers(delta_g: np.ndarray, dz: np.ndarray, tau_dg: float,
                    lod: float | np.ndarray = 0.0, *,
                    validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Binary evidence layers at the native (fine) resolution.

        C2 = 1[ dG >  tau ]   flow network reorganized
        C3 = 1[ dz < -lod ]   surface lowered by more than the detection limit

    Returns:
        (C2, C3) as float arrays of 0.0 / 1.0 / NaN.
    """
    return (c2_layer(delta_g, tau_dg, validate=validate),
            c3_layer(dz, lod, validate=validate))


def apply_domain_mask(arrays: dict[str, np.ndarray], g: np.ndarray,
                      g_min: float, g_channel: float, *,
                      elev: np.ndarray | None = None,
                      elev_min: float | None = None,
                      slope: np.ndarray | None = None,
                      slope_max_deg: float | None = None,
                      validate: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Restrict arrays to the configured analysis domain.

    The mask decides which cells are *analysed*. It must never be folded into the
    definition of what counts as erosion — see the label-circularity argument in
    the method documentation.

    Returns:
        (masked arrays, statistics)
    """
    outside = (g < g_min) | (g >= g_channel) | ~np.isfinite(g)
    stats: dict[str, Any] = {"g_min": float(g_min), "g_channel": float(g_channel)}

    if elev_min is not None:
        if elev is None:
            raise ValueError("elev array is required when elev_min is configured")
        if elev.shape != g.shape:
            raise ValueError(f"elev shape {elev.shape} does not match G shape {g.shape}")
        outside |= ~np.isfinite(elev) | (elev <= elev_min)
        stats["elev_min"] = float(elev_min)

    if slope_max_deg is not None:
        if slope is None:
            raise ValueError("slope array is required when slope_max_deg is configured")
        if slope.shape != g.shape:
            raise ValueError(f"slope shape {slope.shape} does not match G shape {g.shape}")
        slope_limit = float(np.tan(np.deg2rad(slope_max_deg)))
        outside |= ~np.isfinite(slope) | (slope > slope_limit)
        stats["slope_max_deg"] = float(slope_max_deg)
        stats["slope_max_m_per_m"] = slope_limit

    out = {}
    for name, arr in arrays.items():
        a = arr.copy()
        a[outside] = np.nan
        if validate and name in ("C2", "C3"):
            checks.check_mask_effect(arr, a, f"domain mask on {name}", warn=True)
        out[name] = a
    stats.update(n_inside=int((~outside).sum()), n_outside=int(outside.sum()))
    return out, stats


# ──────────────────────────────────────────────── aggregation

@dataclass
class PObsResult:
    """Coarse-resolution observed erosion probability and terrain forcing."""
    p_c2: np.ndarray
    p_c3: np.ndarray
    p_intersection: np.ndarray
    p_union: np.ndarray
    g_mean: np.ndarray
    g_star: np.ndarray
    block: int

    def stats(self) -> dict[str, Any]:
        finite = np.isfinite(self.p_union)
        return {
            "block_size": self.block,
            "n_valid_coarse": int(finite.sum()),
            "p_intersection_mean": float(np.nanmean(self.p_intersection)) if finite.any() else np.nan,
            "p_union_mean": float(np.nanmean(self.p_union)) if finite.any() else np.nan,
        }


def aggregate_p_obs(c2: np.ndarray, c3: np.ndarray, g: np.ndarray, block: int,
                    *, validate: bool = True) -> PObsResult:
    """Aggregate fine evidence to coarse observed erosion probability.

    Both aggregations are produced and kept separate, because the two signals
    detect different processes:

        p_intersection = mean( C2 AND C3 )                  conservative
        p_union        = P(C2) + P(C3) - P(C2 AND C3)       inclusion-exclusion

    The union is exact — P(C2 AND C3) is measured directly, so no independence
    assumption is required. It is clipped to [0, 1] because floating-point error
    in the expression can otherwise place a few cells marginally outside.

    Terrain forcing is aggregated as the cell mean, since the forecast model
    operates at the coarse resolution. The maximum is also returned for continuity
    with earlier work but is not used downstream.
    """
    valid = np.isfinite(c2) & np.isfinite(c3)
    both = np.where(valid, (c2 == 1) & (c3 == 1), np.nan).astype("float32")
    c2v = np.where(valid, c2, np.nan).astype("float32")
    c3v = np.where(valid, c3, np.nan).astype("float32")

    p_c2 = block_aggregate(c2v, block, "mean")
    p_c3 = block_aggregate(c3v, block, "mean")
    p_int = block_aggregate(both, block, "mean")
    p_uni = np.clip(p_c2 + p_c3 - p_int, 0.0, 1.0).astype("float32")

    result = PObsResult(
        p_c2=p_c2, p_c3=p_c3, p_intersection=p_int, p_union=p_uni,
        g_mean=block_aggregate(g, block, "mean"),
        g_star=block_aggregate(g, block, "max"),
        block=block,
    )
    if validate:
        for name, arr in (("p_c2", p_c2), ("p_c3", p_c3),
                          ("p_intersection", p_int), ("p_union", p_uni)):
            checks.check_range(arr, 0.0, 1.0, name)
        checks.check_not_all_nan(p_uni, "p_union")
    return result


# ──────────────────────────────────────────────── inversion

@dataclass
class GammaFit:
    """Gamma fitted to the terrain-forcing distribution on the hillslope domain."""
    k: float
    scale: float
    mean_g: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_gamma(g_mean: np.ndarray, domain: np.ndarray) -> GammaFit:
    """Fit Gamma(k, theta) to G_mean over the hillslope domain by MLE, loc fixed at 0.

    The shape k is treated as a watershed-wide property; the scale is later
    allowed to vary per cell in the inversion.
    """
    sample = g_mean[domain & np.isfinite(g_mean)]
    if sample.size < 2:
        raise ValueError(f"need at least 2 cells to fit a Gamma, got {sample.size}")
    k, _, scale = gamma_dist.fit(sample, floc=0)
    return GammaFit(k=float(k), scale=float(scale),
                    mean_g=float(k * scale), n=int(sample.size))


def invert_threshold(p_obs: np.ndarray, g_mean: np.ndarray, k: float,
                     domain: np.ndarray | None = None, *,
                     validate: bool = True) -> np.ndarray:
    """Invert observed erosion probability to a per-cell initiation threshold.

    Erosion occurs wherever forcing exceeds the local threshold, so the eroded
    fraction of a cell *is* the CDF evaluated at that cell's forcing:

        p_i = F_gamma( G_mean_i ; k, theta_i )

    The Gamma is a scale family, so applying the quantile function isolates the
    local scale and the cell's mean threshold follows:

        C_i = k * theta_i = k * G_mean_i / F_gamma^-1( p_i ; k, 1 )

    Cells with p_obs == 0 cannot be inverted (the quantile is zero and the
    threshold diverges); they return NaN. Use `threshold_at_probability` for a
    bounded estimate on those cells.
    """
    valid = np.isfinite(p_obs) & np.isfinite(g_mean) & (p_obs > 0)
    if domain is not None:
        valid &= domain

    out = np.full(g_mean.shape, np.nan, dtype="float32")
    if valid.any():
        p = np.clip(p_obs[valid], 1e-6, 1 - 1e-6)
        q = gamma_dist.ppf(p, a=k, scale=1)
        ok = q > 0
        vals = np.full(int(valid.sum()), np.nan, dtype="float32")
        vals[ok] = k * (g_mean[valid][ok] / q[ok])
        out[valid] = vals

    if validate and np.isfinite(out).any():
        finite = out[np.isfinite(out)]
        if (finite <= 0).any():
            raise checks.InvariantError(
                f"inverted threshold has {(finite <= 0).sum()} non-positive values")
    return out


def threshold_at_probability(g_mean: np.ndarray, k: float, p: float,
                             domain: np.ndarray | None = None) -> np.ndarray:
    """Threshold implied by a fixed probability, for cells that report no erosion.

    Inverting at a small fixed p gives a spatially variable upper bound through
    G_mean while holding the probability constant. Used for the p_obs == 0 cells,
    which cannot be inverted directly.
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    q = float(gamma_dist.ppf(p, a=k, scale=1))
    valid = np.isfinite(g_mean)
    if domain is not None:
        valid &= domain
    out = np.full(g_mean.shape, np.nan, dtype="float32")
    out[valid] = k * (g_mean[valid] / q)
    return out
