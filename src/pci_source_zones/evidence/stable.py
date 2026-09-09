"""Stable-ground mask: the sample the detection threshold is estimated from.

The mask decides which cells are treated as "no real change", so it sets the
NMAD, and therefore tau_dG and the whole C2 layer. Getting it wrong biases every
downstream product, quietly.

Two methods:

    threshold   |layer| < t for each named change layer, combined with AND/OR.
                Simple, but CIRCULAR when the layers are the same change signals
                whose detection threshold is being estimated: thresholding |dG|
                to pick the cells used to estimate the LoD for dG selects the
                small-residual tail by construction and biases tau low.

    criteria    stable ground from predictors that are independent of the change
                signal — outside the burn perimeter, low slope, non-channel.
                This is the standard dDEM practice (Nuth & Kaeaeb; Lane et al.)
                and is the recommended default.

`threshold` stays available because it is what existing products were built on,
but it must be an explicit choice, not something a caller falls into.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def threshold_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    """1.0 where |arr| < threshold, 0.0 where valid but not, NaN where invalid."""
    out = np.full(arr.shape, np.nan, dtype="float64")
    valid = np.isfinite(arr)
    out[valid] = (np.abs(arr[valid]) < threshold).astype("float64")
    return out


def range_mask(arr: np.ndarray, lo: float | None = None,
               hi: float | None = None) -> np.ndarray:
    """1.0 where lo <= arr < hi, 0.0 where valid but outside, NaN where invalid."""
    out = np.full(arr.shape, np.nan, dtype="float64")
    valid = np.isfinite(arr)
    keep = np.ones(int(valid.sum()), dtype=bool)
    v = arr[valid]
    if lo is not None:
        keep &= v >= lo
    if hi is not None:
        keep &= v < hi
    out[valid] = keep.astype("float64")
    return out


def combine_masks(stacked: np.ndarray, mode: str = "all") -> np.ndarray:
    """Combine {0, 1, NaN} masks. `all` is AND, `any` is OR.

    Under `all` a cell is only decided where every layer is valid, so a missing
    layer produces NaN rather than a silently permissive True.
    """
    if stacked.ndim != 3:
        raise ValueError(f"expected (n_layers, H, W), got shape {stacked.shape}")
    valid = np.isfinite(stacked)
    if mode == "all":
        decided = np.all(valid, axis=0)
        result = np.all(np.where(valid, stacked, 1.0) == 1.0, axis=0)
    elif mode == "any":
        decided = np.any(valid, axis=0)
        result = np.any(np.where(valid, stacked, 0.0) == 1.0, axis=0)
    else:
        raise ValueError(f"combine must be 'all' or 'any', got {mode!r}")
    return np.where(decided, result.astype("float64"), np.nan)


def stable_stats(mask: np.ndarray) -> dict[str, Any]:
    n_valid = int(np.isfinite(mask).sum())
    n_stable = int(np.nansum(mask == 1.0))
    return {
        "n_valid": n_valid,
        "n_stable": n_stable,
        "frac_stable": (n_stable / n_valid) if n_valid else float("nan"),
    }


def check_representative(mask: np.ndarray, covariate: np.ndarray, name: str,
                         *, min_frac: float = 0.05) -> dict[str, Any]:
    """Warn when the stable sample sits in an unrepresentative part of a covariate.

    A mask concentrated in one terrain type gives an NMAD that does not describe
    error where erosion actually happens. Compares the stable sample's median
    against the full valid population's.
    """
    stable = np.isfinite(mask) & (mask == 1.0) & np.isfinite(covariate)
    population = np.isfinite(covariate)
    out: dict[str, Any] = {"covariate": name}
    if not stable.any() or not population.any():
        out["warning"] = f"no overlap between stable mask and {name}"
        return out
    out["median_stable"] = float(np.median(covariate[stable]))
    out["median_all"] = float(np.median(covariate[population]))
    out["frac_of_population"] = float(stable.sum() / population.sum())
    if out["frac_of_population"] < min_frac:
        out["warning"] = (
            f"stable sample is {100 * out['frac_of_population']:.1f}% of valid "
            f"{name} cells (< {100 * min_frac:.0f}%); NMAD may not describe error "
            f"where erosion occurs")
    return out
