"""Invariant assertions for the evidence and inversion pipeline.

Every defect found in this pipeline so far has been *silent* — a plausible wrong
number rather than a crash:

    1. `NaN > threshold` evaluates False in NumPy, so nodata pixels became
       "no erosion" instead of being excluded.
    2. A 10 m channel filter tested a quantity that could never cross its
       threshold, so it masked zero cells while appearing to work.
    3. Floating-point in an inclusion-exclusion expression produced
       probabilities marginally above 1.
    4. `np.nansum` over an all-NaN slice returns 0.0, turning "no coverage"
       into "zero rainfall".

These helpers make each of those loud. They are cheap; call them liberally.

Every function raises `InvariantError` on failure. Use `warn=True` to downgrade
to a printed warning where a hard failure would be inappropriate.
"""
from __future__ import annotations

import numpy as np


class InvariantError(AssertionError):
    """A pipeline invariant was violated."""


def _fail(msg: str, warn: bool) -> None:
    if warn:
        print(f"  WARNING: {msg}")
    else:
        raise InvariantError(msg)


def check_range(arr: np.ndarray, lo: float, hi: float, name: str,
                *, warn: bool = False) -> None:
    """Finite values of `arr` must lie within [lo, hi].

    Catches defect 3 — probabilities drifting outside [0, 1].
    """
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return
    if v.min() < lo or v.max() > hi:
        n_lo = int((v < lo).sum())
        n_hi = int((v > hi).sum())
        _fail(f"{name}: {n_lo} values below {lo}, {n_hi} above {hi} "
              f"(range {v.min():.6g} to {v.max():.6g})", warn)


def check_mask_effect(before: np.ndarray, after: np.ndarray, name: str,
                      *, expect_removal: bool = True, warn: bool = False) -> int:
    """Report how many valid cells a mask removed, and fail if it removed none.

    Catches defect 2 — a filter that silently does nothing. A mask that is
    configured and applied but removes zero cells is almost always a bug in the
    quantity being tested, not a landscape without any cells to remove.

    Returns:
        number of valid cells removed.
    """
    n_before = int(np.isfinite(before).sum())
    n_after = int(np.isfinite(after).sum())
    removed = n_before - n_after
    if removed < 0:
        _fail(f"{name}: mask *added* {-removed} valid cells", warn)
    elif expect_removal and removed == 0:
        _fail(f"{name}: removed 0 of {n_before} valid cells — "
              "the filter is inert; check the quantity being thresholded", warn)
    return removed


def check_nodata_propagated(source: np.ndarray, derived: np.ndarray, name: str,
                            *, warn: bool = False) -> None:
    """Wherever `source` is NaN, `derived` must be NaN too.

    Catches defect 1 — comparison operators return False for NaN, so a derived
    boolean layer silently becomes 0 where its input had no data.
    """
    leaked = int((~np.isfinite(source) & np.isfinite(derived)).sum())
    if leaked:
        _fail(f"{name}: {leaked} cells have data where the source is nodata — "
              "nodata was not propagated through the comparison", warn)


def check_not_all_nan(arr: np.ndarray, name: str, *, warn: bool = False) -> None:
    """`arr` must contain at least one finite value."""
    if not np.isfinite(arr).any():
        _fail(f"{name}: contains no finite values", warn)


def check_sum_preserves_nodata(cube: np.ndarray, total: np.ndarray, name: str,
                               *, warn: bool = False) -> None:
    """Cells with no valid layer must be NaN in the total, not zero.

    Catches defect 4 — `np.nansum` collapsing all-NaN slices to 0.0.
    """
    no_data = ~np.isfinite(cube).any(axis=0)
    bad = int((no_data & np.isfinite(total)).sum())
    if bad:
        _fail(f"{name}: {bad} cells with no valid input are finite in the sum — "
              "all-NaN slices became 0.0; use raster.nansum_masked", warn)


def check_grid_shape(arr: np.ndarray, expected: tuple[int, int], name: str,
                     *, warn: bool = False) -> None:
    """Array shape must match the expected grid."""
    if tuple(arr.shape) != tuple(expected):
        _fail(f"{name}: shape {arr.shape} does not match expected {expected}", warn)


def check_min_valid(arr: np.ndarray, minimum: int, name: str,
                    *, warn: bool = False) -> None:
    """At least `minimum` finite cells must remain.

    Useful after a chain of masks, where each step is individually reasonable
    but the intersection leaves too little to fit anything on.
    """
    n = int(np.isfinite(arr).sum())
    if n < minimum:
        _fail(f"{name}: only {n} valid cells remain, need at least {minimum}", warn)


def check_monotonic_coverage(stages: dict[str, np.ndarray], *, warn: bool = False) -> None:
    """Valid-cell counts must not increase through a masking sequence.

    Args:
        stages: ordered {label: array}. Python dicts preserve insertion order.
    """
    prev_name, prev_n = None, None
    for name, arr in stages.items():
        n = int(np.isfinite(arr).sum())
        if prev_n is not None and n > prev_n:
            _fail(f"coverage increased from {prev_name} ({prev_n}) to {name} ({n}) — "
                  "a masking step should never add valid cells", warn)
        prev_name, prev_n = name, n


def summarize_coverage(stages: dict[str, np.ndarray]) -> str:
    """Human-readable valid-cell count through a masking sequence.

    Print this in pipeline logs so a filter that does nothing is visible in the
    output rather than having to be looked for.
    """
    lines, prev = [], None
    for name, arr in stages.items():
        n = int(np.isfinite(arr).sum())
        delta = "" if prev is None else f"  ({n - prev:+,})"
        lines.append(f"    {name:<28s} {n:>10,}{delta}")
        prev = n
    return "\n".join(lines)
