"""Tests for the pipeline invariant assertions.

Each test names the real defect it guards against.
"""
from __future__ import annotations

import numpy as np
import pytest

from pci_source_zones import checks
from pci_source_zones.checks import InvariantError


# ── defect 1: NaN comparison returning False ────────────────────────

def test_nodata_propagated_catches_leak():
    """`NaN > threshold` is False, so the derived layer becomes 0 not NaN."""
    source = np.array([[1.0, np.nan], [3.0, 4.0]])
    derived = (source > 2).astype("float32")        # the bug: NaN -> 0.0
    with pytest.raises(InvariantError) as exc:
        checks.check_nodata_propagated(source, derived, "C2")
    assert "nodata was not propagated" in str(exc.value)


def test_nodata_propagated_passes_when_handled():
    source = np.array([[1.0, np.nan], [3.0, 4.0]])
    derived = (source > 2).astype("float32")
    derived[~np.isfinite(source)] = np.nan          # the fix
    checks.check_nodata_propagated(source, derived, "C2")


# ── defect 2: inert filter ──────────────────────────────────────────

def test_mask_effect_fails_when_filter_removes_nothing():
    """A configured filter that masks zero cells is a bug, not a landscape."""
    arr = np.ones((4, 4))
    with pytest.raises(InvariantError) as exc:
        checks.check_mask_effect(arr, arr.copy(), "channel mask")
    assert "inert" in str(exc.value)


def test_mask_effect_returns_removed_count():
    before = np.ones((4, 4))
    after = before.copy()
    after[0, :] = np.nan
    assert checks.check_mask_effect(before, after, "mask") == 4


def test_mask_effect_allows_no_removal_when_declared():
    arr = np.ones((3, 3))
    assert checks.check_mask_effect(arr, arr.copy(), "optional",
                                    expect_removal=False) == 0


def test_mask_effect_rejects_added_cells():
    before = np.array([[np.nan, 1.0]])
    after = np.array([[2.0, 1.0]])
    with pytest.raises(InvariantError) as exc:
        checks.check_mask_effect(before, after, "mask")
    assert "added" in str(exc.value)


# ── defect 3: value out of range ────────────────────────────────────

def test_check_range_catches_probability_above_one():
    p = np.array([0.1, 0.5, 1.0000003])
    with pytest.raises(InvariantError) as exc:
        checks.check_range(p, 0.0, 1.0, "p_obs_union")
    assert "above 1" in str(exc.value)


def test_check_range_passes_within_bounds():
    checks.check_range(np.array([0.0, 0.5, 1.0]), 0.0, 1.0, "p")


def test_check_range_ignores_nan():
    checks.check_range(np.array([0.5, np.nan]), 0.0, 1.0, "p")


# ── defect 4: nansum collapsing all-NaN to zero ─────────────────────

def test_sum_preserves_nodata_catches_zero_fill():
    cube = np.full((3, 2, 2), np.nan)
    cube[:, 0, 0] = 1.0
    bad_total = np.nansum(cube, axis=0)              # the bug: NaN cells -> 0.0
    with pytest.raises(InvariantError) as exc:
        checks.check_sum_preserves_nodata(cube, bad_total, "total")
    assert "became 0.0" in str(exc.value)


def test_sum_preserves_nodata_passes_when_masked():
    from pci_source_zones.raster import nansum_masked
    cube = np.full((3, 2, 2), np.nan)
    cube[:, 0, 0] = 1.0
    checks.check_sum_preserves_nodata(cube, nansum_masked(cube), "total")


# ── general guards ──────────────────────────────────────────────────

def test_not_all_nan():
    with pytest.raises(InvariantError):
        checks.check_not_all_nan(np.full((2, 2), np.nan), "layer")
    checks.check_not_all_nan(np.array([[np.nan, 1.0]]), "layer")


def test_grid_shape():
    with pytest.raises(InvariantError):
        checks.check_grid_shape(np.ones((3, 3)), (4, 4), "layer")
    checks.check_grid_shape(np.ones((3, 3)), (3, 3), "layer")


def test_min_valid():
    with pytest.raises(InvariantError) as exc:
        checks.check_min_valid(np.array([1.0, np.nan, np.nan]), 2, "domain")
    assert "only 1 valid" in str(exc.value)
    checks.check_min_valid(np.array([1.0, 2.0]), 2, "domain")


def test_monotonic_coverage_detects_increase():
    stages = {
        "raw": np.array([1.0, np.nan]),
        "masked": np.array([1.0, 2.0]),      # gained a cell
    }
    with pytest.raises(InvariantError) as exc:
        checks.check_monotonic_coverage(stages)
    assert "increased" in str(exc.value)


def test_monotonic_coverage_passes_when_shrinking():
    checks.check_monotonic_coverage({
        "raw": np.array([1.0, 2.0, 3.0]),
        "g_mask": np.array([1.0, 2.0, np.nan]),
        "aoi": np.array([1.0, np.nan, np.nan]),
    })


def test_warn_mode_does_not_raise(capsys):
    checks.check_range(np.array([2.0]), 0.0, 1.0, "p", warn=True)
    assert "WARNING" in capsys.readouterr().out


def test_summarize_coverage_shows_deltas():
    out = checks.summarize_coverage({
        "raw": np.ones(100),
        "after mask": np.concatenate([np.ones(60), np.full(40, np.nan)]),
    })
    assert "100" in out and "60" in out and "-40" in out
