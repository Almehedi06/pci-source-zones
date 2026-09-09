"""Tests for the evidence pipeline core.

Synthetic arrays only. Several tests are regressions for defects found in the
script implementations these functions replace.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from pci_source_zones import evidence
from pci_source_zones.checks import InvariantError


# ─────────────────────────────────────────── detection threshold

def test_delta_g_lod_matches_nmad_definition():
    rng = np.random.default_rng(0)
    dg = rng.normal(0, 1, 10_000).astype("float32")
    stable = np.ones_like(dg, dtype=bool)
    out = evidence.delta_g_lod(dg, stable, scale=2.0)
    # NMAD of a standard normal is ~1
    assert out["nmad"] == pytest.approx(1.0, abs=0.05)
    assert out["delta_g_lod"] == pytest.approx(2 * out["nmad"])
    assert out["n_stable"] == 10_000


def test_delta_g_lod_ignores_unstable_and_nan():
    dg = np.array([0.1, 0.2, 50.0, np.nan])
    stable = np.array([True, True, False, True])
    out = evidence.delta_g_lod(dg, stable)
    assert out["n_stable"] == 2          # the 50.0 and the NaN are excluded


def test_delta_g_lod_empty_sample_raises():
    with pytest.raises(ValueError):
        evidence.delta_g_lod(np.array([1.0, 2.0]), np.array([False, False]))


# ─────────────────────────────────────────── evidence layers

def test_evidence_layers_propagate_nodata():
    """Regression: NaN comparisons return False, turning nodata into 0."""
    dg = np.array([[10.0, np.nan], [1.0, 5.0]])
    dz = np.array([[-1.0, -1.0], [np.nan, 0.5]])
    c2, c3 = evidence.evidence_layers(dg, dz, tau_dg=2.0)
    assert c2[0, 0] == 1.0
    assert np.isnan(c2[0, 1]), "nodata in delta_g must stay NaN in C2"
    assert np.isnan(c3[1, 0]), "nodata in dz must stay NaN in C3"
    assert c3[0, 0] == 1.0 and c3[1, 1] == 0.0


def test_evidence_layers_threshold_is_strict():
    dg = np.array([2.0, 2.001])
    dz = np.zeros(2)
    c2, _ = evidence.evidence_layers(dg, dz, tau_dg=2.0)
    assert c2[0] == 0.0 and c2[1] == 1.0


def test_c3_per_cell_lod_uses_spatial_threshold():
    dz = np.array([-0.30, -0.10, -1.00, np.nan], dtype="float32")
    lod = np.array([0.10, 0.10, 0.60, 0.10]) * 1.96
    c3 = evidence.c3_layer(dz, lod)
    assert list(c3[:3]) == [1.0, 0.0, 0.0]
    assert np.isnan(c3[3])


def test_c3_lod_zero_counts_any_lowering():
    """An already LoD-filtered dDEM wants lod = 0."""
    c3 = evidence.c3_layer(np.array([-0.01, 0.0, 0.01]), 0.0)
    assert list(c3) == [1.0, 0.0, 0.0]


def test_c3_constant_lod():
    c3 = evidence.c3_layer(np.array([-0.30, -0.10]), 0.175)
    assert list(c3) == [1.0, 0.0]


def test_c3_rejects_negative_lod():
    with pytest.raises(ValueError, match="lod must be >= 0"):
        evidence.c3_layer(np.array([-1.0]), -0.1)


def test_c3_rejects_mismatched_lod_raster():
    with pytest.raises(ValueError, match="does not match"):
        evidence.c3_layer(np.zeros((4, 4)), np.zeros((2, 2)))


def test_nmad_is_median_centred():
    """A residual vertical bias must not inflate the noise estimate."""
    rng = np.random.default_rng(0)
    unbiased = evidence.nmad(rng.normal(0.0, 0.1, 50_000))
    biased = evidence.nmad(rng.normal(5.0, 0.1, 50_000))
    assert abs(unbiased - biased) < 0.005


def test_detection_threshold_shared_by_both_signals():
    """C2 and C3 use one rule, so the same input gives the same threshold."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.2, 20_000)
    stable = np.ones_like(x, dtype=bool)
    a = evidence.detection_threshold(x, stable, 2.0, name="DeltaG")
    b = evidence.detection_threshold(x, stable, 2.0, name="dz")
    assert a["lod"] == b["lod"]
    assert abs(a["lod"] - 0.4) < 0.02


# ─────────────────────────────────────────── domain mask

def test_domain_mask_excludes_outside_range():
    g = np.array([[1.0, 50.0], [400.0, np.nan]])
    c2 = np.ones((2, 2), dtype="float32")
    out, stats = evidence.apply_domain_mask({"C2": c2}, g, 5, 300, validate=False)
    assert np.isnan(out["C2"][0, 0])     # below g_min
    assert out["C2"][0, 1] == 1.0        # inside
    assert np.isnan(out["C2"][1, 0])     # channel
    assert np.isnan(out["C2"][1, 1])     # nodata terrain
    assert stats["n_inside"] == 1


def test_domain_mask_applies_elevation_and_slope_filters():
    g = np.full((2, 2), 50.0, dtype="float32")
    c2 = np.ones((2, 2), dtype="float32")
    elev = np.array([[700.0, 500.0], [700.0, 700.0]], dtype="float32")
    slope = np.array([[0.1, 0.1], [2.0, 0.1]], dtype="float32")
    out, stats = evidence.apply_domain_mask(
        {"C2": c2}, g, 5, 300,
        elev=elev, elev_min=600,
        slope=slope, slope_max_deg=45,
        validate=False,
    )
    assert out["C2"][0, 0] == 1.0
    assert np.isnan(out["C2"][0, 1])      # below elevation floor
    assert np.isnan(out["C2"][1, 0])      # above slope ceiling
    assert stats["n_inside"] == 2


def test_domain_mask_inert_filter_warns(capsys):
    """Regression: a filter that removes nothing should be visible."""
    g = np.full((3, 3), 50.0)            # everything inside the domain
    c2 = np.ones((3, 3), dtype="float32")
    evidence.apply_domain_mask({"C2": c2}, g, 5, 300, validate=True)
    assert "inert" in capsys.readouterr().out


# ─────────────────────────────────────────── aggregation

def test_aggregate_p_obs_shapes_and_values():
    c2 = np.ones((4, 4), dtype="float32")
    c3 = np.zeros((4, 4), dtype="float32")
    c3[:2, :2] = 1.0
    g = np.full((4, 4), 30.0, dtype="float32")
    r = evidence.aggregate_p_obs(c2, c3, g, block=2)
    assert r.p_union.shape == (2, 2)
    assert r.p_intersection[0, 0] == pytest.approx(1.0)   # both fire everywhere
    assert r.p_intersection[1, 1] == pytest.approx(0.0)
    assert r.g_mean[0, 0] == pytest.approx(30.0)


def test_union_is_inclusion_exclusion():
    rng = np.random.default_rng(1)
    c2 = (rng.random((10, 10)) > 0.5).astype("float32")
    c3 = (rng.random((10, 10)) > 0.5).astype("float32")
    g = np.full((10, 10), 20.0, dtype="float32")
    r = evidence.aggregate_p_obs(c2, c3, g, block=10)
    expected = r.p_c2[0, 0] + r.p_c3[0, 0] - r.p_intersection[0, 0]
    assert r.p_union[0, 0] == pytest.approx(expected, abs=1e-6)


def test_union_clipped_to_unit_interval():
    """Regression: floating point put 17 cells marginally above 1."""
    c2 = np.ones((4, 4), dtype="float32")
    c3 = np.ones((4, 4), dtype="float32")
    g = np.full((4, 4), 10.0, dtype="float32")
    r = evidence.aggregate_p_obs(c2, c3, g, block=2)
    assert r.p_union.max() <= 1.0


def test_aggregate_all_nodata_raises():
    nan = np.full((4, 4), np.nan, dtype="float32")
    with pytest.raises(InvariantError):
        evidence.aggregate_p_obs(nan, nan, nan, block=2)


# ─────────────────────────────────────────── Gamma fit and inversion

def test_fit_gamma_recovers_known_parameters():
    rng = np.random.default_rng(2)
    sample = gamma_dist.rvs(a=2.1, scale=13.7, size=20_000, random_state=rng)
    g = sample.reshape(200, 100).astype("float32")
    fit = evidence.fit_gamma(g, np.ones_like(g, dtype=bool))
    assert fit.k == pytest.approx(2.1, rel=0.1)
    assert fit.scale == pytest.approx(13.7, rel=0.1)
    assert fit.mean_g == pytest.approx(fit.k * fit.scale)


def test_fit_gamma_too_few_cells_raises():
    with pytest.raises(ValueError):
        evidence.fit_gamma(np.array([[5.0]]), np.ones((1, 1), dtype=bool))


def test_invert_threshold_matches_closed_form():
    k, g_val, p_val = 2.129, 30.0, 0.25
    g = np.full((2, 2), g_val, dtype="float32")
    p = np.full((2, 2), p_val, dtype="float32")
    out = evidence.invert_threshold(p, g, k)
    expected = k * g_val / gamma_dist.ppf(p_val, a=k, scale=1)
    assert out[0, 0] == pytest.approx(expected, rel=1e-5)


def test_invert_threshold_is_monotonic_decreasing_in_p():
    k = 2.129
    g = np.full(4, 30.0, dtype="float32")
    p = np.array([0.05, 0.25, 0.5, 0.75], dtype="float32")
    out = evidence.invert_threshold(p, g, k)
    assert np.all(np.diff(out) < 0), "more erosion must imply a lower threshold"


def test_invert_threshold_returns_nan_for_zero_probability():
    g = np.full((2, 2), 30.0, dtype="float32")
    p = np.zeros((2, 2), dtype="float32")
    out = evidence.invert_threshold(p, g, 2.129)
    assert np.all(np.isnan(out))


def test_invert_threshold_respects_domain():
    g = np.full((2, 2), 30.0, dtype="float32")
    p = np.full((2, 2), 0.5, dtype="float32")
    domain = np.array([[True, False], [False, False]])
    out = evidence.invert_threshold(p, g, 2.129, domain=domain)
    assert np.isfinite(out[0, 0]) and np.isnan(out[0, 1])


def test_global_mean_recovered_when_p_matches_theory():
    """A cell eroding exactly as the global fit predicts returns k*theta."""
    k, theta = 2.129, 13.698
    g_val = 30.0
    p_theory = gamma_dist.cdf(g_val, a=k, scale=theta)
    out = evidence.invert_threshold(np.array([p_theory]), np.array([g_val]), k)
    assert out[0] == pytest.approx(k * theta, rel=1e-4)


def test_threshold_at_probability_scales_with_g():
    k = 2.129
    g = np.array([10.0, 20.0, 40.0], dtype="float32")
    out = evidence.threshold_at_probability(g, k, 0.05)
    assert out[1] == pytest.approx(2 * out[0], rel=1e-5)
    assert out[2] == pytest.approx(4 * out[0], rel=1e-5)


def test_threshold_at_probability_rejects_bad_p():
    g = np.array([10.0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            evidence.threshold_at_probability(g, 2.0, bad)


def test_upper_bound_exceeds_observed_threshold():
    """The p=0.05 bound must be higher than a threshold from a larger p."""
    k = 2.129
    g = np.full(3, 30.0, dtype="float32")
    observed = evidence.invert_threshold(np.full(3, 0.25, dtype="float32"), g, k)
    bound = evidence.threshold_at_probability(g, k, 0.05)
    assert np.all(bound > observed)
