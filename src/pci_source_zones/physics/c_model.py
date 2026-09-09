from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1.0e-12


def bare_soil_roughness(
    d50_m: float | np.ndarray, k: float = 0.0474, p: float = 1.0 / 6.0
) -> float | np.ndarray:
    d50 = np.maximum(np.asarray(d50_m, dtype="float64"), EPS)
    return k * np.power(d50, p)


def critical_shear_stress(
    d50_m: float | np.ndarray,
    tau_star: float = 0.043,
    g: float = 9.81,
    rho_s: float = 2650.0,
    rho_w: float = 1000.0,
) -> float | np.ndarray:
    d50 = np.maximum(np.asarray(d50_m, dtype="float64"), EPS)
    return tau_star * g * (rho_s - rho_w) * d50


def modeled_threshold_c(
    d50_m: float | np.ndarray,
    na: float | np.ndarray,
    rainfall_excess_m_s: float | np.ndarray,
    params: dict[str, Any],
) -> float | np.ndarray:
    """Compute modeled channel-initiation threshold C.

    C = [tau_c / {rho_w*g*nb^1.5*(nb+na)^(-0.9)*r^m1}]^(1/m1)
    """

    m1 = float(params.get("m1", 0.6))
    rho_w = float(params.get("rho_w", 1000.0))
    g = float(params.get("g", 9.81))

    d50 = np.maximum(np.asarray(d50_m, dtype="float64"), EPS)
    added_roughness = np.maximum(np.asarray(na, dtype="float64"), 0.0)
    runoff = np.maximum(np.asarray(rainfall_excess_m_s, dtype="float64"), EPS)

    nb = bare_soil_roughness(d50, float(params.get("k", 0.0474)), float(params.get("p", 1.0 / 6.0)))
    tau_c = critical_shear_stress(
        d50,
        tau_star=float(params.get("tau_star", 0.043)),
        g=g,
        rho_s=float(params.get("rho_s", 2650.0)),
        rho_w=rho_w,
    )

    denominator = (
        rho_w
        * g
        * np.power(nb, 1.5)
        * np.power(nb + added_roughness, -0.9)
        * np.power(runoff, m1)
    )
    denominator = np.maximum(denominator, EPS)
    return np.power(tau_c / denominator, 1.0 / m1)
