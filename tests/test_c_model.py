import numpy as np

from pci_source_zones.c_model import (
    bare_soil_roughness,
    critical_shear_stress,
    modeled_threshold_c,
)


def test_c_model_equation_matches_manual_value():
    params = {
        "rho_w": 1000.0,
        "rho_s": 2650.0,
        "g": 9.81,
        "tau_star": 0.043,
        "k": 0.0474,
        "p": 1 / 6,
        "m1": 0.6,
    }
    d50 = 0.002
    na = 0.03
    r = 45.0 / 1000.0 / 3600.0

    nb = params["k"] * d50 ** params["p"]
    tau_c = params["tau_star"] * params["g"] * (params["rho_s"] - params["rho_w"]) * d50
    denom = (
        params["rho_w"]
        * params["g"]
        * nb**1.5
        * (nb + na) ** (-0.9)
        * r ** params["m1"]
    )
    expected = (tau_c / denom) ** (1 / params["m1"])

    actual = modeled_threshold_c(d50, na, r, params)
    assert np.isclose(actual, expected)
    assert np.isclose(bare_soil_roughness(d50), nb)
    assert np.isclose(critical_shear_stress(d50), tau_c)
