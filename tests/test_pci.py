import numpy as np

from pci_source_zones.pci import deterministic_pci, pci_from_exceedance_count


def test_deterministic_pci():
    g = np.array([[1.0, 3.0], [np.nan, 5.0]])
    c = 2.0
    pci = deterministic_pci(g, c)
    assert np.array_equal(pci[np.isfinite(pci)], np.array([0.0, 1.0, 1.0]))
    assert np.isnan(pci[1, 0])


def test_pci_from_exceedance_count():
    count = np.array([[0, 5], [10, 2]], dtype="uint32")
    valid = np.array([[True, True], [True, False]])
    pci = pci_from_exceedance_count(count, 10, valid)
    assert np.allclose(pci[valid], [0.0, 0.5, 1.0])
    assert np.isnan(pci[1, 1])
