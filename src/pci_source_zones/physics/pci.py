from __future__ import annotations

import numpy as np


def deterministic_pci(
    driving_index: np.ndarray, c_model: float | np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    g = np.asarray(driving_index, dtype="float64")
    c = np.asarray(c_model, dtype="float64")
    if valid_mask is None:
        valid_mask = np.isfinite(g)
    out = np.full(g.shape, np.nan, dtype="float64")
    out[valid_mask] = (c <= g)[valid_mask] if c.shape == g.shape else (c <= g[valid_mask])
    return out


def pci_from_exceedance_count(count: np.ndarray, n_samples: int, valid_mask: np.ndarray) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    pci = np.full(count.shape, np.nan, dtype="float64")
    pci[valid_mask] = count[valid_mask] / float(n_samples)
    return pci
