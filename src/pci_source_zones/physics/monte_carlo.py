from __future__ import annotations

from typing import Any

import numpy as np

from .c_model import modeled_threshold_c
from .inputs import InputProvider
from .pci import pci_from_exceedance_count


def compute_pci_monte_carlo(
    driving_index: np.ndarray,
    d50: InputProvider,
    na: InputProvider,
    rainfall_excess: InputProvider,
    params: dict[str, Any],
    n_samples: int = 1500,
    seed: int = 42,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCI raster and mean C raster without storing the full C ensemble."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    g = np.asarray(driving_index, dtype="float64")
    if valid_mask is None:
        valid_mask = np.isfinite(g)

    rng = np.random.default_rng(seed)
    exceedance_count = np.zeros(g.shape, dtype="uint32")
    c_sum = np.zeros(g.shape, dtype="float64")

    for _ in range(n_samples):
        c = modeled_threshold_c(
            d50.sample(rng),
            na.sample(rng),
            rainfall_excess.sample(rng),
            params,
        )
        c_arr = np.broadcast_to(np.asarray(c, dtype="float64"), g.shape)
        exceedance_count += ((c_arr <= g) & valid_mask).astype("uint32")
        c_sum[valid_mask] += c_arr[valid_mask]

    pci = pci_from_exceedance_count(exceedance_count, n_samples, valid_mask)
    c_mean = np.full(g.shape, np.nan, dtype="float64")
    c_mean[valid_mask] = c_sum[valid_mask] / float(n_samples)
    return pci, c_mean
