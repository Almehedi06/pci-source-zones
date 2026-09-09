"""Physics track: terrain forcing, the resistance model, and PCI.

This is the mechanistic half of the repository. It takes slope and contributing
area to a terrain forcing G, converts a critical shear stress to a resistance
threshold C, and compares the two — deterministically or by Monte Carlo — to get
a probability of coarse initiation.

    terrain      slope, flow accumulation, G = a * S^alpha
    c_model      bare-soil roughness, critical shear stress, threshold C
    pci          G vs C comparison, deterministic and from exceedance counts
    monte_carlo  parameter sampling around c_model
    polygons     raster source masks to vector polygons
    source_area  the source-area workflow built on the above
    pipeline     stage orchestration for the physics track

`config` and `inputs` deliberately stay in the parent package: the evidence and
ML tracks import them too, so they are foundation, not physics.
"""
from .c_model import bare_soil_roughness, critical_shear_stress, modeled_threshold_c
from .monte_carlo import compute_pci_monte_carlo
from .pci import deterministic_pci, pci_from_exceedance_count
from .pipeline import compute_c_and_pci, extract_sources, run_all
from .polygons import extract_source_polygons
from .source_area import run_source_area_workflow, write_binary_raster
from .terrain import prepare_terrain

__all__ = [
    "bare_soil_roughness",
    "compute_c_and_pci",
    "compute_pci_monte_carlo",
    "critical_shear_stress",
    "deterministic_pci",
    "extract_source_polygons",
    "extract_sources",
    "modeled_threshold_c",
    "pci_from_exceedance_count",
    "prepare_terrain",
    "run_all",
    "run_source_area_workflow",
    "write_binary_raster",
]
