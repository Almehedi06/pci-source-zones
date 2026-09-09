from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, output_path
from .inputs import InputProvider, read_raster, write_raster
from .monte_carlo import compute_pci_monte_carlo
from .polygons import extract_source_polygons
from .terrain import prepare_terrain


def compute_c_and_pci(cfg: dict[str, Any]) -> dict[str, Path]:
    g_path = output_path(cfg, "topographic_driving_index", "topographic_driving_index.tif")
    driving_index, profile = read_raster(g_path)
    valid = np.isfinite(driving_index)

    inputs = cfg["inputs"]
    d50 = InputProvider(inputs["d50"], "d50", cfg, reference_shape=driving_index.shape)
    na = InputProvider(inputs["na"], "na", cfg, reference_shape=driving_index.shape)
    rainfall = InputProvider(
        inputs["rainfall_excess"],
        "rainfall_excess",
        cfg,
        reference_shape=driving_index.shape,
    )

    mc = cfg.get("monte_carlo", {})
    pci, c_mean = compute_pci_monte_carlo(
        driving_index,
        d50,
        na,
        rainfall,
        cfg.get("parameters", {}),
        n_samples=int(mc.get("n_samples", 1500)),
        seed=int(mc.get("seed", 42)),
        valid_mask=valid,
    )

    c_path = output_path(cfg, "c_model_mean", "c_model_mean.tif")
    pci_path = output_path(cfg, "pci", "pci.tif")
    write_raster(c_path, c_mean, profile)
    write_raster(pci_path, pci, profile)
    return {"c_model_mean": c_path, "pci": pci_path}


def extract_sources(cfg: dict[str, Any]):
    pci_path = output_path(cfg, "pci", "pci.tif")
    pci, profile = read_raster(pci_path)
    out_path = output_path(cfg, "source_polygons", "source_polygons.gpkg")
    outputs = cfg.get("outputs", {})
    return extract_source_polygons(
        pci,
        profile,
        threshold=float(outputs.get("pci_threshold", 0.7)),
        min_area_m2=float(outputs.get("min_area_m2", 0.0)),
        out_path=out_path,
    )


def run_all(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    terrain_outputs = prepare_terrain(cfg)
    raster_outputs = compute_c_and_pci(cfg)
    polygons = extract_sources(cfg)
    return {
        "terrain": terrain_outputs,
        "rasters": raster_outputs,
        "n_source_polygons": len(polygons),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PCI source-zone workflow.")
    parser.add_argument("--config", default="config/montecito.yaml")
    args = parser.parse_args(argv)
    result = run_all(args.config)
    print(result)


if __name__ == "__main__":
    main()
