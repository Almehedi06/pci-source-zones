from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.source_area import run_source_area_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create source candidate rasters and polygons."
    )
    parser.add_argument("--config", default="config/montecito.yaml")
    args = parser.parse_args()

    result = run_source_area_workflow(load_config(args.config))
    print(f"output_dir: {result['output_dir']}")
    print(f"pci: {result['pci']}")
    print(f"erosion_mask: {result['erosion_mask']}")
    print(f"source_cell_summary: {result['source_cell_summary']}")
    print(f"source_polygon_summary: {result['source_polygon_summary']}")
    print(f"c_band_cell_summary: {result['c_band_cell_summary']}")
    print(f"c_band_polygon_summary: {result['c_band_polygon_summary']}")
    for top_percent, path in result["source_binaries"].items():
        print(f"top {top_percent}% binary: {path}")
    for top_percent, path in result["source_polygons"].items():
        print(f"top {top_percent}% polygons: {path}")
    for c_min, path in result["c_band_binaries"].items():
        print(f"Cmin {c_min} band binary: {path}")
    for c_min, path in result["c_band_polygons"].items():
        print(f"Cmin {c_min} band polygons: {path}")


if __name__ == "__main__":
    main()
