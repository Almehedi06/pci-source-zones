from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config, output_path  # noqa: E402
from pci_source_zones.inputs import read_raster  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print basic PCI output checks.")
    parser.add_argument("--config", default="config/montecito.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pci_path = output_path(cfg, "pci", "pci.tif")
    pci, _profile = read_raster(pci_path)
    valid = pci == pci
    print(f"pci: {pci_path}")
    print(f"valid_cells: {int(valid.sum())}")
    if valid.any():
        print(f"pci_min: {float(pci[valid].min()):.4f}")
        print(f"pci_mean: {float(pci[valid].mean()):.4f}")
        print(f"pci_max: {float(pci[valid].max()):.4f}")


if __name__ == "__main__":
    main()
