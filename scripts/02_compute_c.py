from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.pipeline import compute_c_and_pci  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean modeled C raster.")
    parser.add_argument("--config", default="config/montecito.yaml")
    args = parser.parse_args()
    outputs = compute_c_and_pci(load_config(args.config))
    print(f"c_model_mean: {outputs['c_model_mean']}")
    print(f"pci: {outputs['pci']} (computed as a dependency)")


if __name__ == "__main__":
    main()
