from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.terrain import prepare_terrain  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute slope, specific area, and G.")
    parser.add_argument("--config", default="config/montecito.yaml")
    args = parser.parse_args()
    outputs = prepare_terrain(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
