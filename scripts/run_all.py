from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.physics.pipeline import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full PCI source-zone workflow.")
    parser.add_argument("--config", default="config/legacy/montecito_physics.yaml")
    args = parser.parse_args()
    print(run_all(args.config))


if __name__ == "__main__":
    main()
