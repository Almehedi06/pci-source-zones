from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config, output_path  # noqa: E402
from pci_source_zones.pipeline import extract_sources  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PCI source polygons.")
    parser.add_argument("--config", default="config/montecito.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    gdf = extract_sources(cfg)
    print(f"source_polygons: {output_path(cfg, 'source_polygons', 'source_polygons.gpkg')}")
    print(f"count: {len(gdf)}")


if __name__ == "__main__":
    main()
