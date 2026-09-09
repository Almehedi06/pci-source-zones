#!/usr/bin/env python
"""DEPRECATED — alignment is now config-driven.

The hardcoded layer list that lived here named Montecito-specific files, PRISM
dates and folder layout in source, which blocked every new site. It has been
replaced by `pci_source_zones.align`, driven by an alignment config.

The previous Montecito list was migrated verbatim to
`config/sites/montecito_align.yaml`, which the site config now references, so
alignment is a stage of the normal run rather than a separate task:

    python scripts/pci_zones.py run --config config/sites/montecito_2016_2018.yaml \
        --stages align

This shim still forwards to the underlying utility so existing invocations work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pci_source_zones.align import AlignConfig, run_alignment  # noqa: E402

DEFAULT = Path(__file__).resolve().parents[1] / "config/sites/montecito_align.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT),
                   help="alignment config YAML (default: Montecito)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    print("NOTE: 10_align_features.py is deprecated; alignment is a stage — "
          "use 'pci_zones.py run --config <site.yaml> --stages align'\n")
    run_alignment(AlignConfig.load(args.config), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
