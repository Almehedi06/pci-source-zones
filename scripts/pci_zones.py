#!/usr/bin/env python
"""Launcher for `pci-zones`. Prefer the installed command:

    pip install -e .
    pci-zones run --config config/sites/<site>.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pci_source_zones.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
