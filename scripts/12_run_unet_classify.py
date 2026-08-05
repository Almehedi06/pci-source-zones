"""Run UNet binary classification for postfire source zones.

Usage:
    conda run -n ml_debris python scripts/12_run_unet_classify.py --config config/ml_unet.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pci_source_zones.config import load_config
from pci_source_zones.ml.unet_classify_workflow import run_unet_classify_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet classification pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_unet_classify_workflow(cfg)


if __name__ == "__main__":
    main()
