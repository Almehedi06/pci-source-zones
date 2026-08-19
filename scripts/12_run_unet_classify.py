"""Run UNet binary classification for postfire source zones.

Thin wrapper around the same run_unet_workflow() used by 09_run_unet.py —
it already branches on ml.target.type to do classification (BCE+Dice loss)
vs regression (MSE/Tobit) automatically. This script is just a convenience
entry point defaulting to a classification-flavored config.

Usage:
    conda run -n ml_debris python scripts/12_run_unet_classify.py --config config/ml_unet.yaml
"""
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.ml.unet_workflow import run_unet_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet classification pipeline")
    parser.add_argument("--config", default="config/ml_unet.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_unet_workflow(cfg)

    print(f"model_name      : {result['model_name']}")
    print(f"run_id          : {result['run_id']}")
    print(f"output_dir      : {result['output_dir']}")
    print(f"positive_rule   : {result['positive_rule']}")
    print(f"in_channels     : {result['in_channels']}")
    print(f"target          : {result['target']}")
    print(f"prediction      : {result['prediction']}")
    if result.get("class"):
        print(f"class           : {result['class']}")
    print(f"weights         : {result['weights']}")
    print(f"metrics         : {result['metrics']}")


if __name__ == "__main__":
    main()
