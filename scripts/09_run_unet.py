from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.ml.unet_workflow import run_unet_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a UNet source-zone model.")
    parser.add_argument("--config", default="config/ml_unet.yaml")
    args = parser.parse_args()

    result = run_unet_workflow(load_config(args.config))

    print(f"model_name      : {result['model_name']}")
    print(f"output_dir      : {result['output_dir']}")
    print(f"positive_rule   : {result['positive_rule']}")
    print(f"in_channels     : {result['in_channels']}")
    print(f"train_patches   : {result['n_train_patches']}")
    print(f"val_patches     : {result['n_val_patches']}")
    print(f"target          : {result['target']}")
    print(f"probability     : {result['probability']}")
    print(f"class           : {result['class']}")
    print(f"weights         : {result['weights']}")
    print(f"metrics         : {result['metrics']}")
    print(f"training_history: {result['training_history']}")


if __name__ == "__main__":
    main()
