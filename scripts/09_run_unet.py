from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.ml.unet_workflow import run_unet_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a UNet source-zone model.")
    parser.add_argument("--config", default="config/ml_unet.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Override paths.data_dir in the config. Relative feature/target filenames resolve here.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.setdefault("paths", {})["data_dir"] = args.data_dir

    result = run_unet_workflow(cfg)

    print(f"model_name      : {result['model_name']}")
    print(f"run_id          : {result['run_id']}")
    print(f"output_dir      : {result['output_dir']}")
    print(f"positive_rule   : {result['positive_rule']}")
    print(f"in_channels     : {result['in_channels']}")
    print(f"train_patches   : {result['n_train_patches']}")
    print(f"val_patches     : {result['n_val_patches']}")
    print(f"validation      : {result['validation']}")
    print(f"target          : {result['target']}")
    print(f"prediction      : {result['prediction']}")
    if result.get("class"):
        print(f"class           : {result['class']}")
    print(f"weights         : {result['weights']}")
    print(f"metrics         : {result['metrics']}")
    print(f"training_history: {result['training_history']}")


if __name__ == "__main__":
    main()
