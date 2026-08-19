"""Hyperparameter search for the UNet.

Selection uses leave-one-polygon-out validation (the same spatially separated
scheme the training workflow uses), so configs are ranked on generalization
rather than on how well they exploit spatial autocorrelation. The test
polygons are never touched.

Usage:
    conda run -n ml_debris python scripts/14_tune_unet.py --config config/ml_unet_tune.yaml

    # long sweeps: run detached and watch the log
    nohup conda run --no-capture-output -n ml_debris python -u \\
        scripts/14_tune_unet.py --config config/ml_unet_tune.yaml > unet_tune.log 2>&1 &
"""
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.ml.unet_tuning import run_unet_tuning  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune UNet hyperparameters.")
    parser.add_argument("--config", default="config/ml_unet_tune.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Override paths.data_dir in the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.setdefault("paths", {})["data_dir"] = args.data_dir

    result = run_unet_tuning(cfg)

    print(f"\nrun_id       : {result['run_id']}")
    print(f"output_dir   : {result['output_dir']}")
    print(f"trials       : {result['n_trials']}")
    print(f"folds        : held out polygon id(s) {result['folds_held_out']}")
    print(f"best_params  : {result['best_params']}")
    print(f"best_val_loss: {result['best_val_loss']}")
    print(f"best_val_r2  : {result['best_val_r2']}")
    print(f"results      : {result['results']}")
    print(f"best_config  : {result['best_config']}")


if __name__ == "__main__":
    main()
