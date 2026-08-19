"""Apply an already-trained model to a new fire — no retraining.

Every training script writes a prediction raster for the fire it trained on.
This script is the other direction: take a model trained at one site and
score a different fire's aligned feature stack. That is what the multi-fire
rollout needs, since most fires have no LiDAR-derived labels to train on.

The model bundle carries its own feature list (and, for UNet, its training
normalization statistics), so the config supplied here only needs to say
where the new fire's rasters live — its target/split sections are ignored.

Usage:
    # point at a training run directory (finds the model artifact inside)
    conda run -n ml_debris python scripts/13_predict.py \\
        --model /home/abdullah/ml_output/source_area_workflow/ml/random_forest_regressor/20260819_043034_dcd85c7 \\
        --config config/ml_random_forest.yaml \\
        --data-dir /path/to/new_fire/aligned \\
        --out /path/to/new_fire/predictions

    # or point at the model file directly
    conda run -n ml_debris python scripts/13_predict.py \\
        --model .../unet_weights.pt --config config/ml_unet_p_obs_union.yaml \\
        --data-dir /path/to/new_fire/aligned --out /path/to/out
"""
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.ml.inference import run_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict with a trained source-zone model on a new fire."
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Training run directory, or a model file (*.joblib / *.pt) directly.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config describing the feature stack to score (target/split sections ignored).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Override paths.data_dir — the new fire's aligned/ folder.",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Directory to write prediction rasters and the run manifest into.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold for the class raster (classification models only). "
        "Defaults to ml.prediction.probability_threshold.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.setdefault("paths", {})["data_dir"] = args.data_dir

    result = run_inference(cfg, args.model, args.out, threshold=args.threshold)

    print(f"model         : {result['model_name']}  ({result['model_path']})")
    print(f"train_run_id  : {result['train_run_id']}")
    print(f"run_id        : {result['run_id']}")
    print(f"output_dir    : {result['output_dir']}")
    print(f"valid pixels  : {result['n_valid_pixels']:,}")
    for key in ("prediction", "probability", "class", "sigma"):
        if result.get(key):
            print(f"{key:<14}: {result[key]}")
    print(f"manifest      : {result['manifest']}")


if __name__ == "__main__":
    main()
