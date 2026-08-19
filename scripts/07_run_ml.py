from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from pci_source_zones.config import load_config  # noqa: E402
from pci_source_zones.ml.workflow import run_ml_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a source-zone ML workflow.")
    parser.add_argument("--config", default="config/ml_random_forest.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="random_forest, logistic_regression, or xgboost. Defaults to ml.model.type.",
    )
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

    result = run_ml_workflow(cfg, model_name=args.model)
    print(f"model_name    : {result['model_name']}")
    print(f"run_id        : {result['run_id']}")
    print(f"output_dir    : {result['output_dir']}")
    print(f"target        : {result['target']}")
    print(f"model         : {result['model']}")
    print(f"metrics       : {result['metrics']}")
    print(f"feature_scores: {result['feature_scores']}")
    print(f"split_summary : {result['split_summary']}")
    if result.get("prediction"):
        print(f"prediction    : {result['prediction']}")   # regression
    if result.get("probability"):
        print(f"probability   : {result['probability']}")  # classification
    if result.get("class"):
        print(f"class         : {result['class']}")        # classification
    print(f"features      : {result['feature_names']}")


if __name__ == "__main__":
    main()
