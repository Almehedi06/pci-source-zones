"""Single entry point every ML workflow calls before touching training data.

Order matters: schema errors (typos, missing fields) are cheap and precise,
so they're caught before the data-contract check spends time opening
rasters. Both must pass before any raster is read for real or any model
trains.
"""
from __future__ import annotations

from typing import Any

from .config_schema import validate_ml_config
from .data_contract import ContractReport, validate_data_contract


def run_preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], ContractReport]:
    cfg = validate_ml_config(cfg)
    if not cfg.get("ml", {}).get("target"):
        raise ValueError(
            "Training requires an ml.target section (type, and path or threshold rule). "
            "Configs without a target are inference-only — use scripts/13_predict.py."
        )
    report = validate_data_contract(cfg)
    return cfg, report
