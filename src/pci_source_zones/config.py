from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "parameters": {
        "alpha": 1.167,
        "tau_star": 0.043,
        "rho_w": 1000.0,
        "rho_s": 2650.0,
        "g": 9.81,
        "k": 0.0474,
        "p": 1.0 / 6.0,
        "m1": 0.6,
        "n1": 0.7,
    },
    "monte_carlo": {
        "n_samples": 1500,
        "seed": 42,
    },
    "c_sampling": {
        "d50": {
            "mode": "distribution",
            "distribution": "lognormal",
            "mean": 2.0,
            "std": 0.48,
            "units": "mm",
        },
        "na": {
            "mode": "distribution",
            "distribution": "uniform",
            "min": 0.015,
            "max": 0.10,
            "units": "manning",
        },
        "rainfall_excess": {
            "mode": "distribution",
            "distribution": "uniform",
            "min": 15.0,
            "max": 55.0,
            "units": "mm/hr",
        },
    },
    "source_area": {
        "output_subdir": "source_area_workflow",
        "erosion_threshold_m": -0.277,
        "use_dod_filter": True,
        "top_percent_values": [10, 5],
        "c_min_values": [5, 10, 20, 30, 50, 70],
        "c_channel_max": 120,
        "min_polygon_area_m2": 0.0,
    },
    "terrain": {
        "fill_sinks": True,
        "flow_director": "D8",
    },
    "outputs": {
        "pci_threshold": 0.7,
        "min_area_m2": 0.0,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    cfg = deep_merge(DEFAULT_CONFIG, user_config)
    cfg["_config_path"] = str(config_path)
    cfg["_repo_root"] = str(_infer_repo_root(config_path))
    return cfg


def _infer_repo_root(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def repo_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_repo_root"]).expanduser().resolve()


def resolve_path(cfg: dict[str, Any], path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return repo_root(cfg) / p


def output_path(cfg: dict[str, Any], key: str, default_name: str) -> Path:
    outputs = cfg.get("outputs", {})
    if key in outputs:
        return resolve_path(cfg, outputs[key])

    out_dir = resolve_path(cfg, cfg.get("paths", {}).get("output_dir", "data/outputs"))
    if key in {"slope", "specific_catchment_area", "topographic_driving_index"}:
        out_dir = resolve_path(
            cfg, cfg.get("paths", {}).get("processed_dir", "data/processed")
        )
    return out_dir / default_name


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
