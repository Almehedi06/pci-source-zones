"""Run identity and provenance for trained ML artifacts.

Every model.joblib / unet_weights.pt used to carry no record of which
config, which code, or which data produced it — reruns silently overwrote
the last run's outputs. `new_run_id()` gives every run its own directory
(via outputs.ml_output_dir(..., run_id=...)); `write_run_manifest()` drops a
run_manifest.json into it recording exactly what produced it.
"""
from __future__ import annotations

import getpass
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_contract import ContractReport


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_git_short_hash()}"


def write_run_manifest(
    cfg: dict[str, Any],
    out_dir: Path,
    run_id: str,
    data_report: ContractReport | None = None,
) -> Path:
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user": _best_effort(getpass.getuser),
        "git": _git_info(),
        "python": platform.python_version(),
        "package_versions": _package_versions(),
        "config": cfg,
        "data_contract": _report_to_dict(data_report) if data_report is not None else None,
    }
    path = Path(out_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def _report_to_dict(report: ContractReport) -> dict[str, Any]:
    return {
        "reference": report.reference,
        "rasters": [
            {"name": r.name, "path": r.path, "size_bytes": r.size_bytes, "mtime": r.mtime}
            for r in report.rasters
        ],
    }


def _git_short_hash() -> str:
    out = _run_git(["rev-parse", "--short", "HEAD"])
    return out if out else "nogit"


def _git_info() -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    status = _run_git(["status", "--porcelain"])
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
    }


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "pandas", "sklearn", "xgboost", "torch", "pydantic", "rasterio", "geopandas"):
        versions[module_name] = _best_effort(lambda m=module_name: __import__(m).__version__) or "not installed"
    return versions


def _best_effort(fn):
    try:
        return fn()
    except Exception:
        return None
