from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .config import resolve_path


class InputProvider:
    """One input source that can return a scalar or raster for each sample."""

    def __init__(
        self,
        spec: dict[str, Any],
        quantity: str,
        cfg: dict[str, Any],
        reference_shape: tuple[int, int] | None = None,
    ) -> None:
        self.spec = spec
        self.quantity = quantity
        self.cfg = cfg
        self.reference_shape = reference_shape
        self.mode = str(spec.get("mode", "constant")).lower()

        self._fixed_value: float | np.ndarray | None = None
        if self.mode == "constant":
            self._fixed_value = convert_units(_constant_value(spec), quantity, spec)
        elif self.mode == "raster":
            path = resolve_path(cfg, spec["path"])
            arr, _profile = read_raster(path)
            if reference_shape is not None and arr.shape != reference_shape:
                raise ValueError(
                    f"{quantity} raster shape {arr.shape} does not match "
                    f"reference shape {reference_shape}: {path}"
                )
            self._fixed_value = convert_units(arr, quantity, spec)
        elif self.mode != "distribution":
            raise ValueError(
                f"Unsupported mode for {quantity}: {self.mode!r}. "
                "Expected constant, distribution, or raster."
            )

    def sample(self, rng: np.random.Generator) -> float | np.ndarray:
        if self.mode in {"constant", "raster"}:
            assert self._fixed_value is not None
            return self._fixed_value
        return convert_units(sample_distribution(self.spec, rng), self.quantity, self.spec)


def read_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def write_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
    nodata: float = -9999.0,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(array, dtype="float32")
    write_data = np.where(np.isfinite(data), data, nodata).astype("float32")
    meta = profile.copy()
    meta.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
    )
    if not meta.get("tiled", False):
        meta.pop("blockxsize", None)
        meta.pop("blockysize", None)
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(write_data, 1)
    return out


def _constant_value(spec: dict[str, Any]) -> float:
    for key in ("value", "value_mm", "value_m", "value_mm_hr", "value_m_s"):
        if key in spec:
            return float(spec[key])
    raise ValueError(f"Constant input is missing a value: {spec}")


def sample_distribution(spec: dict[str, Any], rng: np.random.Generator) -> float:
    dist = str(spec.get("distribution", "uniform")).lower()
    if dist == "uniform":
        low = _first_present(spec, ("min", "min_mm", "min_m", "min_mm_hr", "min_m_s"))
        high = _first_present(spec, ("max", "max_mm", "max_m", "max_mm_hr", "max_m_s"))
        return float(rng.uniform(float(low), float(high)))
    if dist == "lognormal":
        mean = float(_first_present(spec, ("mean", "mean_mm", "mean_m", "mean_mm_hr")))
        std = float(_first_present(spec, ("std", "std_mm", "std_m", "std_mm_hr")))
        if mean <= 0 or std <= 0:
            raise ValueError("Lognormal mean and std must be positive.")
        sigma2 = np.log(1.0 + (std / mean) ** 2)
        mu = np.log(mean) - 0.5 * sigma2
        return float(rng.lognormal(mean=mu, sigma=np.sqrt(sigma2)))
    raise ValueError(f"Unsupported distribution: {dist!r}")


def _first_present(spec: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in spec:
            return spec[key]
    raise ValueError(f"Missing one of {keys} in {spec}")


def convert_units(
    value: float | np.ndarray, quantity: str, spec: dict[str, Any]
) -> float | np.ndarray:
    units = str(spec.get("units", "")).lower().replace(" ", "")

    if quantity == "d50":
        if "value_mm" in spec or "mean_mm" in spec or "min_mm" in spec:
            return np.asarray(value, dtype="float64") / 1000.0
        if units in {"mm", "millimeter", "millimeters"}:
            return np.asarray(value, dtype="float64") / 1000.0
        return np.asarray(value, dtype="float64")

    if quantity == "rainfall_excess":
        if "value_mm_hr" in spec or "mean_mm_hr" in spec or "min_mm_hr" in spec:
            return np.asarray(value, dtype="float64") / 1000.0 / 3600.0
        if units in {"mm/hr", "mmh", "mmperhour", "millimeter/hour"}:
            return np.asarray(value, dtype="float64") / 1000.0 / 3600.0
        return np.asarray(value, dtype="float64")

    return np.asarray(value, dtype="float64")
