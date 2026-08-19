"""Schema validation for the `ml.*` config block.

Config is loaded as a plain dict everywhere else in this package (dataset.py,
features.py, targets.py, splits.py, models/*, ...) — that does not change.
This module only adds a fail-fast validation pass right after `load_config()`:
unknown keys (typos), missing required fields for a given target/split/model
type, and impossible combinations get caught here, before any raster I/O or
training starts, instead of silently falling back to a default or surfacing
as a confusing error deep inside training.

`validate_ml_config()` is the only function other code should call; it
returns a plain, normalized dict so nothing downstream needs to change.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(extra="forbid")


class PolygonSplitConfig(BaseModel):
    model_config = _STRICT

    path: str | None = None
    id_field: str = "poly_id"
    train_ids: list[Any] | None = None
    test_ids: list[Any] | None = None
    val_ids: list[Any] | None = None
    train: str | list[str] | None = None
    val: str | list[str] | None = None
    test: str | list[str] | None = None


class RasterSplitConfig(BaseModel):
    model_config = _STRICT

    path: str
    train_value: int = 1
    val_value: int = 2
    test_value: int = 3


class SplitConfig(BaseModel):
    model_config = _STRICT

    method: Literal["random", "polygons", "raster", "aoi_rasters"] = "random"
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.0
    negative_to_positive_ratio: float | None = None
    polygons: PolygonSplitConfig = Field(default_factory=PolygonSplitConfig)
    raster: RasterSplitConfig | None = None
    train: str | None = None
    val: str | None = None
    test: str | None = None

    @model_validator(mode="after")
    def _check_method_requirements(self) -> "SplitConfig":
        if self.method == "polygons":
            has_single_file = bool(self.polygons.path)
            has_per_split_paths = bool(self.polygons.train) or bool(self.train)
            if not (has_single_file or has_per_split_paths):
                raise ValueError(
                    "ml.split.method: polygons requires either "
                    "ml.split.polygons.path (+ id_field/train_ids/test_ids) "
                    "or per-split polygon paths."
                )
            if has_single_file and not self.polygons.train_ids:
                raise ValueError(
                    "ml.split.polygons.path is set but train_ids is missing — "
                    "at least one train polygon id is required."
                )
        if self.method == "raster" and self.raster is None:
            raise ValueError("ml.split.method: raster requires ml.split.raster.path.")
        if self.method == "aoi_rasters" and not (self.train and self.test):
            raise ValueError(
                "ml.split.method: aoi_rasters requires ml.split.train and ml.split.test paths."
            )
        return self


_REGRESSION_TARGET_TYPES = {"raster_continuous"}
_RASTER_PATH_TARGET_TYPES = {"raster", "multiclass_raster", "raster_continuous"}
_C_BAND_TARGET_TYPES = {"physics_dod", "c_band_dod", "g_band_dod"}


class TargetConfig(BaseModel):
    model_config = _STRICT

    type: Literal[
        "physics_dod",
        "c_band_dod",
        "g_band_dod",
        "dod_only",
        "dem_diff",
        "ddem",
        "raster",
        "multiclass_raster",
        "raster_continuous",
    ] = "physics_dod"
    path: str | None = None
    positive_value: int = 1
    class_map: dict[int, int] | None = None
    c_min: float | None = None
    g_min: float | None = None
    c_channel: float | None = None
    g_channel: float | None = None
    erosion_threshold_m: float = -0.55
    exclude_channels_from_training: bool = True
    nodata: int = 255
    export: bool = True

    @model_validator(mode="after")
    def _check_type_requirements(self) -> "TargetConfig":
        if self.type in _RASTER_PATH_TARGET_TYPES and not self.path:
            raise ValueError(f"ml.target.type: {self.type} requires ml.target.path.")
        if self.type in _C_BAND_TARGET_TYPES and self.c_min is None and self.g_min is None:
            raise ValueError(
                f"ml.target.type: {self.type} requires ml.target.c_min (or g_min)."
            )
        return self


class FeaturesConfig(BaseModel):
    model_config = _STRICT

    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_nonempty(self) -> "FeaturesConfig":
        if not self.numeric and not self.categorical:
            raise ValueError("Set ml.features.numeric and/or ml.features.categorical.")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # per-model hyperparameters vary freely

    type: str = "random_forest"


class TuningConfig(BaseModel):
    model_config = _STRICT

    search_method: Literal["grid_search", "random_search", "optuna"] = "random_search"
    method: str | None = None  # legacy alias for search_method
    scoring: str = "mcc"
    n_iter: int = 40
    n_trials: int | None = None
    random_state: int = 42
    n_jobs: int = 1
    include_validation_in_tuning: bool = False
    dataset_cache: bool = False
    cv: dict[str, Any] = Field(default_factory=dict)
    # Hyperparameter names/shapes are model-specific and open-ended by design —
    # not schema-checked beyond "it's a mapping".
    search_space: dict[str, Any] = Field(default_factory=dict)


class TobitPathsConfig(BaseModel):
    model_config = _STRICT

    censored_path: str
    logC_bound_path: str


class UnetConfig(BaseModel):
    model_config = _STRICT

    patch_size: int = 128
    overlap: float = 0.5
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    pos_weight: float = 3.0
    early_stopping_patience: int = 10
    val_fraction: float = 0.15
    num_workers: int = 0
    device: str = "auto"
    use_all_pixels: bool = False
    load_norm_stats: bool = False
    warm_start_epochs: int = 5
    fixed_sigma: bool = False
    censored_weight: float = 1.0
    tobit: TobitPathsConfig | None = None


class PredictionConfig(BaseModel):
    model_config = _STRICT

    probability_threshold: float = 0.5
    exclude_channels: bool = True


class MLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_subdir: str = "source_area_workflow/ml/{model}"
    base_rasters: dict[str, str] = Field(default_factory=dict)
    feature_paths: dict[str, str] = Field(default_factory=dict)
    target: TargetConfig = Field(default_factory=TargetConfig)
    features: FeaturesConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tuning: TuningConfig | None = None
    unet: UnetConfig | None = None
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    @model_validator(mode="after")
    def _check_unet_tobit(self) -> "MLConfig":
        model_wants_tobit = bool(self.model.model_extra and self.model.model_extra.get("tobit"))
        if model_wants_tobit and (self.unet is None or self.unet.tobit is None):
            raise ValueError(
                "ml.model.tobit: true requires ml.unet.tobit.censored_path "
                "and ml.unet.tobit.logC_bound_path."
            )
        return self


def validate_ml_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate cfg["ml"], fail fast on typos/missing/contradictory fields.

    Returns cfg unchanged except cfg["ml"] is replaced with its normalized
    (defaults-filled) form. Every other top-level section (paths, site, ...)
    passes through untouched — this only tightens the ml.* block that the
    ML scripts actually consume.
    """
    ml_raw = cfg.get("ml")
    if ml_raw is None:
        raise ValueError("Config is missing the ml: section required for ML workflows.")

    try:
        validated = MLConfig(**ml_raw)
    except Exception as exc:  # pydantic.ValidationError, re-raised with our context
        raise ValueError(f"Invalid ml config:\n{exc}") from exc

    out = dict(cfg)
    out["ml"] = validated.model_dump(exclude_none=True)
    return out
