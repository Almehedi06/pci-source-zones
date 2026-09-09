"""Canonical site/event configuration schema.

One schema for every fire, every epoch pair, single-site or batch. A new site is
a config file, never a code change.

Replaces four incompatible formats: the single-site evidence config, the batch
evidence config, the old physics config, and the ad-hoc per-experiment configs.

Layout:

    site        identity and epoch pair
    grid        resolutions and the block factor between them
    inputs      explicit paths (single) or roots + templates (batch)
    terrain     flow routing and alpha, for G and DeltaG from the DEM pair
    stable      how the stable-ground sample is defined
    align       optional predictor-raster alignment, run before the rest
    outputs     where products go
    lod         detection-threshold settings
    evidence    C2/C3 definitions and the analysis domain
    ceff        Gamma fit and inversion settings

Validation is deliberately strict: an unknown key is an error, because a silently
ignored typo in a threshold is exactly the class of failure this pipeline has
already suffered from.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import yaml

C3Mode = Literal["raw_threshold", "detectable_dod", "sigma_lod"]
InputMode = Literal["single", "batch"]


class ConfigError(ValueError):
    """Configuration is invalid."""


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"{ctx}: missing required key '{key}'")
    return d[key]


def _reject_unknown(d: dict, allowed: set[str], ctx: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise ConfigError(
            f"{ctx}: unknown key(s) {sorted(extra)}. "
            f"Allowed: {sorted(allowed)}"
        )


# ─────────────────────────────────────────────────────────── sections

@dataclass
class GridSpec:
    """Fine and coarse resolutions, and the aggregation factor between them."""
    fine_res_m: float = 1.0
    coarse_res_m: float = 10.0
    crs: str | None = None

    @property
    def block_size(self) -> int:
        """Fine pixels per coarse cell edge."""
        ratio = self.coarse_res_m / self.fine_res_m
        if abs(ratio - round(ratio)) > 1e-9:
            raise ConfigError(
                f"coarse_res_m ({self.coarse_res_m}) must be an integer multiple "
                f"of fine_res_m ({self.fine_res_m}); got ratio {ratio}")
        return int(round(ratio))

    @classmethod
    def parse(cls, d: dict) -> "GridSpec":
        _reject_unknown(d, {"fine_res_m", "coarse_res_m", "crs"}, "grid")
        out = cls(**d)
        out.block_size            # validate the ratio eagerly
        return out


@dataclass
class InputSpec:
    """Where the input rasters are.

    `single` mode takes explicit paths. `batch` mode takes roots plus filename
    templates containing `{id}`, and discovers analysis areas from the filesystem.
    """
    mode: InputMode = "single"
    # single
    dem_pre: str | None = None
    dem_post: str | None = None
    g_pre: str | None = None
    s_pre: str | None = None
    delta_g: str | None = None
    dz: str | None = None
    stable_cells: str | None = None
    dem: str | None = None
    aoi: str | None = None
    g_pre_coarse: str | None = None
    dz_nodata: float | None = None
    # batch
    roots: dict[str, str] = field(default_factory=dict)
    templates: dict[str, str] = field(default_factory=dict)
    id_glob: str | None = None

    @classmethod
    def parse(cls, d: dict) -> "InputSpec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "inputs")
        out = cls(**d)
        if out.mode not in ("single", "batch"):
            raise ConfigError(f"inputs.mode must be 'single' or 'batch', got {out.mode!r}")
        if out.mode == "single":
            if not any(getattr(out, key) for key in (
                    "dem_pre", "dem_post", "g_pre", "s_pre", "delta_g", "dz",
                    "stable_cells", "dem", "aoi", "g_pre_coarse")):
                raise ConfigError("inputs: single mode requires at least one input path")
        else:
            if not out.roots:
                raise ConfigError("inputs: 'roots' is required in batch mode")
            for k, v in out.templates.items():
                if "{id}" not in v:
                    raise ConfigError(f"inputs.templates.{k}: must contain '{{id}}'")
        return out


@dataclass
class TerrainSpec:
    """How G = a * S^alpha is computed from the DEM pair.

    Routing is always WhiteboxTools D-infinity. alpha is 7/6 from the
    shear-stress derivation and is tied to `m1` in c_model.py.
    """
    alpha: float = 1.167
    fill_sinks: bool = True
    curvature: bool = False
    work_dir: str | None = None

    @classmethod
    def parse(cls, d: dict) -> "TerrainSpec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "terrain")
        out = cls(**d)
        if out.alpha <= 0:
            raise ConfigError(f"terrain.alpha must be > 0, got {out.alpha}")
        return out

    def params(self):
        from .physics.terrain import TerrainParams
        return TerrainParams(alpha=self.alpha, fill_sinks=self.fill_sinks,
                             curvature=self.curvature, work_dir=self.work_dir)


@dataclass
class StableLayer:
    """One criterion contributing to the stable-ground mask."""
    name: str
    path: str                            # raster, or a vector to rasterise (1 inside)
    threshold: float | None = None       # |value| < threshold
    min: float | None = None             # value >= min
    max: float | None = None             # value < max
    invert: bool = False                 # stable OUTSIDE the mask (e.g. burn perimeter)
    required: bool = True

    @classmethod
    def parse(cls, d: dict) -> "StableLayer":
        _reject_unknown(d, set(cls.__dataclass_fields__), "stable.layers[]")
        for key in ("name", "path"):
            if key not in d:
                raise ConfigError(f"stable.layers[]: missing required key '{key}'")
        out = cls(**d)
        is_vector = Path(out.path).suffix.lower() in {
            ".shp", ".gpkg", ".geojson", ".json", ".gdb", ".fgb"}
        if out.threshold is None and out.min is None and out.max is None:
            if not is_vector:
                raise ConfigError(
                    f"stable.layers['{out.name}']: needs one of threshold, min, max")
            # A vector needs no criterion: it rasterises to 1 inside, 0 outside,
            # and `invert: true` is what turns it into "stable outside".
        if out.threshold is not None and (out.min is not None or out.max is not None):
            raise ConfigError(
                f"stable.layers['{out.name}']: threshold and min/max are exclusive")
        return out


@dataclass
class StableSpec:
    """Definition of the stable-ground sample the detection threshold uses.

    `method: criteria` builds the mask from predictors independent of the change
    signal. `method: threshold` reproduces the historical rule, which is circular
    when its layers are the same change signals whose LoD is being estimated —
    allowed, but only as an explicit choice.
    """
    method: Literal["criteria", "threshold"] = "criteria"
    layers: list[StableLayer] = field(default_factory=list)
    combine: Literal["all", "any"] = "all"
    min_frac: float = 0.02

    @classmethod
    def parse(cls, d: dict) -> "StableSpec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "stable")
        layers = [StableLayer.parse(x) for x in d.get("layers", [])]
        out = cls(method=d.get("method", "criteria"), layers=layers,
                  combine=d.get("combine", "all"),
                  min_frac=float(d.get("min_frac", 0.02)))
        if out.method not in ("criteria", "threshold"):
            raise ConfigError(
                f"stable.method must be 'criteria' or 'threshold', got {out.method!r}")
        if out.combine not in ("all", "any"):
            raise ConfigError(f"stable.combine must be 'all' or 'any', got {out.combine!r}")
        if not out.layers:
            raise ConfigError("stable: at least one layer is required")
        names = [x.name for x in out.layers]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ConfigError(f"stable.layers: duplicate name(s) {sorted(dupes)}")
        return out


@dataclass
class LodSpec:
    """Detection threshold for the change signal."""
    method: Literal["nmad"] = "nmad"
    scale: float = 2.0
    exclude_channels: bool = False
    g_channel: float | None = None

    @classmethod
    def parse(cls, d: dict) -> "LodSpec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "lod")
        out = cls(**d)
        if out.method != "nmad":
            raise ConfigError(f"lod.method: only 'nmad' is supported, got {out.method!r}")
        if out.scale <= 0:
            raise ConfigError(f"lod.scale must be > 0, got {out.scale}")
        return out


@dataclass
class C3Spec:
    """How far the surface must drop to count as erosion.

        C3 = 1[ dz < -lod ]

    `lod` takes one of three forms, covering every kind of dDEM input:

        0          the dDEM is already LoD-filtered, so any lowering counts
        0.175      a constant level of detection, in metres
        "auto"     derive it from this site's own data: scale x NMAD of dz on
                   stable ground, exactly the rule already used for DeltaG
        "/s.tif"   a per-cell sigma raster, multiplied by scale

    "auto" is the default because it is the only choice that carries no
    site-specific number, and it makes C2 and C3 share one detection rule.
    """
    lod: float | str = "auto"
    scale: float = 2.0

    @classmethod
    def parse(cls, d: dict) -> "C3Spec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "evidence.c3")
        out = cls(**d)
        if out.scale <= 0:
            raise ConfigError(f"evidence.c3.scale must be > 0, got {out.scale}")
        if isinstance(out.lod, bool) or out.lod is None:
            raise ConfigError(f"evidence.c3.lod: invalid value {out.lod!r}")
        if isinstance(out.lod, (int, float)):
            if out.lod < 0:
                raise ConfigError(f"evidence.c3.lod must be >= 0, got {out.lod}")
        elif isinstance(out.lod, str):
            if out.lod != "auto" and not out.lod.endswith((".tif", ".tiff", ".vrt")):
                raise ConfigError(
                    f"evidence.c3.lod must be a number, 'auto', or a raster path; "
                    f"got {out.lod!r}")
        else:
            raise ConfigError(
                f"evidence.c3.lod must be a number, 'auto', or a raster path; "
                f"got {type(out.lod).__name__}")
        return out

    @property
    def is_auto(self) -> bool:
        return self.lod == "auto"

    @property
    def raster_path(self) -> str | None:
        """The sigma raster, when `lod` names one."""
        return self.lod if isinstance(self.lod, str) and not self.is_auto else None


@dataclass
class DomainSpec:
    """Which cells are analysed. Never which cells count as eroded."""
    g_min: float = 5.0
    g_channel: float = 300.0
    elev_min: float | None = None
    slope_max_deg: float | None = None

    @classmethod
    def parse(cls, d: dict) -> "DomainSpec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "evidence.domain")
        out = cls(**d)
        if out.g_min >= out.g_channel:
            raise ConfigError(
                f"evidence.domain: g_min ({out.g_min}) must be < g_channel ({out.g_channel})")
        if out.slope_max_deg is not None and not 0 < out.slope_max_deg < 90:
            raise ConfigError(
                f"evidence.domain.slope_max_deg must be in (0, 90), got {out.slope_max_deg}")
        return out


@dataclass
class EvidenceSpec:
    c3: C3Spec = field(default_factory=C3Spec)
    domain: DomainSpec = field(default_factory=DomainSpec)
    apply_domain: bool = True

    @classmethod
    def parse(cls, d: dict) -> "EvidenceSpec":
        _reject_unknown(d, {"c3", "domain", "apply_domain"}, "evidence")
        return cls(
            c3=C3Spec.parse(d.get("c3", {})),
            domain=DomainSpec.parse(d.get("domain", {})),
            apply_domain=bool(d.get("apply_domain", True)),
        )


@dataclass
class CeffSpec:
    """Gamma fit and threshold inversion."""
    p_obs_methods: list[str] = field(default_factory=lambda: ["intersection", "union"])
    elev_min: float | None = 600.0
    elev_filter_outputs: bool = True
    p_ceiling: float = 0.05
    g_hillslope_max: float = 300.0
    min_domain_cells: int = 100

    @classmethod
    def parse(cls, d: dict) -> "CeffSpec":
        _reject_unknown(d, set(cls.__dataclass_fields__), "ceff")
        out = cls(**d)
        bad = set(out.p_obs_methods) - {"intersection", "union"}
        if bad:
            raise ConfigError(f"ceff.p_obs_methods: unknown {sorted(bad)}")
        if not 0 < out.p_ceiling < 1:
            raise ConfigError(f"ceff.p_ceiling must be in (0, 1), got {out.p_ceiling}")
        if out.min_domain_cells < 2:
            raise ConfigError(
                f"ceff.min_domain_cells must be >= 2, got {out.min_domain_cells}")
        return out


# ─────────────────────────────────────────────────────────── top level

@dataclass
class SiteConfig:
    """A complete site/epoch-pair configuration."""
    site: str
    epochs: dict[str, Any]
    grid: GridSpec
    inputs: InputSpec
    outputs: dict[str, str]
    lod: LodSpec
    terrain: TerrainSpec
    stable: StableSpec | None
    evidence: EvidenceSpec
    ceff: CeffSpec
    align: str | dict[str, Any] | None = None
    source_path: Path | None = None

    # ---- accessors used by the stages

    @property
    def is_batch(self) -> bool:
        return self.inputs.mode == "batch"

    @property
    def block_size(self) -> int:
        return self.grid.block_size

    def output_root(self) -> Path:
        return Path(self.outputs["root"]).expanduser()

    def output_dir(self, area_id: str | None = None) -> Path:
        """Output directory, per analysis area in batch mode."""
        root = self.output_root()
        if area_id is None:
            return root
        layout = self.outputs.get("layout", "{id}")
        return root / layout.format(id=area_id)

    def input_path(self, key: str, area_id: str | None = None) -> Path | None:
        """Resolve an input path for single or batch mode."""
        if not self.is_batch:
            v = getattr(self.inputs, key, None)
            return Path(v).expanduser() if v else None
        if area_id is None:
            raise ConfigError(f"batch mode requires an area id to resolve '{key}'")
        root = self.inputs.roots.get(key)
        tmpl = self.inputs.templates.get(key)
        if not root or not tmpl:
            return None
        return Path(root).expanduser() / tmpl.format(id=area_id)

    def align_config(self):
        """Build the `AlignConfig` for this site, or None when not configured."""
        if self.align is None:
            return None
        from .align import AlignConfig
        if isinstance(self.align, str):
            return AlignConfig.load(_resolve_rel(self.align, self.source_path))
        return AlignConfig.parse(self.align)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        d["grid"]["block_size"] = self.block_size
        return d

    # ---- construction

    @classmethod
    def parse(cls, d: dict, source_path: Path | None = None) -> "SiteConfig":
        _reject_unknown(
            d, {"site", "epochs", "grid", "inputs", "outputs", "lod", "evidence",
                "ceff", "align", "terrain", "stable"},
            "config root")
        outputs = _require(d, "outputs", "config root")
        if "root" not in outputs:
            raise ConfigError("outputs: missing required key 'root'")
        _reject_ambiguous_sources(d)
        return cls(
            site=str(_require(d, "site", "config root")),
            epochs=d.get("epochs", {}),
            grid=GridSpec.parse(d.get("grid", {})),
            inputs=InputSpec.parse(_require(d, "inputs", "config root")),
            outputs=outputs,
            lod=LodSpec.parse(d.get("lod", {})),
            terrain=TerrainSpec.parse(d.get("terrain", {})),
            stable=StableSpec.parse(d["stable"]) if d.get("stable") else None,
            evidence=EvidenceSpec.parse(d.get("evidence", {})),
            ceff=CeffSpec.parse(d.get("ceff", {})),
            align=validate_align(d.get("align"), source_path),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path, overrides: dict | None = None) -> "SiteConfig":
        """Load from YAML, with an optional deep-merged override file."""
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        if overrides:
            data = deep_merge(data, overrides)
        return cls.parse(data, source_path=path)


def _reject_ambiguous_sources(d: dict) -> None:
    """A quantity may be computed here or supplied — never both.

    When both are configured the stage output silently wins and the supplied
    file is ignored without a word, which is exactly the class of quiet override
    this pipeline treats as a defect. Both routes stay available; the config
    just has to say which one it means.
    """
    inputs = d.get("inputs") or {}
    if d.get("stable") and inputs.get("stable_cells"):
        raise ConfigError(
            "stable ground is configured twice: a 'stable:' section computes the "
            "mask and inputs.stable_cells supplies one. Keep exactly one — the "
            "computed mask would silently win.")
    has_dems = bool(inputs.get("dem_pre") and inputs.get("dem_post"))
    has_g = bool(inputs.get("g_pre") and inputs.get("delta_g"))
    if has_dems and has_g:
        raise ConfigError(
            "terrain forcing is configured twice: dem_pre/dem_post let the "
            "terrain stage derive G, and g_pre/delta_g supply it. Keep exactly "
            "one — the derived rasters would silently win.")


def validate_align(spec: Any, source_path: Path | None) -> str | dict | None:
    """Fail fast on a malformed align section, but keep the config JSON-safe.

    Stored as the raw string or mapping the user wrote; `SiteConfig.align_config`
    turns it into an `AlignConfig` when the stage actually runs. Validating here
    means `pci-zones validate` catches a typo without touching any raster.
    """
    if spec is None:
        return None
    from .align import AlignConfig, AlignError
    try:
        if isinstance(spec, (str, Path)):
            AlignConfig.load(_resolve_rel(spec, source_path))
        elif isinstance(spec, dict):
            AlignConfig.parse(spec)
        else:
            raise ConfigError(
                f"align: expected a path or a mapping, got {type(spec).__name__}")
    except AlignError as exc:
        raise ConfigError(f"align: {exc}") from exc
    except FileNotFoundError as exc:
        raise ConfigError(f"align: file not found: {exc}") from exc
    return str(spec) if isinstance(spec, Path) else spec


def _resolve_rel(spec: str | Path, source_path: Path | None) -> Path:
    """Resolve an align config path relative to the site config that names it."""
    q = Path(spec).expanduser()
    if not q.is_absolute() and source_path is not None:
        cand = Path(source_path).parent / q
        if cand.exists():
            return cand
    return q


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins on conflicts."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
