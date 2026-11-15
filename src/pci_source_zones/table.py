"""Write one site's ML table: one row per cell, every feature, label and split.

    pci-zones table --config config/tables/<site>.yaml

The table is the only thing the ML repository reads. Everything that decides
what a number *means* happens here, once, so every model sees the same data:
which file a feature comes from and in what units, how points or polygons
become labels, what an unmapped cell counts as, and which cells form an
official split. The ML side only chooses among these.

Config shape (input paths are relative to `data_dir`, and `data_dir` and `output` to the
config file, unless absolute; `where:` values are compared as text, so quote them):

    site: montecito
    data_dir: /path/to/aligned
    extent: elevation_aligned.tif         # rows = this raster's valid cells; its grid is the table's grid
    output: /path/to/tables/montecito.parquet
    features:
      slope:          {path: slope_aligned.tif, units: m/m, definition: "...", source: "..."}
      soil_thickness: {path: soil_thickness_aligned.tif, units: m, scale: 0.01, valid_range: [0, 5],
                       definition: "...", source: "SOLUS100 anylithicdpt (cm)"}
    labels:
      p_obs_union:  {raster: p_obs_union_aligned.tif, kind: continuous, valid_range: [0, 1],
                     definition: "share of 1 m cells with C2 or C3 evidence per 10 m cell"}
      erosion:      {raster: p_obs_union_aligned.tif, kind: binary, threshold: 0.10, definition: "..."}
      process:      {polygons: inventory.gpkg, layer: inventory, where: {epoch: "2018_2021"},
                     kind: class, class_field: Primary, classes: {"03_...": 1, ...}, definition: "...",
                     outside: background, domain: {polygons: inventory.gpkg, layer: mapping_area}}
      initiation:   {points: points.gpkg, kind: binary, buffer_cells: 1, outside: background,
                     domain: {polygons: footprint.gpkg}, definition: "..."}

Every label needs a `definition`, as every feature does: the ML side compares it between
sites, so two labels that share a name but not a method cannot be pooled by accident.

A point or polygon label with `outside: background` can keep its negatives away from the
positives, where a mapping offset or an unmapped neighbour makes a 0 doubtful:

      negatives: {min_distance_m: 50}       # background closer than this to a positive is unknown
                                            # (measured from the features' cells on this grid; a
                                            #  feature wholly outside the grid protects nothing)

There is deliberately no rule on terrain or on change. A slope floor drops gentle negatives
but keeps gentle positives, so a model learns that gentle ground is positive; a dDEM rule
drops deposition (a certain negative for initiation) and, where the dDEM does not resolve the
scars (Pioneer, 10 m), drops negatives at random.

A split can also be drawn as whole square blocks, so test cells do not sit beside
training cells; with `stratify` (a binary or class label), test (and val) get their fraction
of that label's positives and of its negatives, each within 5 percentage points or the table
is not written, and a mapped object of that label (touching positive cells) cut by a block
edge follows the block holding at least half of it, so such a scar is in one split only.
Objects of other labels, and objects with no such block, are not moved; the metadata counts
them:

      blocks: {blocks: {size_m: 500, test_fraction: 0.3, val_fraction: 0.0, seed: 42,
                        stratify: initiation}}

Distances and block sizes are in metres: both need a grid in metres with square cells.
    splits:
      official: {polygons: split_polygons.gpkg, id_field: id,
                 assign: {train: [1, 2], test: [3], holdout: [4]}}
    terrain:                              # optional: terrain features from this repo's one method
      dem: elevation_aligned.tif          # on the table grid
      products: [slope, drainage_area, G, curvature, aspect]

`terrain` adds `terrain_<product>` features computed by `physics/terrain.py` (WhiteboxTools,
filled DEM, D-infinity, G = a S^(7/6)) on the given DEM, with the same units and definition at
every site of the same cell size. Terrain built elsewhere (Landlab, a 2 m DEM averaged) is a
different definition and the ML side will say so. Products are cached beside the table.

Columns: `cell_id` (row * width + col), `row`, `col`, `x`, `y` (cell centre),
the features under their own names, then `label_<name>`, `split_<name>` and
`group_<name>` (the polygon id of a polygon split). Grid, units, definitions and
label rules travel inside the Parquet file as metadata under `source_zones_table`.

Mistakes this module refuses, because each one produces a plausible table:

* **A feature in other units.** Units and a definition are required for every
  feature, so the ML side can refuse to mix centimetres with metres.
* **"Not mapped" read as "did not happen".** `outside` must be stated for
  polygon and point labels: `background` makes unmapped cells negatives,
  `unknown` leaves them out. `domain` limits where background can be claimed.
  A cell a polygon only partly covers, or one under a class dropped with
  `null`, is left unknown, never background.
* **Silent class mapping.** Every class value present must be mapped, or
  explicitly dropped with `null`.
* **Overlapping labels.** A cell claimed by two classes is an error, unless
  `on_overlap: unknown` says such cells are left unlabelled.
* **A label with no positives**, which is almost always a wrong path, layer,
  filter or CRS rather than a site where nothing happened.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.windows
import yaml

from .raster import GridMismatch, grids_match

CONTRACT = "source-zones-table/1"
META_KEY = b"source_zones_table"
SPLIT_NAMES = ("train", "val", "test", "holdout")
LABEL_KINDS = ("continuous", "binary", "class")
OUTSIDE = ("background", "unknown")
OVERLAP = ("error", "unknown")
RESERVED_PREFIXES = ("label_", "split_", "group_")
POSITION_COLUMNS = ("cell_id", "row", "col", "x", "y")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
TERRAIN_PRODUCTS = {
    "slope": ("m/m", "slope of the filled DEM (WhiteboxTools)"),
    "drainage_area": ("m2", "D-infinity specific catchment area of the filled DEM x cell width"),
    "G": ("m", "G = a S^(7/6), a the D-infinity specific catchment area, S the slope"),
    "curvature": ("1/m", "plan curvature of the DEM (WhiteboxTools)"),
    "aspect": ("degrees", "aspect clockwise from north (WhiteboxTools)"),
}


class TableError(ValueError):
    """Table configuration or content problem."""


# ─────────────────────────────────────────────────────────────── config

def _check_keys(where: str, d: dict[str, Any], allowed: set[str], required: set[str] = frozenset()) -> None:
    if not isinstance(d, dict):
        raise TableError(f"{where}: expected a mapping, got {type(d).__name__}")
    extra = set(d) - allowed
    if extra:
        raise TableError(f"{where}: unknown key(s) {sorted(extra)}; allowed {sorted(allowed)}")
    missing = set(required) - set(d)
    if missing:
        raise TableError(f"{where}: missing required key(s) {sorted(missing)}")


def _check_name(kind: str, name: str) -> str:
    name = str(name)
    if not _NAME.match(name):
        raise TableError(f"{kind} name {name!r}: use letters, digits and underscores, starting with a letter")
    if kind == "feature" and (name.startswith(RESERVED_PREFIXES) or name in POSITION_COLUMNS):
        raise TableError(f"feature name {name!r} collides with a reserved column or prefix {RESERVED_PREFIXES}")
    return name


def _text(where: str, value: Any) -> str:
    """A required, non-blank string. YAML `key:` with nothing after it is None, not text."""
    if value is None or not isinstance(value, (str, int, float)) or not str(value).strip():
        raise TableError(f"{where} is empty; state it, it is what the ML side compares between sites")
    return str(value).strip()


def _range(where: str, value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise TableError(f"{where}: valid_range must be [low, high]")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise TableError(f"{where}: valid_range low {lo} is above high {hi}")
    return lo, hi


@dataclass
class FeatureSpec:
    name: str
    path: Path
    units: str
    definition: str
    source: str = ""
    scale: float | None = None
    valid_range: tuple[float, float] | None = None

    @classmethod
    def parse(cls, name: str, d: dict[str, Any], base: Path) -> "FeatureSpec":
        where = f"features.{name}"
        _check_keys(where, d, {"path", "units", "definition", "source", "scale", "valid_range"},
                    {"path", "units", "definition"})
        units = _text(f"{where}.units", d["units"])
        definition = _text(f"{where}.definition", d["definition"])
        scale = None if d.get("scale") is None else float(d["scale"])
        if scale is not None and (not np.isfinite(scale) or scale == 0):
            raise TableError(f"{where}.scale must be a finite non-zero number")
        return cls(name=_check_name("feature", name), path=_resolve(base, d["path"]),
                   units=units, definition=definition,
                   source=str(d.get("source") or "").strip(), scale=scale,
                   valid_range=_range(where, d.get("valid_range")))


@dataclass
class VectorSource:
    """Polygons or points, with an optional layer and attribute filter."""
    path: Path
    layer: str | None = None
    where: dict[str, Any] = field(default_factory=dict)

    def read(self, crs: Any):
        import geopandas as gpd

        if not self.path.exists():
            raise FileNotFoundError(f"vector source: {self.path}")
        if self.layer is None:
            try:
                import pyogrio
                layers = [str(l[0]) for l in pyogrio.list_layers(self.path)]
            except Exception:                     # not a multi-layer format, or pyogrio absent
                layers = []
            if len(layers) > 1:
                raise TableError(f"{self.path.name} holds layers {layers}; name one with layer:")
        gdf = gpd.read_file(self.path, layer=self.layer) if self.layer else gpd.read_file(self.path)
        for col, value in self.where.items():
            if col not in gdf.columns:
                raise TableError(f"{self.path.name}: filter field {col!r} not in {list(gdf.columns)}")
            allowed = value if isinstance(value, list) else [value]
            gdf = gdf[gdf[col].astype(str).isin([str(v) for v in allowed])]
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            raise TableError(f"{self.path.name}{'/' + self.layer if self.layer else ''}: "
                             f"no features left after the filter {self.where or '(none)'}")
        if gdf.crs is None:
            raise TableError(f"{self.path.name}: no CRS declared; cannot place it on the grid")
        return gdf.to_crs(crs)


@dataclass
class NegativesSpec:
    """Which background cells of a point or polygon label count as negatives."""
    min_distance_m: float

    @classmethod
    def parse(cls, where: str, d: dict[str, Any]) -> "NegativesSpec":
        _check_keys(where, d, {"min_distance_m"}, {"min_distance_m"})
        spec = cls(min_distance_m=float(d["min_distance_m"]))
        if spec.min_distance_m <= 0:
            raise TableError(f"{where}.min_distance_m must be > 0")
        return spec


@dataclass
class LabelSpec:
    name: str
    kind: str
    raster: Path | None = None
    polygons: VectorSource | None = None
    points: VectorSource | None = None
    threshold: float | None = None
    valid_range: tuple[float, float] | None = None
    class_field: str | None = None
    classes: dict[str, int | None] | None = None
    outside: str | None = None
    domain: dict[str, Any] | None = None
    on_overlap: str = "error"
    buffer_cells: int = 0
    all_touched: bool = False
    definition: str = ""
    description: str = ""
    negatives: NegativesSpec | None = None

    @classmethod
    def parse(cls, name: str, d: dict[str, Any], base: Path) -> "LabelSpec":
        where = f"labels.{name}"
        _check_keys(where, d, {"raster", "polygons", "points", "layer", "where", "kind", "threshold",
                               "valid_range", "class_field", "classes", "outside", "domain", "on_overlap",
                               "buffer_cells", "all_touched", "definition", "description", "negatives"},
                    {"kind", "definition"})
        definition = _text(f"{where}.definition", d.get("definition"))
        sources = [k for k in ("raster", "polygons", "points") if k in d]
        if len(sources) != 1:
            raise TableError(f"{where}: give exactly one of raster, polygons or points (got {sources or 'none'})")
        kind = str(d["kind"]).lower()
        if kind not in LABEL_KINDS:
            raise TableError(f"{where}.kind must be one of {list(LABEL_KINDS)}, got {kind!r}")
        spec = cls(name=_check_name("label", name), kind=kind,
                   threshold=None if d.get("threshold") is None else float(d["threshold"]),
                   valid_range=_range(where, d.get("valid_range")),
                   class_field=d.get("class_field"), outside=d.get("outside"), domain=d.get("domain"),
                   on_overlap=str(d.get("on_overlap", "error")).lower(),
                   buffer_cells=int(d.get("buffer_cells", 0)), all_touched=bool(d.get("all_touched", False)),
                   definition=definition, description=str(d.get("description") or ""))
        if d.get("classes") is not None:
            if not isinstance(d["classes"], dict) or not d["classes"]:
                raise TableError(f"{where}.classes must be a non-empty mapping of value -> integer code (or null)")
            spec.classes = {str(k): (None if v is None else int(v)) for k, v in d["classes"].items()}
        vector_keys = {"layer", "where", "class_field", "classes", "outside", "domain", "on_overlap", "all_touched"}
        if spec.classes is not None and kind != "class":
            raise TableError(f"{where}: classes apply to kind class only; this label is {kind}")
        if "raster" in d:
            spec.raster = _resolve(base, d["raster"])
            misplaced = vector_keys & set(d) - {"classes"}
            if misplaced or "buffer_cells" in d:
                raise TableError(f"{where}: {sorted(misplaced | ({'buffer_cells'} & set(d)))} apply to "
                                 "polygon or point labels, not a raster label")
            if spec.threshold is not None and kind != "binary":
                raise TableError(f"{where}: threshold makes a binary label; set kind: binary")
        else:
            src = VectorSource(path=_resolve(base, d.get("polygons", d.get("points"))),
                               layer=d.get("layer"), where=dict(d.get("where") or {}))
            if "polygons" in d:
                spec.polygons = src
                if "buffer_cells" in d:
                    raise TableError(f"{where}: buffer_cells applies to points, not polygons")
            else:
                spec.points = src
                if spec.buffer_cells < 0:
                    raise TableError(f"{where}.buffer_cells must be >= 0")
            if kind == "continuous":
                raise TableError(f"{where}: polygon and point labels are binary or class, not continuous")
            if spec.threshold is not None or spec.valid_range is not None:
                raise TableError(f"{where}: threshold and valid_range apply to raster labels only")
            if spec.outside not in OUTSIDE:
                raise TableError(f"{where}.outside must be stated: {list(OUTSIDE)}. 'background' makes every "
                                 "unmapped cell a negative; 'unknown' leaves it unlabelled")
            if spec.on_overlap not in OVERLAP:
                raise TableError(f"{where}.on_overlap must be one of {list(OVERLAP)}")
            if kind == "class" and (spec.class_field is None or spec.classes is None):
                raise TableError(f"{where}: kind class needs class_field and classes (value -> code)")
            if kind == "binary" and (spec.class_field is not None or spec.classes is not None):
                raise TableError(f"{where}: kind binary takes every feature as positive; "
                                 "use where: to select, or kind: class")
            if spec.domain is not None:
                _check_keys(f"{where}.domain", spec.domain, {"raster", "polygons", "layer", "where"})
                if ("raster" in spec.domain) == ("polygons" in spec.domain):
                    raise TableError(f"{where}.domain: give exactly one of raster or polygons")
                dom = dict(spec.domain)
                for key in ("raster", "polygons"):
                    if key in dom:
                        dom[key] = str(_resolve(base, dom[key]))
                spec.domain = dom
        if "negatives" in d:
            if d["negatives"] is None:
                raise TableError(f"{where}.negatives is empty; give min_distance_m, or remove the key")
            if spec.raster is not None or spec.outside != "background":
                raise TableError(f"{where}.negatives narrows background cells: it needs a point or polygon "
                                 "label with outside: background")
            spec.negatives = NegativesSpec.parse(f"{where}.negatives", d["negatives"])
        if spec.classes is not None:
            codes = [c for c in spec.classes.values() if c is not None]
            if any(c < 0 or c > 32767 for c in codes):
                raise TableError(f"{where}.classes: codes must be between 0 and 32767")
            if spec.outside == "background" and 0 in codes:
                raise TableError(f"{where}.classes: code 0 is the background; number classes from 1")
        return spec


@dataclass
class SplitSpec:
    name: str
    polygons: VectorSource | None = None
    raster: Path | None = None
    id_field: str | None = None
    assign: dict[str, list[Any]] = field(default_factory=dict)
    blocks: dict[str, Any] | None = None

    @classmethod
    def parse(cls, name: str, d: dict[str, Any], base: Path) -> "SplitSpec":
        where = f"splits.{name}"
        if "blocks" in d:
            _check_keys(where, d, {"blocks"})
            b = d["blocks"]
            _check_keys(f"{where}.blocks", b, {"size_m", "test_fraction", "val_fraction", "seed", "stratify"},
                        {"size_m", "test_fraction"})
            if "stratify" in b and b["stratify"] is None:
                raise TableError(f"{where}.blocks.stratify is empty; name a label, or remove the key")
            blocks = {"size_m": float(b["size_m"]), "test_fraction": float(b["test_fraction"]),
                      "val_fraction": float(b.get("val_fraction", 0.0)), "seed": int(b.get("seed", 42)),
                      "stratify": b.get("stratify")}
            if blocks["size_m"] <= 0:
                raise TableError(f"{where}.blocks.size_m must be > 0")
            if not (0 < blocks["test_fraction"] < 1 and 0 <= blocks["val_fraction"] < 1
                    and blocks["test_fraction"] + blocks["val_fraction"] < 1):
                raise TableError(f"{where}.blocks: fractions between 0 and 1, summing below 1")
            return cls(name=_check_name("split", name), blocks=blocks)
        _check_keys(where, d, {"polygons", "raster", "layer", "where", "id_field", "assign"}, {"assign"})
        if ("polygons" in d) == ("raster" in d):
            raise TableError(f"{where}: give exactly one of polygons, raster or blocks")
        assign = d["assign"]
        if not isinstance(assign, dict) or not assign:
            raise TableError(f"{where}.assign must map split names to ids (polygons) or values (raster)")
        bad = set(assign) - set(SPLIT_NAMES)
        if bad:
            raise TableError(f"{where}.assign: unknown split name(s) {sorted(bad)}; use {list(SPLIT_NAMES)}")
        out = {k: (v if isinstance(v, list) else [v]) for k, v in assign.items()}
        seen: dict[str, str] = {}
        for split, ids in out.items():
            for i in ids:
                if str(i) in seen:
                    raise TableError(f"{where}.assign: id {i!r} is in both {seen[str(i)]} and {split}")
                seen[str(i)] = split
        spec = cls(name=_check_name("split", name), assign=out)
        if "polygons" in d:
            if "id_field" not in d:
                raise TableError(f"{where}: polygons need id_field")
            spec.polygons = VectorSource(_resolve(base, d["polygons"]), d.get("layer"), dict(d.get("where") or {}))
            spec.id_field = str(d["id_field"])
        else:
            if "layer" in d or "where" in d or "id_field" in d:
                raise TableError(f"{where}: layer, where and id_field apply to polygons, not a raster")
            spec.raster = _resolve(base, d["raster"])
        return spec


@dataclass
class TerrainSpec:
    dem: Path
    products: list[str]
    cache_dir: Path | None = None

    @classmethod
    def parse(cls, d: dict[str, Any], base: Path) -> "TerrainSpec":
        _check_keys("terrain", d, {"dem", "products", "cache_dir"}, {"dem"})
        products = list(d.get("products") or TERRAIN_PRODUCTS)
        bad = sorted(set(products) - set(TERRAIN_PRODUCTS))
        if bad:
            raise TableError(f"terrain.products: unknown {bad}; choose from {list(TERRAIN_PRODUCTS)}")
        return cls(dem=_resolve(base, d["dem"]), products=products,
                   cache_dir=Path(d["cache_dir"]).expanduser() if d.get("cache_dir") else None)


@dataclass
class TableConfig:
    site: str
    extent: Path
    output: Path
    features: list[FeatureSpec]
    labels: list[LabelSpec] = field(default_factory=list)
    splits: list[SplitSpec] = field(default_factory=list)
    terrain: TerrainSpec | None = None
    data_dir: Path | None = None
    description: str = ""
    config_path: Path | None = None
    config_sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "TableConfig":
        path = Path(path)
        text = path.read_text()
        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise TableError(f"{path}: not valid YAML:\n{exc}") from exc
        cfg = cls.from_dict(raw, config_dir=path.parent)
        cfg.config_path = path.resolve()
        cfg.config_sha256 = hashlib.sha256(text.encode()).hexdigest()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any], config_dir: Path | None = None) -> "TableConfig":
        _check_keys("table config", raw, {"site", "data_dir", "extent", "output", "features", "labels",
                                          "splits", "terrain", "description"}, {"site", "extent", "output", "features"})
        root = config_dir or Path(".")
        data_dir = _resolve(root, raw["data_dir"]) if raw.get("data_dir") else None
        base = data_dir or root
        features_raw = raw["features"]
        if not isinstance(features_raw, dict) or not features_raw:
            raise TableError("features: give at least one feature")
        cfg = cls(site=_check_name("site", raw["site"]), extent=_resolve(base, raw["extent"]),
                  output=_resolve(root, raw["output"]),
                  features=[FeatureSpec.parse(n, d or {}, base) for n, d in features_raw.items()],
                  labels=[LabelSpec.parse(n, d or {}, base) for n, d in (raw.get("labels") or {}).items()],
                  splits=[SplitSpec.parse(n, d or {}, base) for n, d in (raw.get("splits") or {}).items()],
                  terrain=TerrainSpec.parse(raw["terrain"], base) if raw.get("terrain") else None,
                  data_dir=data_dir, description=str(raw.get("description", "")))
        if cfg.terrain is not None:
            clash = sorted({f"terrain_{p}" for p in cfg.terrain.products} & {f.name for f in cfg.features})
            if clash:
                raise TableError(f"features {clash} collide with the terrain products; rename them")
        if cfg.output.suffix != ".parquet":
            raise TableError(f"output must end in .parquet: {cfg.output}")
        kinds = {spec.name: spec.kind for spec in cfg.labels}
        for split in cfg.splits:
            strat = (split.blocks or {}).get("stratify")
            if strat is not None and strat not in kinds:
                raise TableError(f"splits.{split.name}.blocks.stratify: no label {strat!r}; labels are {sorted(kinds)}")
            if strat is not None and kinds[strat] == "continuous":
                raise TableError(f"splits.{split.name}.blocks.stratify: {strat!r} is continuous; stratify needs a "
                                 "binary or class label, whose positives are what the split balances")
        return cfg


def _resolve(base: Path, p: Any) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else base / q


# ─────────────────────────────────────────────────────────────── grid

@dataclass
class Grid:
    profile: dict[str, Any]
    valid: np.ndarray            # the extent raster's valid cells, which become rows

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.profile["height"]), int(self.profile["width"])

    @property
    def transform(self):
        return self.profile["transform"]

    @property
    def crs(self):
        return self.profile["crs"]


def _read(path: Path, grid: Grid | None, what: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Band 1 as a float array (float32 or float64, as stored) with nodata as NaN."""
    if not path.exists():
        raise FileNotFoundError(f"{what}: {path}")
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()
    if grid is not None and not grids_match(grid.profile, profile):
        raise GridMismatch(f"{what}: {path.name} is {profile['height']}x{profile['width']} "
                           f"{profile.get('crs')} {tuple(profile['transform'])[:6]}; the table grid is "
                           f"{grid.shape[0]}x{grid.shape[1]} {grid.crs} {tuple(grid.transform)[:6]}")
    out = arr.astype("float64" if arr.dtype == np.float64 else "float32")
    bad = ~np.isfinite(out)
    if nodata is not None and np.isfinite(nodata):
        bad |= arr == nodata
    out[bad] = np.nan
    return out, profile


def load_grid(cfg: TableConfig) -> Grid:
    arr, profile = _read(cfg.extent, None, "extent")
    if profile.get("crs") is None:
        raise TableError(f"extent {cfg.extent.name} declares no CRS")
    valid = np.isfinite(arr)
    if not valid.any():
        raise TableError(f"extent {cfg.extent.name} has no valid cells")
    return Grid(profile=profile, valid=valid)


# ─────────────────────────────────────────────────────────────── features

def build_feature(spec: FeatureSpec, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    arr, _ = _read(spec.path, grid, f"feature {spec.name}")
    if spec.scale is not None:
        arr = arr.astype("float64") * spec.scale
    values = arr[grid.valid]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise TableError(f"feature {spec.name}: no valid values inside the extent")
    if spec.valid_range is not None:
        lo, hi = spec.valid_range
        n_lo, n_hi = int((finite < lo).sum()), int((finite > hi).sum())
        if n_lo or n_hi:
            raise TableError(f"feature {spec.name}: {n_lo} values below {lo} and {n_hi} above {hi} "
                             f"(range {finite.min():.6g} to {finite.max():.6g}, units {spec.units!r}"
                             f"{', after scale ' + str(spec.scale) if spec.scale is not None else ''}). "
                             "Wrong units, or a fill value that is not declared as nodata")
    info = {"units": spec.units, "definition": spec.definition, "source": spec.source,
            "path": str(spec.path), "scale": spec.scale, "dtype": str(values.dtype),
            "valid_rows": int(finite.size), "min": float(finite.min()), "max": float(finite.max()),
            "median": float(np.median(finite))}
    return values, info


# ─────────────────────────────────────────────────────────────── terrain

def build_terrain(spec: TerrainSpec, grid: Grid, output: Path) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Terrain features from this repo's one terrain method, cached beside the table.

    The cache is reused only while the DEM's size and modification time are unchanged.
    """
    from .physics.terrain import TerrainParams, driving_index, mask_edges, plan_curvature, route, valid_cells
    from .raster import read_float, write_float

    _read(spec.dem, grid, "terrain.dem")                       # same grid, or GridMismatch
    cache = spec.cache_dir or output.with_name(output.stem + "_terrain")
    stamp = {"dem": str(spec.dem.resolve()), "size": spec.dem.stat().st_size,
             "mtime": spec.dem.stat().st_mtime, "alpha": TerrainParams().alpha}
    stamp_path = cache / "terrain.json"
    files = {p: cache / f"terrain_{p}.tif" for p in TERRAIN_PRODUCTS}
    fresh = stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp and all(f.exists() for f in files.values())
    if not fresh:
        print(f"  terrain: routing {spec.dem.name} (WhiteboxTools D-infinity) -> {cache}")
        work = cache / "_work"
        work.mkdir(parents=True, exist_ok=True)
        # WhiteboxTools cannot read every GeoTIFF flavour (a float predictor, for one) and then
        # writes nothing. Route a plain float32 copy with -9999 as nodata.
        dem_arr, dem_profile = read_float(spec.dem)
        dem = work / "dem.tif"
        plain = {k: v for k, v in dem_profile.items()
                 if k not in ("compress", "predictor", "tiled", "blockxsize", "blockysize", "interleave")}
        write_float(dem, dem_arr, {**plain, "driver": "GTiff"})
        slope, sca, profile = route(dem, work, "dem", fill_sinks=True)
        cell = abs(float(profile["transform"].a))
        write_float(files["slope"], slope, profile)
        write_float(files["drainage_area"], sca * cell, profile)
        write_float(files["G"], driving_index(sca, slope, TerrainParams().alpha), profile)
        curv, _ = plan_curvature(dem, work, "dem")
        write_float(files["curvature"], curv, profile)
        from .physics.terrain import _whitebox
        aspect_path = work / "aspect_dem.tif"
        _whitebox().aspect(str(dem), str(aspect_path))
        if not aspect_path.exists():
            raise TableError(f"WhiteboxTools wrote no aspect for {dem}")
        aspect, _ = read_float(aspect_path)
        write_float(files["aspect"], mask_edges(aspect, valid_cells(dem)), profile)
        stamp_path.write_text(json.dumps(stamp))
        shutil.rmtree(work)                  # intermediates; kept only when routing fails
    res = abs(float(grid.transform.a))
    out = {}
    for product in spec.products:
        units, desc = TERRAIN_PRODUCTS[product]
        arr, _ = _read(files[product], grid, f"terrain {product}")
        values = arr[grid.valid]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise TableError(f"terrain {product}: no valid values inside the extent")
        info = {"units": units, "definition": f"{desc}; pci-source-zones terrain on a {res:g} m DEM",
                "source": str(spec.dem), "path": str(files[product]), "scale": None, "dtype": str(values.dtype),
                "valid_rows": int(finite.size), "min": float(finite.min()), "max": float(finite.max()),
                "median": float(np.median(finite))}
        out[f"terrain_{product}"] = (values, info)
    return out


# ─────────────────────────────────────────────────────────────── labels

def _rasterize(geoms: list, grid: Grid, all_touched: bool) -> np.ndarray:
    from rasterio.features import rasterize

    if not geoms:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize([(g, 1) for g in geoms], out_shape=grid.shape, transform=grid.transform,
                     fill=0, all_touched=all_touched, dtype="uint8").astype(bool)


def _covers_a_cell(geom, grid: Grid, all_touched: bool) -> bool:
    """Whether one polygon labels at least one valid cell, rasterized in its own window only."""
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds

    h, w = grid.shape
    win = from_bounds(*geom.bounds, transform=grid.transform)
    r0, c0 = max(int(np.floor(win.row_off)) - 1, 0), max(int(np.floor(win.col_off)) - 1, 0)
    r1 = min(int(np.ceil(win.row_off + win.height)) + 1, h)
    c1 = min(int(np.ceil(win.col_off + win.width)) + 1, w)
    if r1 <= r0 or c1 <= c0:
        return False
    sub = rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0),
                    transform=rasterio.windows.transform(Window(c0, r0, c1 - c0, r1 - r0), grid.transform),
                    fill=0, all_touched=all_touched, dtype="uint8").astype(bool)
    return bool((sub & grid.valid[r0:r1, c0:c1]).any())


def _domain_mask(spec: LabelSpec, grid: Grid, base_where: str) -> np.ndarray:
    d = spec.domain
    if d is None:
        return np.ones(grid.shape, dtype=bool)
    if "raster" in d:
        arr, _ = _read(Path(d["raster"]).expanduser(), grid, f"{base_where}.domain")
        return np.isfinite(arr) & (arr != 0)
    src = VectorSource(Path(d["polygons"]).expanduser(), d.get("layer"), dict(d.get("where") or {}))
    mask = _rasterize(list(src.read(grid.crs).geometry), grid, all_touched=False)
    if not mask.any():
        raise TableError(f"{base_where}.domain rasterised to no cells; check its CRS and extent")
    return mask


def _class_codes(spec: LabelSpec, gdf) -> tuple[Any, list[int]]:
    """The code of every feature (None = dropped) and the codes in use."""
    if spec.kind == "binary":
        return [1] * len(gdf), [1]
    if spec.class_field not in gdf.columns:
        raise TableError(f"labels.{spec.name}: class_field {spec.class_field!r} not in {list(gdf.columns)}")
    values = gdf[spec.class_field].astype(str)
    unmapped = sorted(set(values) - set(spec.classes))
    if unmapped:
        raise TableError(f"labels.{spec.name}: values of {spec.class_field!r} with no code: {unmapped}. "
                         "Map each one, or map it to null to drop it on purpose")
    codes = [spec.classes[v] for v in values]
    return codes, sorted({c for c in codes if c is not None})


def _metric_cell(grid: Grid, where: str) -> float:
    """The cell size in metres. Distances and block sizes are given in metres."""
    a, e = abs(float(grid.transform.a)), abs(float(grid.transform.e))
    if grid.transform.b != 0 or grid.transform.d != 0:
        raise TableError(f"{where}: the grid is rotated or sheared; distances need a north-up grid")
    units = str(grid.crs.linear_units or "").lower() if grid.crs.is_projected else "degrees"
    if units not in ("metre", "meter", "metres", "meters", "m"):
        raise TableError(f"{where}: distances are in metres; this grid's CRS is in {units}")
    if not np.isclose(a, e, rtol=1e-6):
        raise TableError(f"{where}: distances need square cells; these are {a:g} x {e:g}")
    return a


def _apply_negatives(spec: LabelSpec, label: np.ndarray, grid: Grid, sources: np.ndarray) -> dict[str, Any]:
    """Leave unknown every background cell closer than min_distance_m to a mapped positive.

    `sources` is every cell a positive feature claims or touches, before conflicts, domain or
    extent masking: a small polygon whose cells end up unknown still keeps its neighbours
    from being a confident 0."""
    from scipy.ndimage import distance_transform_edt

    where = f"labels.{spec.name}.negatives"
    cell = _metric_cell(grid, where)
    background = (label == 0) & grid.valid
    if not sources.any():
        raise TableError(f"labels.{spec.name}: no positive cell, so no distance to one")
    near = distance_transform_edt(~sources) * cell < spec.negatives.min_distance_m
    removed = int((background & near).sum())
    if removed == 0:
        raise TableError(f"{where}: min_distance_m {spec.negatives.min_distance_m:g} removed no background "
                         "cell; the rule is inert (shorter than one cell?)")
    label[background & near] = np.nan
    kept = int(((label == 0) & grid.valid).sum())
    if kept == 0:
        raise TableError(f"{where}: every background cell is within min_distance_m of a positive")
    return {"min_distance_m": spec.negatives.min_distance_m, "background_before": int(background.sum()),
            "removed_near_a_positive": removed, "negatives": kept}


def _vector_label(spec: LabelSpec, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    where = f"labels.{spec.name}"
    src = spec.polygons or spec.points
    gdf = src.read(grid.crs)
    if spec.points is not None:
        gdf = gdf.explode(index_parts=False)
        kinds = set(gdf.geom_type)
        if kinds != {"Point"}:
            raise TableError(f"{where}: points layer holds {sorted(kinds)}; give point geometries")
    codes, used = _class_codes(spec, gdf)
    per_code = {c: np.zeros(grid.shape, dtype=bool) for c in used}
    dropped = sum(c is None for c in codes)

    unknown_extra = np.zeros(grid.shape, dtype=bool)        # cells a mapped feature touches but does not claim
    without_cells: dict[str, int] = {}
    if spec.polygons is not None:
        for code in used:
            geoms = [g for g, c in zip(gdf.geometry, codes) if c == code]
            per_code[code] = _rasterize(geoms, grid, spec.all_touched)
            empty = sum(1 for g in geoms if not _covers_a_cell(g, grid, spec.all_touched))
            if empty:
                without_cells[str(code)] = empty
        # A cell a polygon covers only in part (its centre outside), and a cell under a polygon whose
        # class was dropped on purpose, is not "mapped, no process": leave it unknown, never background.
        unknown_extra = _rasterize(list(gdf.geometry), grid, all_touched=True)
        n_outside = 0
        if without_cells:
            print(f"  {where}: polygons that cover no cell centre and add no labelled cell, by code: "
                  f"{without_cells} (set all_touched: true to keep them)")
    else:
        inv = ~grid.transform
        h, w = grid.shape
        n_outside = 0
        k = spec.buffer_cells
        for geom, code in zip(gdf.geometry, codes):
            col_f, row_f = inv * (geom.x, geom.y)
            r, c = int(np.floor(row_f)), int(np.floor(col_f))
            if not (0 <= r < h and 0 <= c < w):
                n_outside += 1
                continue
            block = (slice(max(r - k, 0), r + k + 1), slice(max(c - k, 0), c + k + 1))
            if code is None:                                  # dropped on purpose: unknown, not background
                unknown_extra[block] = True
                continue
            per_code[code][block] = True
        if n_outside:
            raise TableError(f"{where}: {n_outside} point(s) fall outside the grid; check the CRS, "
                             "or select points with where:")

    claimed = np.zeros(grid.shape, dtype=np.int16)
    label = np.full(grid.shape, np.nan, dtype="float32")
    for code, mask in per_code.items():
        claimed += mask
        label[mask] = code
    conflict = claimed > 1
    if spec.kind == "class" and conflict.any():
        clash = conflict & grid.valid
        n_clash = int(clash.sum())
        if spec.on_overlap == "error" and n_clash:
            raise TableError(f"{where}: {n_clash} cells are claimed by more than one class. Fix the "
                             "polygons, or set on_overlap: unknown to leave those cells unlabelled")
        label[conflict] = np.nan
    else:
        n_clash = 0

    domain = _domain_mask(spec, grid, where)
    labelled = claimed > 0
    if spec.outside == "background":
        label[~labelled & domain] = 0
    n_touched_unknown = int((unknown_extra & ~labelled & domain & grid.valid).sum())
    label[unknown_extra & ~labelled] = np.nan
    label[~domain] = np.nan
    label[~grid.valid] = np.nan
    negatives = None
    if spec.negatives is not None:
        sources = claimed > 0
        if spec.polygons is not None:
            sources |= _rasterize([g for g, c in zip(gdf.geometry, codes) if c is not None], grid, all_touched=True)
        negatives = _apply_negatives(spec, label, grid, sources)

    positives = int(np.isfinite(label[grid.valid]).sum() - (label[grid.valid] == 0).sum())
    if positives == 0:
        raise TableError(f"{where}: no positive cell inside the extent and domain. Check the path, layer, "
                         "filter, CRS and domain")
    info = {"kind": spec.kind, "source": str(src.path), "layer": src.layer, "where": src.where,
            "geometry": "polygons" if spec.polygons is not None else "points",
            "outside": spec.outside, "domain": spec.domain, "on_overlap": spec.on_overlap,
            "buffer_cells": spec.buffer_cells if spec.points is not None else None,
            "all_touched": spec.all_touched if spec.polygons is not None else None,
            "class_field": spec.class_field, "classes": spec.classes,
            "features_mapped": int(len(gdf) - dropped), "features_dropped": int(dropped),
            "features_without_cells": without_cells,
            "cells_touched_not_claimed_left_unknown": n_touched_unknown,
            "cells_with_conflicting_classes": n_clash, "definition": spec.definition,
            "description": spec.description, "negatives": negatives}
    return label, info


def _raster_label(spec: LabelSpec, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    where = f"labels.{spec.name}"
    arr, _ = _read(spec.raster, grid, where)
    finite = np.isfinite(arr)
    if spec.valid_range is not None:
        lo, hi = spec.valid_range
        v = arr[finite & grid.valid]
        n_bad = int(((v < lo) | (v > hi)).sum())
        if n_bad:
            raise TableError(f"{where}: {n_bad} values outside {spec.valid_range} "
                             f"(range {v.min():.6g} to {v.max():.6g})")
    rule = "value as stored"
    if spec.kind == "binary":
        if spec.threshold is not None:
            out = np.full(arr.shape, np.nan, dtype="float32")
            out[finite] = (arr[finite] >= spec.threshold).astype("float32")   # NaN stays NaN
            arr, rule = out, f"value >= {spec.threshold}"
        else:
            values = set(np.unique(arr[finite & grid.valid]).tolist())
            if not values <= {0.0, 1.0}:
                raise TableError(f"{where}: a binary raster must hold only 0 and 1, found "
                                 f"{sorted(values)[:8]}; give a threshold to binarise it")
    elif spec.kind == "class":
        v_all = arr[finite & grid.valid]
        if np.any(v_all != np.round(v_all)):
            raise TableError(f"{where}: a class raster must hold whole numbers (resampled with nearest, not "
                             "bilinear); found fractional values")
        if v_all.size and (v_all.min() < 0 or v_all.max() > 32767):
            raise TableError(f"{where}: class values must be between 0 and 32767; found {v_all.min():g} to "
                             f"{v_all.max():g} (an undeclared fill value?)")
        if spec.classes is not None:
            out = np.full(arr.shape, np.nan, dtype="float32")
            present = {str(int(v)) for v in np.unique(v_all)}
            unmapped = sorted(present - set(spec.classes))
            if unmapped:
                raise TableError(f"{where}: raster values with no code: {unmapped}; map each one, or to null")
            for raw, code in spec.classes.items():
                if code is not None:
                    out[finite & (arr == float(raw))] = code
            arr, rule = out, f"mapped {spec.classes}"
    arr = arr.copy()
    arr[~grid.valid] = np.nan
    values = arr[grid.valid]
    if not np.isfinite(values).any():
        raise TableError(f"{where}: no labelled cell inside the extent")
    if spec.kind != "continuous" and not np.nanmax(values) > 0:
        raise TableError(f"{where}: no positive cell inside the extent")
    info = {"kind": spec.kind, "source": str(spec.raster), "geometry": "raster", "rule": rule,
            "valid_range": spec.valid_range, "classes": spec.classes, "definition": spec.definition,
            "description": spec.description}
    return arr, info


def build_label(spec: LabelSpec, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    arr, info = _raster_label(spec, grid) if spec.raster is not None else _vector_label(spec, grid)
    values = arr[grid.valid]
    finite = np.isfinite(values)
    info["labelled_rows"] = int(finite.sum())
    if spec.kind == "continuous":
        v = values[finite]
        info.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))
    else:
        codes, counts = np.unique(values[finite].astype(int), return_counts=True)
        info["counts"] = {str(int(c)): int(n) for c, n in zip(codes, counts)}
    return values, info


# ─────────────────────────────────────────────────────────────── splits

SHARE_TOLERANCE = 0.05     # how far a block split's share of positives or negatives may miss its fraction


def _block_split(spec: SplitSpec, grid: Grid, labels: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Whole square blocks to test, val and train, in a seeded random order.

    With `stratify`, each mapped object (8-connected positive cells of that label) follows the
    block holding most of its cells, so an object cut by a block edge is never in two splits.
    Blocks with positives are placed first, on their positives; their negatives count towards
    the negatives' quota, which blocks without positives then fill; blocks with no labelled
    cell fill a quota of cells. The shares reached are checked, not assumed."""
    from scipy.ndimage import label as connected

    b, where = spec.blocks, f"splits.{spec.name}.blocks"
    cell = _metric_cell(grid, where)
    if b["size_m"] < cell:
        raise TableError(f"{where}.size_m {b['size_m']:g} is smaller than one {cell:g} m cell; blocks of "
                         "single cells put test cells beside training cells")
    k = int(round(b["size_m"] / cell))
    rows, cols = np.nonzero(grid.valid)
    uniq, inv = np.unique((rows // k).astype(np.int64) * (grid.shape[1] // k + 1) + cols // k, return_inverse=True)
    nb = len(uniq)
    follow = inv.copy()                                   # the block each row goes with
    strat, objects_cut, objects_left_cut = b["stratify"], 0, 0
    if strat is not None:
        if strat not in labels:
            raise TableError(f"{where}.stratify: no label {strat!r}; labels are {sorted(labels)}")
        lab_grid = labels[strat]
        lab = lab_grid[rows, cols]
        is_pos, is_neg = np.isfinite(lab) & (lab > 0), np.isfinite(lab) & (lab == 0)
        if not is_pos.any():
            raise TableError(f"{where}.stratify: label {strat!r} has no positive cell")
        comp = connected(np.isfinite(lab_grid) & (lab_grid > 0), structure=np.ones((3, 3), dtype=bool))[0][rows, cols]
        key, n = np.unique(comp[is_pos].astype(np.int64) * nb + inv[is_pos], return_counts=True)
        kcomp, kblock = key // nb, key % nb
        order = np.lexsort((-n, kcomp))                   # by object, the block with most of it first
        first = np.r_[True, kcomp[order][1:] != kcomp[order][:-1]]
        size = np.bincount(comp[is_pos], minlength=int(comp.max()) + 1)
        best, best_n = kcomp[order][first], n[order][first]
        movable = best[2 * best_n >= size[best]]              # a block holds at least half of the object
        majority = np.full(int(comp.max()) + 1, -1, dtype=np.int64)
        majority[movable] = kblock[order][first][2 * best_n >= size[best]]
        spans = np.bincount(kcomp) > 1
        objects_cut = int(spans.sum())
        objects_left_cut = int((spans & (majority[:len(spans)] < 0)).sum())
        moves = is_pos & (majority[comp] >= 0)
        follow[moves] = majority[comp[moves]]
        pos = np.bincount(follow, weights=is_pos.astype(float), minlength=nb)
        neg = np.bincount(follow, weights=is_neg.astype(float), minlength=nb)
        kind = np.where(pos > 0, 0, np.where(neg > 0, 1, 2))
    else:
        is_pos = is_neg = np.zeros(len(rows), dtype=bool)
        pos = neg = np.zeros(nb)
        kind = np.full(nb, 2)
    cells = np.bincount(follow, minlength=nb).astype(float)
    free = cells * (kind == 2)                            # cells of blocks with no labelled cell
    measure, total = (pos, neg, free), (pos.sum(), neg.sum(), free.sum())
    fractions = {"test": b["test_fraction"], "val": b["val_fraction"]}
    filled = {split: np.zeros(3) for split in fractions}
    role = np.zeros(nb, dtype=np.int8)                    # index into SPLIT_NAMES; 0 = train
    order = np.random.default_rng(b["seed"]).permutation(nb)
    for i in order[np.argsort(kind[order], kind="stable")]:   # blocks with positives first
        q = kind[i]
        for split, frac in fractions.items():
            if frac > 0 and filled[split][q] + measure[q][i] / 2 <= frac * total[q]:
                role[i] = SPLIT_NAMES.index(split)
                filled[split] += (pos[i], neg[i], free[i])
                break
    code = role[follow]
    names = np.array(SPLIT_NAMES, dtype=object)[code]
    used = [i for i, _ in enumerate(SPLIT_NAMES) if (code == i).any()]
    info: dict[str, Any] = {"source": "blocks", **b, "block_cells": k, "blocks": int(nb),
                            "blocks_by_split": {SPLIT_NAMES[i]: int((role == i).sum()) for i in used},
                            "rows": {SPLIT_NAMES[i]: int((code == i).sum()) for i in used}}
    checks = {"rows": np.ones(len(code), dtype=bool)}
    if strat is not None:
        info["positives_by_split"] = {SPLIT_NAMES[i]: int((is_pos & (code == i)).sum()) for i in used}
        info["negatives_by_split"] = {SPLIT_NAMES[i]: int((is_neg & (code == i)).sum()) for i in used}
        info["objects_cut_by_a_block_edge"] = objects_cut
        info["objects_left_cut_no_block_holds_half"] = objects_left_cut
        info["cells_following_another_block"] = int((follow != inv).sum())
        if objects_left_cut:
            print(f"  {where}: {objects_left_cut} of {objects_cut} objects cut by a block edge have no block "
                  "holding half of them and stay cut between splits")
        checks = {"positives": is_pos, **({"negatives": is_neg} if is_neg.any() else {})}
    shares = {}
    for split in ["train"] + [s for s, f in fractions.items() if f > 0]:
        here = code == SPLIT_NAMES.index(split)
        if not here.any():
            raise TableError(f"{where}: {split} has no row; use smaller blocks or other fractions")
        if strat is not None and not (is_pos & here).any():
            raise TableError(f"{where}: no positive landed in {split}; use smaller blocks or another seed")
    for split, frac in fractions.items():
        if frac == 0:
            continue
        here = code == SPLIT_NAMES.index(split)
        for what, mask in checks.items():
            share = float((mask & here).sum() / mask.sum())
            shares[f"{split}_{what}"] = round(share, 4)
            if abs(share - frac) > SHARE_TOLERANCE:
                raise TableError(f"{where}: {split} holds {share:.1%} of the {what}, not {frac:.0%} (tolerance "
                                 f"{SHARE_TOLERANCE:.0%}); the blocks are too coarse for this site: use smaller "
                                 "blocks, or another seed")
    info["shares"] = shares
    return names, uniq[follow], info


def build_split(spec: SplitSpec, grid: Grid,
                labels: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Split name per row (None outside every assignment) and the group (polygon or block) per row."""
    where = f"splits.{spec.name}"
    if spec.blocks is not None:
        return _block_split(spec, grid, labels or {})
    code = np.full(grid.shape, -1, dtype=np.int8)          # index into SPLIT_NAMES, -1 = unassigned
    group = None
    if spec.polygons is not None:
        gdf = spec.polygons.read(grid.crs)
        if spec.id_field not in gdf.columns:
            raise TableError(f"{where}: id_field {spec.id_field!r} not in {list(gdf.columns)}")
        ids = gdf[spec.id_field].astype(str)
        group = np.full(grid.shape, -1, dtype=np.int64)
        for split, wanted in spec.assign.items():
            wanted = [str(i) for i in wanted]
            missing = sorted(set(wanted) - set(ids))
            if missing:
                raise TableError(f"{where}: ids {missing} for {split} not found in {spec.polygons.path.name}")
            for pid in wanted:
                mask = _rasterize(list(gdf.geometry[ids == pid]), grid, all_touched=False)
                clash = mask & (code >= 0)
                if clash.any():
                    raise TableError(f"{where}: polygon {pid} ({split}) overlaps another assigned polygon "
                                     f"in {int(clash.sum())} cells")
                if not (mask & grid.valid).any():
                    raise TableError(f"{where}: polygon {pid} ({split}) covers no valid cell")
                code[mask] = SPLIT_NAMES.index(split)
                try:
                    group[mask] = int(pid)
                except ValueError:
                    raise TableError(f"{where}: id_field values must be integers to serve as groups, got {pid!r}")
        source = str(spec.polygons.path)
    else:
        arr, _ = _read(spec.raster, grid, where)
        for split, values in spec.assign.items():
            for v in values:
                mask = np.isfinite(arr) & (arr == float(v))
                if not (mask & grid.valid).any():
                    raise TableError(f"{where}: no valid cell has value {v} ({split})")
                code[mask] = SPLIT_NAMES.index(split)
        source = str(spec.raster)
    row_codes = code[grid.valid]
    rows = np.array([None, *SPLIT_NAMES], dtype=object)[row_codes + 1]
    counts = {s: int((row_codes == i).sum()) for i, s in enumerate(SPLIT_NAMES) if (row_codes == i).any()}
    info = {"source": source, "id_field": spec.id_field, "assign": spec.assign, "rows": counts}
    return rows, (group[grid.valid] if group is not None else None), info


# ─────────────────────────────────────────────────────────────── write

def write_table(cfg: TableConfig, *, overwrite: bool = False) -> dict[str, Any]:
    """Build the table and write it as one Parquet file. Returns the metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if cfg.output.exists() and not overwrite:
        raise TableError(f"{cfg.output} exists; pass --overwrite to replace it")
    names = [f.name for f in cfg.features] + [f"terrain_{p}" for p in (cfg.terrain.products if cfg.terrain else [])] \
        + [f"label_{l.name}" for l in cfg.labels] \
        + [f"split_{s.name}" for s in cfg.splits] + [f"group_{s.name}" for s in cfg.splits]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise TableError(f"duplicate column names: {dupes}")

    grid = load_grid(cfg)
    h, w = grid.shape
    rows, cols = np.nonzero(grid.valid)
    t = grid.transform
    x = t.c + (cols + 0.5) * t.a + (rows + 0.5) * t.b
    y = t.f + (cols + 0.5) * t.d + (rows + 0.5) * t.e
    columns: dict[str, Any] = {
        "cell_id": pa.array(rows.astype(np.int64) * w + cols, pa.int64()),
        "row": pa.array(rows.astype(np.int32)), "col": pa.array(cols.astype(np.int32)),
        "x": pa.array(x.astype(np.float64)), "y": pa.array(y.astype(np.float64)),
    }
    print(f"{cfg.site}: {len(rows):,} rows from {cfg.extent.name} ({h} x {w}, {grid.crs})")

    meta_features = {}
    for spec in cfg.features:
        values, info = build_feature(spec, grid)
        columns[spec.name] = pa.array(values, from_pandas=True)
        meta_features[spec.name] = info
        print(f"  feature {spec.name:<26} {info['units']:<14} valid {info['valid_rows']:>10,}  "
              f"median {info['median']:.4g}")

    if cfg.terrain is not None:
        for name, (values, info) in build_terrain(cfg.terrain, grid, cfg.output).items():
            columns[name] = pa.array(values, from_pandas=True)
            meta_features[name] = info
            print(f"  feature {name:<26} {info['units']:<14} valid {info['valid_rows']:>10,}  "
                  f"median {info['median']:.4g}")

    label_grids: dict[str, np.ndarray] = {}
    meta_labels = {}
    for spec in cfg.labels:
        values, info = build_label(spec, grid)
        full = np.full(grid.shape, np.nan, dtype="float32")
        full[grid.valid] = values
        label_grids[spec.name] = full
        if spec.kind == "continuous":
            columns[f"label_{spec.name}"] = pa.array(values.astype("float32"), from_pandas=True)
        else:
            mask = ~np.isfinite(values)
            columns[f"label_{spec.name}"] = pa.array(np.where(mask, 0, values).astype(np.int16),
                                                     mask=mask, type=pa.int16())
        meta_labels[spec.name] = info
        detail = info.get("counts") or f"range {info['min']:.3g} to {info['max']:.3g}"
        print(f"  label   {spec.name:<26} {spec.kind:<14} labelled {info['labelled_rows']:>10,}  {detail}")
        if info.get("negatives"):
            n = info["negatives"]
            print(f"          negatives: {n['negatives']:,} of {n['background_before']:,} background cells, "
                  f"{n['removed_near_a_positive']:,} within {n['min_distance_m']:g} m of a positive left unknown")

    meta_splits = {}
    for spec in cfg.splits:
        names_, group, info = build_split(spec, grid, label_grids)
        columns[f"split_{spec.name}"] = pa.array(list(names_), pa.string())
        if group is not None:
            columns[f"group_{spec.name}"] = pa.array(group, mask=group < 0, type=pa.int64())
        meta_splits[spec.name] = info
        print(f"  split   {spec.name:<26} {info['rows']}")

    crs = grid.crs
    meta = {
        "contract": CONTRACT, "site": cfg.site, "description": cfg.description,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "writer": f"pci-source-zones {_version()} table",
        "config": str(cfg.config_path) if cfg.config_path else None, "config_sha256": cfg.config_sha256,
        "grid": {"crs_wkt": crs.to_wkt(), "epsg": crs.to_epsg(), "transform": list(t)[:6],
                 "width": w, "height": h, "extent": str(cfg.extent)},
        "rows": int(len(rows)), "features": meta_features, "labels": meta_labels, "splits": meta_splits,
    }
    table = pa.table(columns)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           META_KEY: json.dumps(meta, default=str).encode()})
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.output.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    back = pq.read_table(tmp)
    if back.num_rows != table.num_rows or back.schema.metadata.get(META_KEY) is None:
        tmp.unlink()
        raise TableError("read-back of the written table failed")
    tmp.replace(cfg.output)
    print(f"wrote {cfg.output} ({cfg.output.stat().st_size / 1e6:.1f} MB, {table.num_columns} columns)")
    return meta


def read_metadata(path: str | Path) -> dict[str, Any]:
    """The `source_zones_table` metadata of a written table."""
    import pyarrow.parquet as pq

    raw = pq.read_schema(path).metadata or {}
    if META_KEY not in raw:
        raise TableError(f"{path} carries no {META_KEY.decode()} metadata; it was not written by this command")
    return json.loads(raw[META_KEY])


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("pci-source-zones")
    except Exception:
        return "unknown"
