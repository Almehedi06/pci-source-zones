"""Site ML tables: features with units, labels from rasters, polygons and points, splits."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from pci_source_zones.cli import main
from pci_source_zones.raster import GridMismatch
from pci_source_zones.table import META_KEY, TableConfig, TableError, read_metadata, write_table

H, W, RES = 20, 30, 10.0
X0, Y0 = 1000.0, 2000.0            # upper-left corner
CRS = "EPSG:32610"


def _raster(path, arr, nodata=-9999.0, dtype="float32", origin=(X0, Y0)):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                crs=CRS, nodata=nodata, transform=from_origin(origin[0], origin[1], RES, RES))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _cell_box(r0, c0, r1, c1):
    """Polygon covering rows r0..r1-1 and cols c0..c1-1 exactly."""
    return box(X0 + c0 * RES, Y0 - r1 * RES, X0 + c1 * RES, Y0 - r0 * RES)


def _cell_point(r, c):
    return Point(X0 + (c + 0.5) * RES, Y0 - (r + 0.5) * RES)


@pytest.fixture
def site(tmp_path):
    rng = np.random.default_rng(0)
    elev = 100 + rng.random((H, W)) * 50
    elev[0, :] = -9999.0                                   # top row outside the extent
    _raster(tmp_path / "elev.tif", elev)
    _raster(tmp_path / "slope.tif", rng.random((H, W)))
    thick = np.full((H, W), 120.0)                         # centimetres
    thick[5, 5] = -9999.0
    _raster(tmp_path / "thick_cm.tif", thick)
    pobs = rng.random((H, W))
    pobs[2:4, :] = -9999.0
    _raster(tmp_path / "pobs.tif", pobs)

    gpd.GeoDataFrame({"Primary": ["slide", "flow", "flow"], "epoch": ["A", "A", "B"]},
                     geometry=[_cell_box(2, 2, 5, 5), _cell_box(10, 10, 12, 14), _cell_box(15, 0, 18, 3)],
                     crs=CRS).to_file(tmp_path / "inv.gpkg", layer="inventory")
    gpd.GeoDataFrame({"name": ["mapped"]}, geometry=[_cell_box(1, 0, 14, 20)],
                     crs=CRS).to_file(tmp_path / "inv.gpkg", layer="mapping_area")
    gpd.GeoDataFrame({"FailureTyp": ["LS", "Runoff"]}, geometry=[_cell_point(8, 8), _cell_point(15, 25)],
                     crs=CRS).to_file(tmp_path / "pts.gpkg")
    gpd.GeoDataFrame({"id": [1, 2, 3]},
                     geometry=[_cell_box(1, 0, 10, 15), _cell_box(10, 0, 20, 15), _cell_box(1, 15, 20, 30)],
                     crs=CRS).to_file(tmp_path / "split.gpkg")
    return tmp_path


def _cfg(site, **over):
    raw = {
        "site": "demo", "data_dir": str(site), "extent": "elev.tif", "output": str(site / "out" / "demo.parquet"),
        "features": {
            "elevation": {"path": "elev.tif", "units": "m", "definition": "DEM elevation"},
            "slope": {"path": "slope.tif", "units": "m/m", "definition": "slope, 10 m DEM"},
            "soil_thickness": {"path": "thick_cm.tif", "units": "m", "scale": 0.01, "valid_range": [0, 5],
                               "definition": "depth to lithic contact", "source": "test (cm)"},
        },
    }
    raw.update(over)
    return TableConfig.from_dict(raw)


def _read(path):
    t = pq.read_table(path)
    return t.to_pandas(), json.loads(t.schema.metadata[META_KEY])


def test_features_positions_and_metadata(site):
    cfg = _cfg(site)
    write_table(cfg)
    df, meta = _read(cfg.output)
    assert len(df) == (H - 1) * W                          # the extent's nodata row is not a row
    assert list(df.columns[:5]) == ["cell_id", "row", "col", "x", "y"]
    assert (df.cell_id == df.row * W + df.col).all()
    r = df.iloc[0]
    assert r.x == X0 + (r.col + 0.5) * RES and r.y == Y0 - (r.row + 0.5) * RES
    assert df.soil_thickness.max() == pytest.approx(1.2)    # scaled to metres
    assert df.soil_thickness.isna().sum() == 1              # nodata stays missing, never 0 or -9999
    assert meta["contract"] == "source-zones-table/1"
    assert meta["grid"]["width"] == W and meta["grid"]["epsg"] == 32610
    assert meta["features"]["soil_thickness"]["units"] == "m"
    assert meta["features"]["soil_thickness"]["scale"] == 0.01
    assert read_metadata(cfg.output)["site"] == "demo"


def test_valid_range_catches_wrong_units(site):
    cfg = _cfg(site)
    cfg.features[2].scale = None                           # centimetres declared as metres
    with pytest.raises(TableError, match="above 5"):
        write_table(cfg)


def test_units_and_definition_are_required(site):
    raw = {"site": "demo", "extent": str(site / "elev.tif"), "output": str(site / "x.parquet"),
           "features": {"slope": {"path": str(site / "slope.tif"), "units": "m/m"}}}
    with pytest.raises(TableError, match="definition"):
        TableConfig.from_dict(raw)


def test_unknown_keys_are_errors(site):
    with pytest.raises(TableError, match="unknown key"):
        _cfg(site, featurs={})
    with pytest.raises(TableError, match="unknown key"):
        _cfg(site, labels={"p": {"raster": "pobs.tif", "kind": "continuous", "definition": "test label", "treshold": 0.1}})


def test_raster_labels_continuous_and_threshold(site):
    cfg = _cfg(site, labels={
        "p": {"raster": "pobs.tif", "kind": "continuous", "definition": "test label", "valid_range": [0, 1]},
        "eroded": {"raster": "pobs.tif", "kind": "binary", "definition": "test label", "threshold": 0.5},
    })
    write_table(cfg)
    df, meta = _read(cfg.output)
    assert df.label_p.isna().sum() == 2 * W                # nodata rows 2-3 stay unlabelled
    nan_p = df.label_p.isna()
    assert df.label_eroded[nan_p].isna().all()             # NaN >= t is not "0"
    assert ((df.label_p >= 0.5) == (df.label_eroded == 1))[~nan_p].all()
    assert meta["labels"]["eroded"]["rule"] == "value >= 0.5"


def test_polygon_class_label_with_domain_and_background(site):
    cfg = _cfg(site, labels={"process": {
        "polygons": "inv.gpkg", "layer": "inventory", "where": {"epoch": "A"}, "kind": "class", "definition": "test label",
        "class_field": "Primary", "classes": {"slide": 1, "flow": 2}, "outside": "background",
        "domain": {"polygons": "inv.gpkg", "layer": "mapping_area"}}})
    write_table(cfg)
    df, meta = _read(cfg.output)
    lab = df.set_index(["row", "col"]).label_process
    assert lab[(3, 3)] == 1 and lab[(11, 12)] == 2
    assert lab[(1, 19)] == 0                                # mapped area, no polygon: background
    assert pd_isna(lab[(1, 25)])                            # outside the mapped area: unknown
    assert pd_isna(lab[(16, 1)])                            # epoch B polygon filtered out, and unmapped
    assert meta["labels"]["process"]["counts"] == {"0": int((df.label_process == 0).sum()),
                                                   "1": 9, "2": 8}


def pd_isna(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v) == "<NA>"


def test_unmapped_class_value_is_an_error(site):
    cfg = _cfg(site, labels={"process": {
        "polygons": "inv.gpkg", "layer": "inventory", "kind": "class", "definition": "test label", "class_field": "Primary",
        "classes": {"slide": 1}, "outside": "unknown"}})
    with pytest.raises(TableError, match="no code"):
        write_table(cfg)


def test_outside_must_be_stated(site):
    with pytest.raises(TableError, match="outside must be stated"):
        _cfg(site, labels={"x": {"polygons": "inv.gpkg", "kind": "binary", "definition": "test label"}})


def test_overlapping_classes(site):
    gpd.GeoDataFrame({"Primary": ["a", "b"]}, geometry=[_cell_box(2, 2, 6, 6), _cell_box(4, 4, 8, 8)],
                     crs=CRS).to_file(site / "overlap.gpkg")
    spec = {"polygons": "overlap.gpkg", "kind": "class", "definition": "test label", "class_field": "Primary",
            "classes": {"a": 1, "b": 2}, "outside": "unknown"}
    with pytest.raises(TableError, match="more than one class"):
        write_table(_cfg(site, labels={"o": spec}))
    write_table(_cfg(site, labels={"o": {**spec, "on_overlap": "unknown"}}))
    df, meta = _read(site / "out" / "demo.parquet")
    lab = df.set_index(["row", "col"]).label_o
    assert pd_isna(lab[(4, 4)]) and lab[(2, 2)] == 1 and lab[(7, 7)] == 2
    assert meta["labels"]["o"]["cells_with_conflicting_classes"] == 4


def test_point_labels_with_buffer(site):
    cfg = _cfg(site, labels={"init": {"points": "pts.gpkg", "kind": "class", "definition": "test label", "class_field": "FailureTyp",
                                      "classes": {"LS": 1, "Runoff": 2}, "buffer_cells": 1,
                                      "outside": "unknown"}})
    write_table(cfg)
    df, _ = _read(cfg.output)
    lab = df.set_index(["row", "col"]).label_init
    assert (df.label_init == 1).sum() == 9 and (df.label_init == 2).sum() == 9
    assert lab[(7, 7)] == 1 and lab[(16, 26)] == 2 and pd_isna(lab[(10, 10)])


def test_points_outside_grid_are_an_error(site):
    gpd.GeoDataFrame({"k": [1]}, geometry=[Point(0, 0)], crs=CRS).to_file(site / "far.gpkg")
    cfg = _cfg(site, labels={"x": {"points": "far.gpkg", "kind": "binary", "definition": "test label", "outside": "unknown"}})
    with pytest.raises(TableError, match="outside the grid"):
        write_table(cfg)


def test_label_with_no_positives_is_an_error(site):
    _raster(site / "zeros.tif", np.zeros((H, W)))
    cfg = _cfg(site, labels={"z": {"raster": "zeros.tif", "kind": "binary", "definition": "test label"}})
    with pytest.raises(TableError, match="no positive"):
        write_table(cfg)


def test_polygon_split_and_groups(site):
    cfg = _cfg(site, splits={"official": {"polygons": "split.gpkg", "id_field": "id",
                                          "assign": {"train": [1], "test": [2], "holdout": [3]}}})
    write_table(cfg)
    df, meta = _read(cfg.output)
    by = df.set_index(["row", "col"])
    assert by.split_official[(5, 5)] == "train" and by.group_official[(5, 5)] == 1
    assert by.split_official[(15, 5)] == "test" and by.split_official[(15, 20)] == "holdout"
    assert df.split_official.notna().all()
    assert meta["splits"]["official"]["rows"] == {"train": 135, "test": 150, "holdout": 285}


def test_split_id_in_two_splits_is_an_error(site):
    with pytest.raises(TableError, match="both"):
        _cfg(site, splits={"s": {"polygons": "split.gpkg", "id_field": "id",
                                 "assign": {"train": [1, 2], "test": [2]}}})


def test_grid_mismatch_is_loud(site):
    _raster(site / "shifted.tif", np.ones((H, W)), origin=(X0 + 5, Y0))
    cfg = _cfg(site)
    cfg.features.append(type(cfg.features[0])(name="shifted", path=site / "shifted.tif", units="-",
                                              definition="x"))
    with pytest.raises(GridMismatch):
        write_table(cfg)


def test_existing_output_needs_overwrite(site):
    cfg = _cfg(site)
    write_table(cfg)
    with pytest.raises(TableError, match="--overwrite"):
        write_table(cfg)
    write_table(cfg, overwrite=True)


def test_cli(site):
    cfg_path = site / "table.yaml"
    raw = {"site": "demo", "data_dir": str(site), "extent": "elev.tif", "output": str(site / "cli.parquet"),
           "features": {"slope": {"path": "slope.tif", "units": "m/m", "definition": "slope"}}}
    cfg_path.write_text(yaml.safe_dump(raw))
    assert main(["table", "--config", str(cfg_path)]) == 0
    assert (site / "cli.parquet").exists()
    assert main(["table", "--config", str(cfg_path)]) == 2          # exists, no --overwrite


# ───────────────────────────────────────────── terrain from this repo's one method

def _plane(site, name="dem.tif", gradient=0.2):
    rows, cols = np.mgrid[0:H, 0:W]
    dem = 500.0 - gradient * RES * rows + 0.001 * cols      # falls to the south, 0.2 m/m
    return _raster(site / name, dem)


def test_terrain_products_and_shared_definitions(site, tmp_path):
    _plane(site)
    raw = {"site": "demo", "data_dir": str(site), "extent": "dem.tif", "output": str(site / "t1.parquet"),
           "features": {"elevation": {"path": "dem.tif", "units": "m", "definition": "ground elevation"}},
           "terrain": {"dem": "dem.tif"}}
    write_table(TableConfig.from_dict(raw))
    df, meta = _read(site / "t1.parquet")
    for p in ("slope", "drainage_area", "G", "curvature", "aspect"):
        assert f"terrain_{p}" in df.columns
    assert not (site / "t1_terrain" / "_work").exists()                     # intermediates cleaned up
    inner = df[(df.row > 2) & (df.row < H - 3) & (df.col > 2) & (df.col < W - 3)]
    assert inner.terrain_slope.median() == pytest.approx(0.2, abs=0.01)
    assert (inner.terrain_drainage_area > 0).all()
    assert inner.terrain_aspect.median() == pytest.approx(180, abs=5)       # facing south
    assert df.terrain_slope.isna().sum() > 0                                 # the edge ring is nodata
    # a second site with the same cell size gets the same definitions, so the ML side can mix them
    other = tmp_path / "other"
    other.mkdir()
    _plane(other, gradient=0.3)
    raw2 = {**raw, "site": "other", "data_dir": str(other), "output": str(other / "t2.parquet")}
    write_table(TableConfig.from_dict(raw2))
    _, meta2 = _read(other / "t2.parquet")
    for p in ("slope", "drainage_area", "G", "curvature", "aspect"):
        assert meta["features"][f"terrain_{p}"]["definition"] == meta2["features"][f"terrain_{p}"]["definition"]
        assert "10 m DEM" in meta["features"][f"terrain_{p}"]["definition"]


def test_terrain_cache_is_reused(site, monkeypatch):
    _plane(site)
    raw = {"site": "demo", "data_dir": str(site), "extent": "dem.tif", "output": str(site / "c.parquet"),
           "features": {"elevation": {"path": "dem.tif", "units": "m", "definition": "e"}},
           "terrain": {"dem": "dem.tif", "products": ["slope", "G"]}}
    write_table(TableConfig.from_dict(raw))
    import pci_source_zones.physics.terrain as terrain
    monkeypatch.setattr(terrain, "route", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerouted")))
    write_table(TableConfig.from_dict(raw), overwrite=True)                 # cache hit: no routing
    df, _ = _read(site / "c.parquet")
    assert {"terrain_slope", "terrain_G"} <= set(df.columns) and "terrain_aspect" not in df.columns


def test_terrain_dem_must_be_on_the_grid(site):
    _raster(site / "shifted_dem.tif", np.full((H, W), 100.0), origin=(X0 + 5, Y0))
    raw = {"site": "demo", "data_dir": str(site), "extent": "elev.tif", "output": str(site / "g.parquet"),
           "features": {"elevation": {"path": "elev.tif", "units": "m", "definition": "e"}},
           "terrain": {"dem": "shifted_dem.tif"}}
    with pytest.raises(GridMismatch):
        write_table(TableConfig.from_dict(raw))


def test_terrain_name_clash_and_unknown_product(site):
    base = {"site": "demo", "data_dir": str(site), "extent": "elev.tif", "output": str(site / "x.parquet")}
    with pytest.raises(TableError, match="collide"):
        TableConfig.from_dict({**base, "terrain": {"dem": "elev.tif"},
                               "features": {"terrain_slope": {"path": "slope.tif", "units": "m/m", "definition": "s"}}})
    with pytest.raises(TableError, match="unknown"):
        TableConfig.from_dict({**base, "terrain": {"dem": "elev.tif", "products": ["roughness"]},
                               "features": {"slope": {"path": "slope.tif", "units": "m/m", "definition": "s"}}})


def test_terrain_reads_a_float_predictor_geotiff(site):
    """WhiteboxTools writes nothing for a float-predictor GeoTIFF; the writer routes a plain copy."""
    rows, cols = np.mgrid[0:H, 0:W]
    dem = (500.0 - 2.0 * rows + 0.01 * cols).astype("float32")
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32", crs=CRS, nodata=-9999.0,
                transform=from_origin(X0, Y0, RES, RES), compress="deflate", predictor=3)
    with rasterio.open(site / "dem_p3.tif", "w", **prof) as dst:
        dst.write(dem, 1)
    raw = {"site": "demo", "data_dir": str(site), "extent": "dem_p3.tif", "output": str(site / "p3.parquet"),
           "features": {"elevation": {"path": "dem_p3.tif", "units": "m", "definition": "e"}},
           "terrain": {"dem": "dem_p3.tif", "products": ["slope", "aspect"]}}
    write_table(TableConfig.from_dict(raw))
    df, _ = _read(site / "p3.parquet")
    assert df.terrain_slope.median() == pytest.approx(0.2, abs=0.01)


# ───────────────────────────────────────────── review fixes

def test_partly_covered_cells_and_dropped_classes_are_unknown_not_background(site):
    # one polygon covers the left half of cell (10, 10) only (centre outside), one is a dropped class
    half = box(X0 + 10 * RES, Y0 - 11 * RES, X0 + 10.4 * RES, Y0 - 10 * RES)
    gpd.GeoDataFrame({"Primary": ["slide", "rockfall", "slide"]},
                     geometry=[_cell_box(2, 2, 5, 5), _cell_box(6, 6, 8, 8), half],
                     crs=CRS).to_file(site / "partial.gpkg")
    cfg = _cfg(site, labels={"p": {
        "polygons": "partial.gpkg", "kind": "class", "class_field": "Primary", "definition": "d",
        "classes": {"slide": 1, "rockfall": None}, "outside": "background"}})
    write_table(cfg)
    df, meta = _read(cfg.output)
    lab = df.set_index(["row", "col"]).label_p
    assert lab[(3, 3)] == 1
    assert pd_isna(lab[(10, 10)])                     # partly covered: unknown, not "no process"
    assert pd_isna(lab[(6, 6)]) and pd_isna(lab[(7, 7)])   # under the dropped class: unknown
    assert lab[(15, 15)] == 0                         # untouched mapped area: background
    info = meta["labels"]["p"]
    assert info["features_without_cells"] == {"1": 1}  # the half-cell polygon labels nothing
    assert info["features_mapped"] == 2 and info["features_dropped"] == 1


def test_blank_units_or_definition_are_refused(site):
    raw = {"site": "demo", "extent": str(site / "elev.tif"), "output": str(site / "x.parquet"),
           "features": {"slope": {"path": str(site / "slope.tif"), "units": None, "definition": "s"}}}
    with pytest.raises(TableError, match="units is empty"):
        TableConfig.from_dict(raw)
    raw["features"]["slope"].update(units="m/m", definition=None)
    with pytest.raises(TableError, match="definition is empty"):
        TableConfig.from_dict(raw)


def test_labels_need_a_definition(site):
    with pytest.raises(TableError, match="definition"):
        _cfg(site, labels={"p": {"raster": "pobs.tif", "kind": "continuous"}})


def test_classes_only_on_class_labels(site):
    with pytest.raises(TableError, match="kind class only"):
        _cfg(site, labels={"p": {"raster": "pobs.tif", "kind": "binary", "threshold": 0.5,
                                 "classes": {"0": None, "1": 1}, "definition": "d"}})


def test_class_raster_must_be_whole_numbers_and_in_range(site):
    arr = np.ones((H, W))
    arr[4, 4] = 1.5
    _raster(site / "frac.tif", arr)
    cfg = _cfg(site, labels={"c": {"raster": "frac.tif", "kind": "class", "classes": {"1": 1}, "definition": "d"}})
    with pytest.raises(TableError, match="whole numbers"):
        write_table(cfg)
    arr[4, 4] = 40000
    _raster(site / "big.tif", arr)
    cfg = _cfg(site, labels={"c": {"raster": "big.tif", "kind": "class", "definition": "d"}})
    with pytest.raises(TableError, match="between 0 and 32767"):
        write_table(cfg)
    with pytest.raises(TableError, match="between 0 and 32767"):
        _cfg(site, labels={"c": {"raster": "frac.tif", "kind": "class", "classes": {"1": 40000},
                                 "definition": "d"}})


def test_multi_layer_geopackage_needs_a_layer(site):
    cfg = _cfg(site, labels={"p": {"polygons": "inv.gpkg", "kind": "class", "class_field": "Primary",
                                   "classes": {"slide": 1, "flow": 2}, "outside": "unknown", "definition": "d"}})
    with pytest.raises(TableError, match="name one with layer"):
        write_table(cfg)


def test_relative_output_and_data_dir_resolve_against_the_config(site, tmp_path, monkeypatch):
    cfg_dir = site / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "t.yaml").write_text(yaml.safe_dump({
        "site": "demo", "data_dir": "..", "extent": "elev.tif", "output": "out/rel.parquet",
        "features": {"slope": {"path": "slope.tif", "units": "m/m", "definition": "s"}}}))
    monkeypatch.chdir(tmp_path)
    cfg = TableConfig.load(cfg_dir / "t.yaml")
    assert cfg.output == cfg_dir / "out" / "rel.parquet" and cfg.extent == cfg_dir / ".." / "elev.tif"
    write_table(cfg)
    assert (cfg_dir / "out" / "rel.parquet").exists()


# ───────────────────────────────────────────── negatives and block splits

def _pts_label(**extra):
    return {"init": {"points": "pts.gpkg", "kind": "binary", "definition": "test label", "buffer_cells": 1,
                     "outside": "background", **extra}}


def test_negatives_keep_a_distance_from_positives(site):
    cfg = _cfg(site, labels=_pts_label(negatives={"min_distance_m": 30}))
    write_table(cfg)
    df, meta = _read(cfg.output)
    lab = df.set_index(["row", "col"]).label_init
    assert (df.label_init == 1).sum() == 18                       # positives untouched
    assert pd_isna(lab[(8, 11)])                                  # 20 m from the block around (8, 8)
    assert lab[(8, 12)] == 0                                      # 30 m: far enough
    n = meta["labels"]["init"]["negatives"]
    assert n["removed_near_a_positive"] > 0 and n["negatives"] == (df.label_init == 0).sum()


def test_inert_negatives_rule_is_an_error(site):
    with pytest.raises(TableError, match="inert"):
        write_table(_cfg(site, labels=_pts_label(negatives={"min_distance_m": 5})))   # under one 10 m cell


def test_negatives_need_a_vector_label_with_background(site):
    with pytest.raises(TableError, match="outside: background"):
        _cfg(site, labels={"x": {"raster": "pobs.tif", "kind": "binary", "threshold": 0.5, "definition": "d",
                                 "negatives": {"min_distance_m": 30}}})
    with pytest.raises(TableError, match="outside: background"):
        _cfg(site, labels=_pts_label(outside="unknown", negatives={"min_distance_m": 30}))
    with pytest.raises(TableError, match="unknown key"):
        _cfg(site, labels=_pts_label(negatives={"min_distance_m": 30, "change": "dz.tif"}))


def _many_objects(tmp_path, n=40, size=60, seed=0, crs=CRS, res=RES):
    """A flat site of size x size cells with n mapped points, so a split has many objects to place."""
    rng = np.random.default_rng(seed)
    prof = dict(driver="GTiff", height=size, width=size, count=1, dtype="float32", crs=crs, nodata=-9999.0,
                transform=from_origin(X0, Y0, res, res))
    with rasterio.open(tmp_path / "big.tif", "w", **prof) as dst:
        dst.write(np.ones((size, size), dtype="float32"), 1)
    rc = rng.integers(2, size - 2, (n, 2))
    pts = [Point(X0 + (c + 0.5) * res, Y0 - (r + 0.5) * res) for r, c in rc]
    gpd.GeoDataFrame({"k": range(n)}, geometry=pts, crs=crs).to_file(tmp_path / "many.gpkg")
    return {"site": "big", "data_dir": str(tmp_path), "extent": "big.tif", "output": str(tmp_path / "big.parquet"),
            "features": {"e": {"path": "big.tif", "units": "m", "definition": "flat"}},
            "labels": {"init": {"points": "many.gpkg", "kind": "binary", "definition": "test label", "buffer_cells": 1,
                                "outside": "background", "negatives": {"min_distance_m": 30}}}}


def _splits_per_object(df, split_col):
    """For each mapped object (8-connected positive cells), how many splits its cells are in."""
    from scipy.ndimage import label as connected
    grid = np.zeros((df.row.max() + 1, df.col.max() + 1), dtype=bool)
    pos = df[df.label_init == 1]
    grid[pos.row, pos.col] = True
    comp = connected(grid, structure=np.ones((3, 3), dtype=bool))[0][pos.row, pos.col]
    return pos.assign(comp=comp).groupby("comp")[split_col].nunique()


def test_block_split_keeps_blocks_and_objects_whole_with_both_shares_checked(tmp_path):
    raw = _many_objects(tmp_path)
    raw["splits"] = {"b": {"blocks": {"size_m": 100, "test_fraction": 0.3, "seed": 1, "stratify": "init"}}}
    write_table(TableConfig.from_dict(raw))
    df, meta = _read(tmp_path / "big.parquet")
    info = meta["splits"]["b"]
    assert (_splits_per_object(df, "split_b") == 1).all()             # no mapped object in two splits
    assert info["objects_cut_by_a_block_edge"] > 0 and info["cells_following_another_block"] > 0
    pos, neg = df.label_init == 1, df.label_init == 0
    test = df.split_b == "test"
    assert abs((pos & test).sum() / pos.sum() - 0.3) <= 0.05             # positives in proportion
    assert abs((neg & test).sum() / neg.sum() - 0.3) <= 0.05             # and negatives too
    assert info["positives_by_split"]["test"] == (pos & test).sum()
    assert info["negatives_by_split"]["test"] == (neg & test).sum()
    # every cell that did not follow an object keeps its own block's split
    plain = df[~pos]
    assert (plain.groupby("group_b").split_b.nunique() == 1).all()
    first = df.split_b.copy()
    write_table(TableConfig.from_dict(raw), overwrite=True)            # the seed fixes the draw
    assert (_read(tmp_path / "big.parquet")[0].split_b == first).all()


def test_block_split_too_coarse_for_the_site_is_an_error(tmp_path):
    raw = _many_objects(tmp_path)
    raw["splits"] = {"b": {"blocks": {"size_m": 300, "test_fraction": 0.3, "seed": 1, "stratify": "init"}}}
    with pytest.raises(TableError, match="too coarse|no positive landed"):
        write_table(TableConfig.from_dict(raw))
    raw["splits"] = {"b": {"blocks": {"size_m": 5, "test_fraction": 0.3}}}
    with pytest.raises(TableError, match="smaller than one"):
        write_table(TableConfig.from_dict(raw))


def test_distances_need_a_grid_in_metres(tmp_path):
    raw = _many_objects(tmp_path, crs="EPSG:4326", res=0.0001)          # degrees: 50 "m" would mean nothing
    with pytest.raises(TableError, match="metres"):
        write_table(TableConfig.from_dict(raw))


def test_negatives_keep_away_from_a_mapped_polygon_that_claims_no_cell(site):
    tiny = box(X0 + 200.5, Y0 - 53.0, X0 + 203.0, Y0 - 51.0)              # inside cell (5, 20), off its centre
    gpd.GeoDataFrame({"k": [1, 2]}, geometry=[tiny, _cell_box(15, 0, 18, 3)], crs=CRS).to_file(site / "tiny.gpkg")
    cfg = _cfg(site, labels={"x": {"polygons": "tiny.gpkg", "kind": "binary", "definition": "test label",
                                   "outside": "background", "negatives": {"min_distance_m": 30}}})
    write_table(cfg)
    lab = _read(cfg.output)[0].set_index(["row", "col"]).label_x
    assert pd_isna(lab[(5, 20)])                                         # touched, not claimed: unknown
    assert pd_isna(lab[(5, 22)]) and pd_isna(lab[(7, 20)])               # 20 m away: unknown, not a 0
    assert lab[(5, 23)] == 0                                             # 30 m away: a negative


def test_empty_negatives_key_is_an_error(site):
    with pytest.raises(TableError, match="empty"):
        _cfg(site, labels=_pts_label(negatives=None))


def test_block_split_with_nothing_in_test_is_an_error(site):
    blocks = {"size_m": 50, "test_fraction": 0.01, "stratify": "init"}
    with pytest.raises(TableError, match="test has no row|no positive landed in test"):
        write_table(_cfg(site, labels=_pts_label(), splits={"b": {"blocks": blocks}}))


def test_block_split_config_errors(site):
    with pytest.raises(TableError, match="no label"):
        write_table(_cfg(site, labels=_pts_label(),
                         splits={"b": {"blocks": {"size_m": 50, "test_fraction": 0.3, "stratify": "nope"}}}))
    with pytest.raises(TableError, match="fractions"):
        _cfg(site, splits={"b": {"blocks": {"size_m": 50, "test_fraction": 0.7, "val_fraction": 0.4}}})
    with pytest.raises(TableError, match="unknown key"):
        _cfg(site, splits={"b": {"blocks": {"size_m": 50, "test_fraction": 0.3}, "id_field": "id"}})


def test_stratify_needs_a_yes_no_or_class_label_and_cannot_be_empty(site):
    with pytest.raises(TableError, match="continuous"):
        _cfg(site, labels={"p": {"raster": "pobs.tif", "kind": "continuous", "definition": "d"}},
             splits={"b": {"blocks": {"size_m": 50, "test_fraction": 0.3, "stratify": "p"}}})
    with pytest.raises(TableError, match="stratify is empty"):
        _cfg(site, labels=_pts_label(), splits={"b": {"blocks": {"size_m": 50, "test_fraction": 0.3, "stratify": None}}})


def test_a_split_with_no_row_is_an_error(tmp_path):
    raw = _many_objects(tmp_path)
    raw["splits"] = {"b": {"blocks": {"size_m": 300, "test_fraction": 0.05}}}     # 4 blocks, none fits 5 %
    with pytest.raises(TableError, match="test has no row"):
        write_table(TableConfig.from_dict(raw))


def test_rotated_grid_is_refused_for_distances(tmp_path):
    from rasterio.transform import Affine
    prof = dict(driver="GTiff", height=30, width=30, count=1, dtype="float32", crs=CRS, nodata=-9999.0,
                transform=Affine.translation(X0, Y0) * Affine.rotation(45) * Affine.scale(RES, -RES))
    with rasterio.open(tmp_path / "rot.tif", "w", **prof) as dst:
        dst.write(np.ones((30, 30), dtype="float32"), 1)
    centre = prof["transform"] * (15.5, 15.5)
    gpd.GeoDataFrame({"k": [1]}, geometry=[Point(*centre)], crs=CRS).to_file(tmp_path / "p.gpkg")
    raw = {"site": "r", "data_dir": str(tmp_path), "extent": "rot.tif", "output": str(tmp_path / "r.parquet"),
           "features": {"e": {"path": "rot.tif", "units": "m", "definition": "flat"}},
           "labels": {"x": {"points": "p.gpkg", "kind": "binary", "definition": "d", "outside": "background",
                            "negatives": {"min_distance_m": 30}}}}
    with pytest.raises(TableError, match="rotated"):
        write_table(TableConfig.from_dict(raw))


def test_an_object_with_no_block_holding_half_stays_cut_and_is_counted(tmp_path):
    raw = _many_objects(tmp_path)
    pts = gpd.read_file(tmp_path / "many.gpkg")
    boxes = [box(p.x - 1.5 * RES, p.y - 1.5 * RES, p.x + 1.5 * RES, p.y + 1.5 * RES) for p in pts.geometry]
    strip = box(X0 + 5 * RES, Y0 - 31 * RES, X0 + 35 * RES, Y0 - 30 * RES)       # row 30, cols 5-34: 3 blocks
    gpd.GeoDataFrame({"k": range(len(boxes) + 1)}, geometry=boxes + [strip], crs=CRS).to_file(tmp_path / "poly.gpkg")
    raw["labels"] = {"init": {"polygons": "poly.gpkg", "kind": "binary", "definition": "test label",
                              "outside": "background", "negatives": {"min_distance_m": 30}}}
    raw["splits"] = {"b": {"blocks": {"size_m": 100, "test_fraction": 0.3, "seed": 1, "stratify": "init"}}}
    write_table(TableConfig.from_dict(raw))
    df, meta = _read(tmp_path / "big.parquet")
    info = meta["splits"]["b"]
    assert info["objects_left_cut_no_block_holds_half"] >= 1
    assert info["objects_cut_by_a_block_edge"] > info["objects_left_cut_no_block_holds_half"]
    strip_cells = df[(df.row == 30) & df.col.between(5, 34)]
    assert strip_cells.label_init.eq(1).all()
    assert strip_cells.groupby(strip_cells.col // 10).group_b.nunique().eq(1).all()   # each part keeps its own block
