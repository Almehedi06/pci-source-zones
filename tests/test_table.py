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
        _cfg(site, labels={"p": {"raster": "pobs.tif", "kind": "continuous", "treshold": 0.1}})


def test_raster_labels_continuous_and_threshold(site):
    cfg = _cfg(site, labels={
        "p": {"raster": "pobs.tif", "kind": "continuous", "valid_range": [0, 1]},
        "eroded": {"raster": "pobs.tif", "kind": "binary", "threshold": 0.5},
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
        "polygons": "inv.gpkg", "layer": "inventory", "where": {"epoch": "A"}, "kind": "class",
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
        "polygons": "inv.gpkg", "layer": "inventory", "kind": "class", "class_field": "Primary",
        "classes": {"slide": 1}, "outside": "unknown"}})
    with pytest.raises(TableError, match="no code"):
        write_table(cfg)


def test_outside_must_be_stated(site):
    with pytest.raises(TableError, match="outside must be stated"):
        _cfg(site, labels={"x": {"polygons": "inv.gpkg", "kind": "binary"}})


def test_overlapping_classes(site):
    gpd.GeoDataFrame({"Primary": ["a", "b"]}, geometry=[_cell_box(2, 2, 6, 6), _cell_box(4, 4, 8, 8)],
                     crs=CRS).to_file(site / "overlap.gpkg")
    spec = {"polygons": "overlap.gpkg", "kind": "class", "class_field": "Primary",
            "classes": {"a": 1, "b": 2}, "outside": "unknown"}
    with pytest.raises(TableError, match="more than one class"):
        write_table(_cfg(site, labels={"o": spec}))
    write_table(_cfg(site, labels={"o": {**spec, "on_overlap": "unknown"}}))
    df, meta = _read(site / "out" / "demo.parquet")
    lab = df.set_index(["row", "col"]).label_o
    assert pd_isna(lab[(4, 4)]) and lab[(2, 2)] == 1 and lab[(7, 7)] == 2
    assert meta["labels"]["o"]["cells_with_conflicting_classes"] == 4


def test_point_labels_with_buffer(site):
    cfg = _cfg(site, labels={"init": {"points": "pts.gpkg", "kind": "class", "class_field": "FailureTyp",
                                      "classes": {"LS": 1, "Runoff": 2}, "buffer_cells": 1,
                                      "outside": "unknown"}})
    write_table(cfg)
    df, _ = _read(cfg.output)
    lab = df.set_index(["row", "col"]).label_init
    assert (df.label_init == 1).sum() == 9 and (df.label_init == 2).sum() == 9
    assert lab[(7, 7)] == 1 and lab[(16, 26)] == 2 and pd_isna(lab[(10, 10)])


def test_points_outside_grid_are_an_error(site):
    gpd.GeoDataFrame({"k": [1]}, geometry=[Point(0, 0)], crs=CRS).to_file(site / "far.gpkg")
    cfg = _cfg(site, labels={"x": {"points": "far.gpkg", "kind": "binary", "outside": "unknown"}})
    with pytest.raises(TableError, match="outside the grid"):
        write_table(cfg)


def test_label_with_no_positives_is_an_error(site):
    _raster(site / "zeros.tif", np.zeros((H, W)))
    cfg = _cfg(site, labels={"z": {"raster": "zeros.tif", "kind": "binary"}})
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
