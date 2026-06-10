import numpy as np
from rasterio.transform import from_origin

from pci_source_zones.polygons import extract_source_polygons


def test_extract_source_polygons_from_small_raster():
    pci = np.array(
        [
            [0.1, 0.8, 0.8],
            [0.2, 0.9, 0.1],
            [0.1, 0.1, 0.1],
        ],
        dtype="float64",
    )
    profile = {
        "height": 3,
        "width": 3,
        "transform": from_origin(0, 3, 1, 1),
        "crs": "EPSG:32611",
    }
    gdf = extract_source_polygons(pci, profile, threshold=0.7, min_area_m2=0)
    assert len(gdf) == 1
    assert int(gdf.iloc[0]["mass__wasting_id"]) == 1
    assert np.isclose(gdf.iloc[0]["area_m2"], 3.0)
    assert gdf.iloc[0]["pci_mean"] > 0.8
