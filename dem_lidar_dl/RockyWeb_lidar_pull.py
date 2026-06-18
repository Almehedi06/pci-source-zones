# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:27:38 2026

@author: lguido
"""

import os
import requests
import subprocess
import json
from bs4 import BeautifulSoup
import shapefile  # pyshp
from shapely.geometry import shape, mapping, box
from pyproj import Transformer

# The code will convert the shapefile extent to a bounding box automatically
SHAPEFILE_PATH = r"path\to\.shp"

# Base URL for the LAZ data - inefficient becuase you have to locate them yourself/know the names
# which is why it is just a work around for TMN troubles when it is down
BASE_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "CA_LosAngeles_B23/CA_LosAngeles_1_B23/LAZ/" #CHANGE THIS AS NEEDED
)

# Folder where LAZ files will be saved
OUTPUT_DIR = r"out\folder"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_aoi_bbox():
    """
    Read AOI from shapefile and reproject to match LAZ CRS.
    Returns bbox as dict.
    """
    import fiona
    with fiona.open(SHAPEFILE_PATH) as shp:
        shp_crs = shp.crs
        # Merge all geometries
        merged = None
        for feat in shp:
            geom = shape(feat["geometry"])
            if merged is None:
                merged = geom
            else:
                merged = merged.union(geom)
        minx, miny, maxx, maxy = merged.bounds
        
        # Transform to LAZ CRS !!! MAKE SURE YOU UPDATE CRS 
        transformer = Transformer.from_crs(shp_crs, "EPSG:6340", always_xy=True)
        xmin, ymin = transformer.transform(minx, miny)
        xmax, ymax = transformer.transform(maxx, maxy)
        
        bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        print(f"AOI bounding box (reprojected): {bbox}")
        return bbox

def get_file_list():
    """
    Fetch the directory listing from RockyWeb and return .laz filenames.
    """
    print("Fetching file listing from RockyWeb...")
    r = requests.get(BASE_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    files = [a["href"] for a in soup.find_all("a") if a["href"].endswith(".laz")]
    print(f"Found {len(files)} LAZ files total.")
    return files


def get_laz_bbox(url):
    """
    Use PDAL to get the bounding box of a LAZ file.
    Returns (minx, miny, maxx, maxy) or None if PDAL fails.
    """
    try:
        cmd = ["pdal", "info", "--summary", url]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = json.loads(output)
        bbox = info["summary"]["bounds"]
        return bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
    except subprocess.CalledProcessError as e:
        print(f"PDAL error on: {url}")
        print(e.output.decode())
        return None


def overlaps(aoi, tile):
    """
    Check whether a tile bounding box overlaps the AOI bounding box.
    Returns True if they intersect.
    """
    minx, miny, maxx, maxy = tile
    return not (
        maxx < aoi["xmin"] or
        minx > aoi["xmax"] or
        maxy < aoi["ymin"] or
        miny > aoi["ymax"]
    )


def download_file(url):
    """
    Download a LAZ file to OUTPUT_DIR if it does not already exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_path = os.path.join(OUTPUT_DIR, os.path.basename(url))
    if os.path.exists(local_path):
        print(f"Already downloaded: {local_path}")
        return

    print(f"Downloading: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Saved to: {local_path}")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    # Determine AOI bounding box
    aoi_bbox = get_aoi_bbox()

    # Fetch the list of available LAZ files
    files = get_file_list()

    needed_tiles = []

    print("\nAnalyzing tile footprints (via PDAL)...\n")

    for f in files:
        url = BASE_URL + f
        bbox = get_laz_bbox(url)
        if bbox is None:
            continue

        if overlaps(aoi_bbox, bbox):
            print(f"Tile intersects AOI: {f}")
            needed_tiles.append(url)
        else:
            print(f"Outside AOI: {f}")

    print("\nSummary")
    print(f"Tiles needed for AOI: {len(needed_tiles)}\n")

    # Download only the tiles that intersect the AOI
    for url in needed_tiles:
        download_file(url)


if __name__ == "__main__":
    main()
