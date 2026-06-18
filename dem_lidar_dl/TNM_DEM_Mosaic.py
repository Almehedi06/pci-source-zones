# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:07:59 2026

@author: lguido
"""

from osgeo import gdal
import os
import re
import glob
from collections import defaultdict

# ----------------------------
# Configuration
# ----------------------------

# Root folder where subfolders with DEM tiles are stored
ROOT_DIR = r"D:\DEMs"

# Output folder for mosaics
OUTPUT_DIR = os.path.join(ROOT_DIR, "_mosaics")

# Regex to extract dataset key from filename (e.g., 'CO_DRCOG_2020_B20')
DATASET_RE = re.compile(r"_([A-Z]{2}_.+?)\.tif$", re.IGNORECASE)


# ----------------------------
# Helper Functions
# ----------------------------

def find_tifs(root):
    """
    Find all .tif files in the given folder (no subfolder assumption).
    Returns a list of file paths.
    """
    return glob.glob(os.path.join(root, "*.tif"))


def dataset_key_from_name(fname):
    """
    Extract a dataset key from the filename using regex.
    Returns None if no match.
    """
    m = DATASET_RE.search(os.path.basename(fname))
    return m.group(1) if m else None


def build_vrt_dataset(src_list, src_nodata=None):
    """
    Build an in-memory VRT from a list of source files.
    - src_nodata specifies nodata value to propagate.
    - VRT allows us to mosaic multiple tiles without writing intermediate files.
    """
    opts = gdal.BuildVRTOptions(resolution='highest', srcNodata=src_nodata, VRTNodata=src_nodata)
    vrt_ds = gdal.BuildVRT("", src_list, options=opts)  # "" -> in-memory VRT
    return vrt_ds


def translate_to_tif_from_ds(src_ds, out_path, bigtiff="IF_SAFER"):
    """
    Translate a GDAL dataset (e.g., VRT) to a GeoTIFF on disk.
    Compression, tiling, and bigtiff options are applied.
    """
    creation_opts = [
        "COMPRESS=DEFLATE",
        "PREDICTOR=3",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        f"BIGTIFF={bigtiff}"
    ]
    opts = gdal.TranslateOptions(format="GTiff", creationOptions=creation_opts)
    out_ds = gdal.Translate(out_path, src_ds, options=opts)
    return out_ds


def get_srs_wkt(ds):
    """
    Get the WKT string of the dataset's spatial reference system.
    Returns None if no SRS defined.
    """
    srs = ds.GetSpatialRef()
    return srs.ExportToWkt() if srs else None


def srs_key_from_wkt(wkt):
    """
    Return a stable key for grouping CRS.
    Uses EPSG code if available; otherwise returns WKT.
    """
    srs = gdal.osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    auth = srs.GetAuthorityCode(None)
    return f"EPSG:{auth}" if auth else wkt


def choose_majority_srs(tif_paths):
    """
    Determine the most common CRS among a list of TIF files.
    Returns:
    - majority key (EPSG or WKT)
    - corresponding WKT
    - counts for each CRS
    - mapping of file paths to their WKT
    """
    srs_counts = defaultdict(int)
    srs_wkts = {}
    file_wkts = {}

    for p in tif_paths:
        ds = gdal.Open(p)
        wkt = get_srs_wkt(ds)
        ds = None

        if wkt:
            key = srs_key_from_wkt(wkt)
            srs_counts[key] += 1
            srs_wkts[key] = wkt
            file_wkts[p] = wkt

    majority_key = max(srs_counts, key=srs_counts.get)
    return majority_key, srs_wkts[majority_key], srs_counts, file_wkts


def reproject_if_needed(src_path, target_wkt, tmp_dir):
    """
    Reproject a raster to the target WKT if needed.
    Returns path to reprojected raster (or original if no reprojection needed).
    """
    ds = gdal.Open(src_path)
    src_wkt = get_srs_wkt(ds)
    ds = None

    if src_wkt == target_wkt:
        return src_path

    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, os.path.basename(src_path))

    warp_opts = gdal.WarpOptions(
        dstSRS=target_wkt,
        resampleAlg="bilinear",
        multithread=True,
        format="GTiff",
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "TILED=YES"
        ]
    )

    gdal.Warp(out_path, src_path, options=warp_opts)
    return out_path


# ----------------------------
# Main workflow
# ----------------------------

def main():
    # Enable GDAL exceptions to see real errors
    gdal.UseExceptions()

    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all tiles under subfolders
    tifs = find_tifs(ROOT_DIR)
    if not tifs:
        print("No .tif tiles found under directory folders. Check ROOT_DIR.")
        return

    # Group tiles by dataset key
    groups = defaultdict(list)
    for p in tifs:
        key = dataset_key_from_name(p)
        if key:
            groups[key].append(p)

    total_tiles = sum(len(v) for v in groups.values())
    print(f"Found {total_tiles} tiles in {len(groups)} dataset groups.")

    # Process each dataset group
    for key, srcs in sorted(groups.items()):
        # Deduplicate by filename (sometimes same tile in multiple folders)
        by_name = {}
        for s in srcs:
            by_name.setdefault(os.path.basename(s), s)
        srcs = list(by_name.values())

        print(f"\nProcessing dataset {key} with {len(srcs)} tiles")

        # Determine majority CRS to harmonize
        maj_key, maj_wkt, srs_counts, file_wkts = choose_majority_srs(srcs)

        print("  CRS distribution:")
        for k, v in srs_counts.items():
            print(f"    {k}: {v} tiles")
        
        print(f"  Using majority CRS: {maj_key}")

        # Reproject minority tiles to majority CRS
        tmp_reproj = os.path.join(OUTPUT_DIR, "_tmp_reprojected", key)
        final_srcs = []
        for s in srcs:
            if file_wkts[s] == maj_wkt:
                final_srcs.append(s)
            else:
                final_srcs.append(
                    reproject_if_needed(s, maj_wkt, tmp_reproj)
                )

        # Probe nodata value from first tile
        ds0 = gdal.Open(final_srcs[0], gdal.GA_ReadOnly)
        band0 = ds0.GetRasterBand(1)
        src_nodata = band0.GetNoDataValue()
        ds0 = None

        # Build VRT in memory
        vrt_ds = build_vrt_dataset(final_srcs, src_nodata=src_nodata)
        if vrt_ds is None:
            print(f"  Failed to build VRT for {key}")
            continue

        # Translate VRT to GeoTIFF mosaic
        tif = os.path.join(OUTPUT_DIR, f"{key}.tif")
        try:
            out_ds = translate_to_tif_from_ds(vrt_ds, tif, bigtiff="IF_SAFER")
        except Exception as e:
            print(f"  Translate failed for {key}: {e}")
            continue

        if out_ds is None:
            print(f"  Failed to translate VRT to GeoTIFF for {key}")
            continue

        print(f"  Mosaic written: {tif}")

        # Optional: build overviews for faster visualization
        try:
            out_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
            print("  Overviews built")
        except Exception:
            pass

        # Close datasets explicitly
        out_ds = None
        vrt_ds = None
        ds0 = None


if __name__ == "__main__":
    main()
