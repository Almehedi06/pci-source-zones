# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:16:01 2026

@author: lguido
"""

from osgeo import gdal, osr
import os
import glob
from pathlib import Path
from collections import defaultdict

# ---- Configuration (hardcoded) ----
ROOT_DIR = r"E:\OpenTopography\rater folder\"
OUTPUT_TIF = os.path.join(ROOT_DIR, "name_mosaic.tif")

# ---- Helpers ----
def find_tifs_recursive(root: str):
    """Find all .tif tiles under root recursively."""
    root_path = Path(root)
    return [str(p) for p in root_path.rglob("*.tif")]

def get_srs_wkt(ds):
    srs = ds.GetSpatialRef()
    return srs.ExportToWkt() if srs else None

def srs_key_from_wkt(wkt: str):
    s = osr.SpatialReference()
    s.ImportFromWkt(wkt)
    auth = s.GetAuthorityCode(None)
    return f"EPSG:{auth}" if auth else wkt

def choose_majority_srs(tif_paths):
    counts = defaultdict(int)
    wkts = {}
    file_wkts = {}
    for p in tif_paths:
        ds = gdal.Open(str(p), gdal.GA_ReadOnly)
        if ds is None:
            print(f"  [warn] Could not open raster: {p}")
            continue
        wkt = get_srs_wkt(ds)
        ds = None
        if wkt:
            key = srs_key_from_wkt(wkt)
            counts[key] += 1
            wkts[key] = wkt
            file_wkts[p] = wkt
    if not counts:
        raise RuntimeError("No valid CRS found in any tiles.")
    maj_key = max(counts, key=counts.get)
    return maj_key, wkts[maj_key], file_wkts

def reproject_if_needed(src_path: str, target_wkt: str, tmp_dir: str, resample="bilinear"):
    ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open source raster for reprojection: {src_path}")
    src_wkt = get_srs_wkt(ds)
    ds = None
    if src_wkt == target_wkt:
        return src_path

    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, os.path.basename(src_path))

    warp_opts = gdal.WarpOptions(
        dstSRS=target_wkt,
        resampleAlg=resample,
        multithread=True,
        format="GTiff",
        creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"]
    )
    gdal.Warp(str(out_path), str(src_path), options=warp_opts)
    return out_path

def safe_probe_nodata(raster_path: str):
    """Safely get NoData from band 1; return None if unavailable."""
    ds = gdal.Open(str(raster_path), gdal.GA_ReadOnly)
    if ds is None:
        print(f"  [warn] Cannot open for NoData probe: {raster_path}")
        return None
    band = ds.GetRasterBand(1)
    ds = None
    if band is None:
        print(f"  [warn] No band 1 found: {raster_path}")
        return None
    try:
        return band.GetNoDataValue()
    except Exception:
        # Some environments/types can throw here; continue without NoData
        return None

# ---- Main ----
def main():
    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

    tifs = find_tifs_recursive(ROOT_DIR)
    if not tifs:
        print("No .tif files found under ROOT_DIR.")
        return

    maj_key, maj_wkt, file_wkts = choose_majority_srs(tifs)
    print(f"Using majority CRS: {maj_key}")

    tmp_dir = os.path.join(ROOT_DIR, "_tmp_reprojected")
    final_srcs = []
    for s in tifs:
        # Use the known WKT for each file where available; else assume reprojection needed
        src_wkt = file_wkts.get(s)
        if src_wkt == maj_wkt:
            final_srcs.append(s)
        else:
            final_srcs.append(reproject_if_needed(s, maj_wkt, tmp_dir, resample="bilinear"))

    # Probe NoData safely (OK if None)
    nodata = safe_probe_nodata(final_srcs[0])

    # Build VRT in memory
    vrt_opts = gdal.BuildVRTOptions(
        resolution="highest",
        srcNodata=nodata,
        VRTNodata=nodata
    )
    vrt_ds = gdal.BuildVRT("", [str(p) for p in final_srcs], options=vrt_opts)
    if vrt_ds is None:
        print("Failed to build VRT.")
        return

    # Translate VRT -> GeoTIFF
    creation_opts = ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=IF_SAFER"]
    trans_opts = gdal.TranslateOptions(format="GTiff", creationOptions=creation_opts)
    out_ds = gdal.Translate(str(OUTPUT_TIF), vrt_ds, options=trans_opts)
    if out_ds is None:
        print("Failed to translate VRT to GeoTIFF.")
        return

    # Optional: overviews
    try:
        out_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
        print("Overviews built.")
    except Exception:
        print("Failed to build overviews (continuing).")

    print(f"Mosaic written: {OUTPUT_TIF}")

    # Cleanup handles
    out_ds = None
    vrt_ds = None

if __name__ == "__main__":
    main()

def main():
    gdal.UseExceptions()
    tifs = find_tifs_recursive(ROOT_DIR)
    if not tifs:
        print("No .tif files found.")
        return
    maj_key, maj_wkt, file_wkts = choose_majority_srs(tifs)
    print(f"Using majority CRS: {maj_key}")
    tmp_dir = os.path.join(ROOT_DIR, "_tmp_reprojected")
    final_srcs = [reproject_if_needed(s, maj_wkt, tmp_dir) for s in tifs]
    nodata = gdal.Open(final_srcs[0]).GetRasterBand(1).GetNoDataValue()
    vrt_ds = gdal.BuildVRT("", final_srcs, options=gdal.BuildVRTOptions(resolution="highest",
                                                                        srcNodata=nodata,
                                                                        VRTNodata=nodata))
    out_ds = gdal.Translate(OUTPUT_TIF, vrt_ds,
                            options=gdal.TranslateOptions(format="GTiff",
                                                          creationOptions=["COMPRESS=DEFLATE",
                                                                           "TILED=YES",
                                                                           "BIGTIFF=IF_SAFER"]))
    out_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
    print(f"Mosaic written: {OUTPUT_TIF}")

if __name__ == "__main__":
    main()

