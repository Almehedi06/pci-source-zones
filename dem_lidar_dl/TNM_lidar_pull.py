# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:34:28 2026

@author: lguido
"""

import os
import time
import pathlib
import hashlib
from typing import Dict, List, Tuple, Optional

import shapefile  # pyshp: lightweight shapefile reader
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from pyproj import CRS, Transformer
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Input fire perimeter (or other polygon) shapefile.
# Each polygon is queried independently against TNM.
SHAPEFILE_PATH = r"C:\path to shapefile\fire.shp"

# Root output directory. A subfolder named after the shapefile will be created.
OUTPUT_DIR     = r"D:\LPC"

# TNM (The National Map) API endpoints and dataset selection.
TNM_BASE       = "https://tnmaccess.nationalmap.gov/api/v1"
PRODUCTS_URL   = f"{TNM_BASE}/products"

# Dataset name must match TNM’s internal label exactly.
DATASET_NAME   = "Lidar Point Cloud (LPC)"

# Request and pagination controls.
REQUEST_TIMEOUT       = 60     # seconds per HTTP request
MAX_PER_PAGE          = 1000   # maximum items per TNM page
SLEEP_BETWEEN_CALLS   = 0.5    # polite delay between API calls

# If True, URLs are listed but no files are downloaded.
DRY_RUN               = False

# Global cache to avoid re-downloading the same files across polygons.
GLOBAL_CACHE_DIR      = os.path.join(OUTPUT_DIR, "_cache")

# Tracks URLs already seen during this run.
SEEN_URLS: set        = set()

# If True, polygons whose bounding boxes do not look like lon/lat are skipped.
# This prevents accidental queries using projected coordinates.
SKIP_IF_NOT_LONLAT = True

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def read_prj_crs(prj_path: str) -> Optional[CRS]:
    """
    Read a CRS from a .prj file using pyproj.
    Returns None if the file is missing or cannot be parsed.
    """
    if not os.path.exists(prj_path):
        return None

    txt = pathlib.Path(prj_path).read_text(encoding="utf-8", errors="ignore")

    # Try multiple parsing strategies since ESRI WKT can be inconsistent.
    for try_fn in (CRS.from_wkt, CRS.from_user_input):
        try:
            crs = try_fn(txt)
            if crs and (crs.is_geographic or crs.is_projected):
                return crs
        except Exception:
            pass

    return None


def ensure_wgs84_transform(crs: Optional[CRS]) -> Optional[Transformer]:
    """
    Create a transformer from the input CRS to WGS84 (EPSG:4326).

    Returns None if:
      - CRS is unknown
      - CRS is already WGS84
    """
    if crs is None:
        return None

    wgs84 = CRS.from_epsg(4326)
    try_epsg = crs.to_epsg()

    if try_epsg == 4326 or crs == wgs84:
        return None

    return Transformer.from_crs(crs, wgs84, always_xy=True)


def transform_coords(
    transformer: Optional[Transformer],
    coords: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Apply a CRS transformation to a list of (x, y) coordinates.
    If no transformer is provided, coordinates are returned unchanged.
    """
    if transformer is None:
        return coords

    return [transformer.transform(x, y) for x, y in coords]


def shape_to_rings(shape) -> List[List[Tuple[float, float]]]:
    """
    Convert a pyshp shape into a list of coordinate rings.

    Handles multipart polygons by splitting on the 'parts' indices.
    """
    pts = shape.points
    parts = list(shape.parts) + [len(pts)]

    rings = []
    for i in range(len(parts) - 1):
        start, end = parts[i], parts[i + 1]
        rings.append(pts[start:end])

    return rings


def build_geometry_from_shape(shape, transformer: Optional[Transformer]):
    """
    Convert a pyshp shape into a shapely Polygon or MultiPolygon.

    This workflow only needs valid geometry to compute bounding boxes,
    so holes and topology are not explicitly reconstructed.
    """
    rings = shape_to_rings(shape)
    if not rings:
        return None

    # Transform coordinates to WGS84 if needed
    rings_ll = [transform_coords(transformer, r) for r in rings]

    polys = []
    for r in rings_ll:
        try:
            p = orient(Polygon(r), sign=1.0)
            if p.is_valid and not p.is_empty:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return None

    if len(polys) == 1:
        return polys[0]

    return MultiPolygon(polys)


def geom_to_bbox(geom) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) from a shapely geometry."""
    return geom.bounds


def tnm_query_products(params: Dict) -> Dict:
    """
    Query the TNM /products endpoint with basic retry logic
    to handle transient network or server issues.
    """
    tries = 0
    last_exc = None

    while tries < 3:
        tries += 1
        try:
            r = requests.get(PRODUCTS_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            time.sleep(SLEEP_BETWEEN_CALLS + 0.5 * tries)

    # If all retries fail, raise the last exception
    if last_exc:
        raise last_exc


def extract_download_urls(item: Dict) -> List[str]:
    """
    Extract direct download URLs from a TNM / ScienceBase item.

    The API is not perfectly consistent, so multiple possible fields
    are checked and results are de-duplicated.
    """
    urls: List[str] = []

    if item.get("downloadURL"):
        urls.append(item["downloadURL"])

    for key in ("webLinks", "distributionLinks", "files"):
        links = item.get(key)
        if not isinstance(links, list):
            continue

        for link in links:
            url = link.get("url") or link.get("href") or link.get("linkUrl")
            if not url:
                continue

            label = (link.get("type") or link.get("rel") or "").lower()
            if "download" in label or url.lower().startswith("http"):
                urls.append(url)

    # Preserve order while removing duplicates
    return list(dict.fromkeys(urls))


def safe_filename_from_url(url: str) -> str:
    """
    Generate a filesystem-safe filename from a URL.
    Falls back to a hashed name if the URL path is empty.
    """
    name = url.split("?")[0].split("/")[-1].strip()
    return name or f"download_{abs(hash(url))}.dat"


def cache_file_path(url: str) -> str:
    """
    Generate a deterministic cache path for a URL.
    A short SHA1 suffix avoids collisions between similarly named files.
    """
    base = safe_filename_from_url(url)
    suffix = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    name, ext = os.path.splitext(base)

    return os.path.join(GLOBAL_CACHE_DIR, f"{name}.{suffix}{ext}")


def ensure_link_or_copy(src: str, dst_dir: str, desired_name: Optional[str] = None) -> str:
    """
    Link or copy a cached file into the polygon-specific output directory.
    Hard links are preferred; copy is used as a fallback.
    """
    os.makedirs(dst_dir, exist_ok=True)

    dst_basename = desired_name or os.path.basename(src)
    dst = os.path.join(dst_dir, dst_basename)

    if os.path.exists(dst):
        return dst

    try:
        os.link(src, dst)
    except Exception:
        import shutil
        shutil.copy2(src, dst)

    return dst


def download_file(url: str, out_dir: str) -> Optional[str]:
    """
    Download a file into the global cache, then link or copy it
    into the polygon output directory.

    Uses streaming downloads and progress bars for large files.
    """
    base_name = safe_filename_from_url(url)
    cached = cache_file_path(url)

    # If we've already seen this URL, try to reuse cached content
    if url in SEEN_URLS:
        if os.path.exists(cached):
            return ensure_link_or_copy(cached, out_dir, desired_name=base_name)

    SEEN_URLS.add(url)

    if os.path.exists(cached):
        return ensure_link_or_copy(cached, out_dir, desired_name=base_name)

    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            tmp_path = f"{cached}.part"

            with open(tmp_path, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=base_name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            os.replace(tmp_path, cached)
            return ensure_link_or_copy(cached, out_dir, desired_name=base_name)

    except requests.RequestException:
        # Clean up partial downloads
        try:
            os.remove(f"{cached}.part")
        except Exception:
            pass

        return None


def safe_folder_name(s: str) -> str:
    """Remove unsafe characters from folder names."""
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in ("_", "-", "."))


def looks_like_lonlat_bbox(
    minx: float, miny: float, maxx: float, maxy: float
) -> Tuple[bool, str]:
    """
    Heuristic check that a bounding box looks like EPSG:4326 coordinates.

    This helps catch cases where projected coordinates are accidentally
    sent to TNM, which will quietly return zero results.
    """
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and
            -90.0  <= miny <= 90.0  and -90.0  <= maxy <= 90.0):
        return (False, "Coordinates fall outside lon/lat bounds")

    if minx > maxx or miny > maxy:
        return (False, "BBox min/max are inverted")

    width = abs(maxx - minx)
    height = abs(maxy - miny)

    if width > 60 or height > 60:
        return (False, f"BBox spans too many degrees (width={width:.2f}, height={height:.2f})")

    return (True, "BBox looks like lon/lat")


def looks_like_projected_bbox(minx: float, miny: float, maxx: float, maxy: float) -> bool:
    """
    Detect bounding boxes that resemble projected coordinates
    (UTM or State Plane) based on magnitude and extent.
    """
    width = abs(maxx - minx)
    height = abs(maxy - miny)

    meter_like_coords = (
        (150_000 <= abs(minx) <= 10_000_000) or
        (150_000 <= abs(maxx) <= 10_000_000) or
        (150_000 <= abs(miny) <= 10_000_000) or
        (150_000 <= abs(maxy) <= 10_000_000)
    )

    meter_like_extent = (width >= 50_000 or height >= 50_000)

    return meter_like_coords or meter_like_extent


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """
    Query TNM for LiDAR Point Cloud products using polygon bounding boxes
    and download all associated deliverables.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GLOBAL_CACHE_DIR, exist_ok=True)

    # Read shapefile and attempt to infer CRS
    reader = shapefile.Reader(SHAPEFILE_PATH)
    prj = os.path.splitext(SHAPEFILE_PATH)[0] + ".prj"

    crs = read_prj_crs(prj)
    transformer = ensure_wgs84_transform(crs)

    if crs is None:
        print("NOTE: .prj not found or could not be parsed.")
        print("      If the shapefile is projected, bbox checks may fail.")
    elif transformer is None:
        print("CRS is WGS84 (EPSG:4326); no transform needed.")
    else:
        try_epsg = crs.to_epsg()
        print(f"CRS detected: {crs.to_string()} (EPSG:{try_epsg if try_epsg else 'unknown'})")
        print("Transforming geometries to WGS84 for TNM queries.")

    shapefile_base = os.path.splitext(os.path.basename(SHAPEFILE_PATH))[0]
    out_dir = os.path.join(OUTPUT_DIR, safe_folder_name(shapefile_base))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {reader.numRecords} polygons in shapefile '{shapefile_base}'")
    print(f"All outputs will go to: {out_dir}")

    # Process each polygon independently
    for idx, sr in enumerate(reader.shapeRecords(), 1):
        geom = build_geometry_from_shape(sr.shape, transformer)
        if geom is None:
            print(f"Skipping polygon {idx}: invalid geometry")
            continue

        minx, miny, maxx, maxy = geom_to_bbox(geom)

        # Sanity check CRS before querying TNM
        is_ll, msg = looks_like_lonlat_bbox(minx, miny, maxx, maxy)
        if not is_ll:
            print(f"WARNING: Polygon {idx} bbox does not look like lon/lat -> {msg}")

            if transformer is None:
                print("  Note: No CRS transform was applied.")
            else:
                print("  A transform was applied, but bbox still looks off.")

            if looks_like_projected_bbox(minx, miny, maxx, maxy):
                print("  Hint: Bbox resembles projected coordinates (UTM or State Plane).")

            if SKIP_IF_NOT_LONLAT:
                print("  Skipping this polygon to avoid bad TNM queries.")
                continue
            else:
                print("  Proceeding anyway.")

        query_params = {
            "datasets": DATASET_NAME,
            "outputFormat": "JSON",
            "max": MAX_PER_PAGE,
            "offset": 0,
            "bbox": f"{minx},{miny},{maxx},{maxy}",
        }

        print(f"\nProcessing polygon {idx}")
        print("Querying TNM products (bbox-only)...")

        # Page through TNM results
        all_items = []
        total = None

        while True:
            resp = tnm_query_products(query_params)

            if isinstance(resp, dict):
                items = resp.get("items", [])
                total = resp.get("total", len(items)) if total is None else total
            elif isinstance(resp, list):
                items = resp
                total = len(items) if total is None else total
            else:
                raise ValueError(f"Unexpected TNM response type: {type(resp)}")

            all_items.extend(items)

            if len(items) == 0 or len(all_items) >= total:
                break

            query_params["offset"] += len(items)
            time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"Found {len(all_items)} items (TNM total reported {total})")

        if not all_items:
            print(f"No items returned for polygon {idx}")
            continue

        # Extract and download deliverables
        num_downloads = 0

        for item in all_items:
            urls = extract_download_urls(item)
            if not urls:
                continue

            # Prefer common LiDAR deliverables when available
            preferred = [
                u for u in urls
                if any(ext in u.lower() for ext in (".laz", ".las", ".zip", ".fgdb", "metadata"))
            ]

            targets = preferred or urls

            if DRY_RUN:
                for u in targets:
                    print(f"DRY-RUN: would download {u}")
                continue

            for u in targets:
                if u in SEEN_URLS:
                    continue

                SEEN_URLS.add(u)
                path = download_file(u, out_dir)

                if path:
                    num_downloads += 1
                    print(f"Saved: {path}")

        print(f"Polygon {idx}: downloaded {num_downloads} files")

    print("All done.")


if __name__ == "__main__":
    main()

