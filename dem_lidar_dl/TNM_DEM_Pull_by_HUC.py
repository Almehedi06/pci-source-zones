# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:49:56 2026

@author: lguido
"""

import os
import json
import time
import math
import pathlib
import hashlib
from typing import Dict, List, Tuple, Optional

import shapefile  # pyshp: read shapefiles
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely import wkt as shapely_wkt
from pyproj import CRS, Transformer
import requests
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

# Path to input shapefile (watersheds / HUC12s)
SHAPEFILE_PATH = r"C:\path to your watersheds of interest\HUCs.shp"

# Root directory where downloaded DEMs will be stored
OUTPUT_DIR = r"D:\DEMs"

# National Map base URL
TNM_BASE = "https://tnmaccess.nationalmap.gov/api/v1"
PRODUCTS_URL = f"{TNM_BASE}/products"

# Focus on 1-meter DEM GeoTIFF products
DATASET_NAME = "Digital Elevation Model (DEM) 1 meter"
PROD_FORMATS = "GeoTIFF"

# Request and paging settings
REQUEST_TIMEOUT = 60       # seconds per HTTP request
MAX_PER_PAGE = 1000        # number of items per page from TNM API
SLEEP_BETWEEN_CALLS = 0.5  # polite rate limit

# Choose whether to send full polygon WKT or just bounding box for query
USE_POLYGON_WHEN_POSSIBLE = True

# Simplify very dense polygons to reduce WKT string length
SIMPLIFY_DENSE_POLYGONS = True
SIMPLIFY_MAX_COORDS = 2000
SIMPLIFY_TOLERANCE_DEG = 0.0002  # ~20 meters

# Threshold for GET URL length; if exceeded, use POST instead
LONG_URL_THRESHOLD = 1900

# If True, just print URLs without downloading
DRY_RUN = False

# Global cache folder: one copy of each file to avoid redundant downloads
GLOBAL_CACHE_DIR = os.path.join(OUTPUT_DIR, "_cache")

# Memory for URLs seen in this run to avoid duplicates
SEEN_URLS: set = set()


# ----------------------------
# Helper Functions
# ----------------------------

def read_prj_crs(prj_path: str) -> Optional[CRS]:
    """
    Read the .prj file associated with a shapefile to determine its CRS.
    Returns None if no .prj exists or parsing fails.
    """
    if not os.path.exists(prj_path):
        return None
    txt = pathlib.Path(prj_path).read_text(encoding="utf-8", errors="ignore")
    try:
        return CRS.from_wkt(txt)
    except Exception:
        return None


def ensure_wgs84_transform(crs: Optional[CRS]) -> Optional[Transformer]:
    """
    Create a transformer from input CRS to WGS84 (EPSG:4326).
    Returns None if input CRS is already WGS84 or missing.
    """
    if crs is None:
        return None  # assume already WGS84
    wgs84 = CRS.from_epsg(4326)
    if crs.to_epsg() == 4326:
        return None
    return Transformer.from_crs(crs, wgs84, always_xy=True)


def transform_coords(transformer: Optional[Transformer], coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Transform a list of (x, y) coordinates using the given transformer."""
    if transformer is None:
        return coords
    return [transformer.transform(x, y) for x, y in coords]


def shape_to_polygons(shape) -> List[List[Tuple[float, float]]]:
    """
    Convert a pyshp shape (Polygon/MultiPolygon) to a list of rings.
    Each ring is a list of coordinates. Handles multipart shapes.
    """
    pts = shape.points
    parts = list(shape.parts) + [len(pts)]
    rings = []
    for i in range(len(parts) - 1):
        start, end = parts[i], parts[i+1]
        rings.append(pts[start:end])
    return rings


def build_geometry_from_shape(shape, transformer: Optional[Transformer]):
    """
    Convert a pyshp shape to a shapely Polygon or MultiPolygon.
    Each part becomes a separate polygon shell to avoid misassigning holes.
    """
    rings = shape_to_polygons(shape)
    rings_ll = [transform_coords(transformer, r) for r in rings]

    if len(rings_ll) == 1:
        poly = Polygon(rings_ll[0])
        return orient(poly, sign=1.0)  # enforce CCW orientation

    polys = [orient(Polygon(r), sign=1.0) for r in rings_ll if Polygon(r).is_valid and not Polygon(r).is_empty]
    if not polys:
        return None
    return MultiPolygon(polys)


def geom_to_bbox(geom) -> Tuple[float, float, float, float]:
    """Return bounding box (minx, miny, maxx, maxy) for a shapely geometry."""
    return geom.bounds


def estimate_url_length(base_url: str, params: Dict) -> int:
    """Estimate length of a GET URL to decide GET vs POST."""
    from urllib.parse import urlencode
    return len(base_url) + 1 + len(urlencode(params))


def tnm_query_products(params: Dict, prefer_post: bool = False) -> Dict:
    """
    Query TNM /products endpoint, handling GET vs POST depending on URL length.
    Retries a few times on failure or if GET is too long (414).
    """
    tries = 0
    url_too_long = estimate_url_length(PRODUCTS_URL, params) > LONG_URL_THRESHOLD
    methods = ["POST", "GET"] if (prefer_post or url_too_long) else ["GET", "POST"]

    while True:
        tries += 1
        last_exc = None
        for method in methods:
            try:
                if method == "POST":
                    r = requests.post(PRODUCTS_URL, json=params, timeout=REQUEST_TIMEOUT)
                else:
                    r = requests.get(PRODUCTS_URL, params=params, timeout=REQUEST_TIMEOUT)

                if r.status_code == 414 and method == "GET":
                    continue  # try POST next

                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_exc = e
                time.sleep(SLEEP_BETWEEN_CALLS)

        if last_exc and tries < 3:
            time.sleep(1.5 * tries)
            continue
        if last_exc:
            raise last_exc


def extract_download_urls(item: Dict) -> List[str]:
    """
    Extract direct download URLs from a TNM/ScienceBase item.
    Handles multiple possible fields and de-duplicates results.
    """
    urls = []

    if "downloadURL" in item and item["downloadURL"]:
        urls.append(item["downloadURL"])

    for key in ("webLinks", "distributionLinks", "files"):
        if key in item and isinstance(item[key], list):
            for link in item[key]:
                url = link.get("url") or link.get("href") or link.get("linkUrl")
                if not url:
                    continue
                label = (link.get("type") or link.get("rel") or "").lower()
                if "download" in label or "http" in url:
                    urls.append(url)

    return list(dict.fromkeys(urls))  # de-duplicate


def safe_filename_from_url(url: str) -> str:
    """Create a safe filename from URL; fallback if URL path empty."""
    name = url.split("?")[0].split("/")[-1].strip()
    return name or f"download_{abs(hash(url))}.dat"


def cache_file_path(url: str) -> str:
    """
    Return a deterministic cache path for a URL.
    Uses SHA1 hash to avoid collisions.
    """
    base = safe_filename_from_url(url)
    suffix = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    name, ext = os.path.splitext(base)
    return os.path.join(GLOBAL_CACHE_DIR, f"{name}.{suffix}{ext}")


def ensure_link_or_copy(src: str, dst_dir: str, desired_name: Optional[str] = None) -> str:
    """
    Link or copy a file from src into dst_dir. Returns final path.
    Attempts hardlink first, then copy if needed.
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
    Download a file to cache, then link/copy to output folder.
    Uses global cache to prevent multiple downloads of same URL.
    """
    base_name = safe_filename_from_url(url)
    cached = cache_file_path(url)

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
                unit="B", unit_scale=True,
                desc=base_name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        os.replace(tmp_path, cached)
        return ensure_link_or_copy(cached, out_dir, desired_name=base_name)
    except requests.RequestException:
        if os.path.exists(f"{cached}.part"):
            os.remove(f"{cached}.part")
        return None


def simplify_if_huge(geom, max_coords=SIMPLIFY_MAX_COORDS, tolerance_deg=SIMPLIFY_TOLERANCE_DEG):
    """Simplify geometry if it has too many coordinates, to reduce query size."""
    try:
        if geom.geom_type == "Polygon":
            if len(geom.exterior.coords) > max_coords:
                return geom.simplify(tolerance_deg, preserve_topology=True)
        elif geom.geom_type == "MultiPolygon":
            total = sum(len(p.exterior.coords) for p in geom.geoms)
            if total > max_coords:
                return geom.simplify(tolerance_deg, preserve_topology=True)
    except Exception:
        pass
    return geom


def safe_folder_name(s: str) -> str:
    """Sanitize folder names to include only safe characters."""
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in ("_", "-", "."))


# ----------------------------
# Main workflow
# ----------------------------

def main():
    """Query TNM DEM products for each HUC12 in shapefile and download files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GLOBAL_CACHE_DIR, exist_ok=True)

    # Read shapefile + CRS
    reader = shapefile.Reader(SHAPEFILE_PATH)
    prj = os.path.splitext(SHAPEFILE_PATH)[0] + ".prj"
    crs = read_prj_crs(prj)
    transformer = ensure_wgs84_transform(crs)

    # Attribute field names
    fields = [f[0] for f in reader.fields[1:]]

    # Guess HUC12 field name
    huc_field_candidates = ["HUC12", "HUC_12", "huc12", "HUC", "HUC12_TXT", "HUC12_CODE"]

    def get_huc12(record):
        for nm in huc_field_candidates:
            if nm in fields:
                val = record[fields.index(nm)]
                if val is not None:
                    return str(val)
        return str(record[0])

    print(f"Found {reader.numRecords} features in shapefile")

    for sr in reader.shapeRecords():
        shape = sr.shape
        record = sr.record
        huc12 = get_huc12(record)

        geom = build_geometry_from_shape(shape, transformer)
        if geom is None:
            print(f"Skipping {huc12}: invalid geometry")
            continue

        if SIMPLIFY_DENSE_POLYGONS:
            geom = simplify_if_huge(geom)

        # Prepare query parameters
        query_params = {
            "datasets": DATASET_NAME,
            "prodFormats": PROD_FORMATS,
            "outputFormat": "JSON",
            "max": MAX_PER_PAGE,
            "offset": 0
        }

        use_polygon = USE_POLYGON_WHEN_POSSIBLE and isinstance(geom, Polygon)
        prefer_post = False

        if use_polygon:
            query_params["polygon"] = geom.wkt
            prefer_post = True
        else:
            minx, miny, maxx, maxy = geom_to_bbox(geom)
            query_params["bbox"] = f"{minx},{miny},{maxx},{maxy}"

        print(f"\nHUC12 {huc12}")
        print("Querying TNM products...")

        # Page through results
        all_items = []
        while True:
            try:
                resp = tnm_query_products(query_params, prefer_post=prefer_post)
            except requests.RequestException:
                if "polygon" in query_params:
                    minx, miny, maxx, maxy = geom_to_bbox(geom)
                    query_params.pop("polygon", None)
                    query_params["bbox"] = f"{minx},{miny},{maxx},{maxy}"
                    prefer_post = False
                    resp = tnm_query_products(query_params, prefer_post=prefer_post)
                else:
                    raise

            items = resp.get("items", [])
            total = resp.get("total", 0)
            all_items.extend(items)

            if len(all_items) >= total or not items:
                break

            query_params["offset"] += len(items)

        print(f"Found {len(all_items)} items (TNM total reported {total})")

        urls = []
        for it in all_items:
            urls.extend(extract_download_urls(it))
        urls = list(dict.fromkeys(urls))

        if not urls:
            print("No download URLs parsed for this HUC12")
            continue

        out_dir = os.path.join(OUTPUT_DIR, f"HUC12_{safe_folder_name(huc12)}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Preparing to download {len(urls)} files")
        if DRY_RUN:
            for u in urls:
                print(u)
            continue

        for u in urls:
            p = download_file(u, out_dir)
            if p is None:
                print(f"Failed to download: {u}")
            else:
                print(f"Saved: {p}")

    print("All done.")


if __name__ == "__main__":
    main()
