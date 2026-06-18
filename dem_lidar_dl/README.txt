================================================================================
READ ME — Wildfire/HUC Elevation & LiDAR Data Toolkit
================================================================================

Overview
--------
This toolkit automates the discovery, download, and mosaicking of USGS National Map
(“TNM”) 1-meter Digital Elevation Model (DEM) tiles and LiDAR Point Cloud (LPC)
deliverables, with optional fallbacks to RockyWeb (USGS staging) and bulk pulls
from OpenTopography (OT). It also includes a utility to bulk-download USGS hazard
assessment shapefiles indexed by fire.

Typical Use Cases
-----------------
- Build a local repository of USGS wildfire hazard shapefiles for multiple fires. (Hazard_Assessment_Download.py)
- Pull 1 m DEM tiles for a fire perimeter or a set of HUC12 watersheds. (TNM_DEM_Pull_by_Fire.py)(TNM_DEM_Pull_by_HUC.py)
- Pull LiDAR project deliverables (e.g., LAZ/LAS, ZIPs, FGDB) for an AOI. (TNM_lidar_pull.py)
- Mosaic downloaded DEM tiles into compressed GeoTIFF(s) for analysis. (TNM_DEM_Mosaic.py)(OT_DEM_Mosaic.py)
- Fallback to RockyWeb when TNM is unavailable (less efficient). (RockyWeb_lidar_pull.py)
- Bulk-download a known OpenTopography collection (by dataset prefix). (OT_Bulk_Known_Collection.py)

--------------------------------------------------------------------------------
Repository Contents (Scripts)
--------------------------------------------------------------------------------
1) Hazard_Assessment_Download.py
   Bulk-scrapes the USGS landslides-realtime fire parent directory; for each fire
   folder, downloads "Shapefiles.zip" with resume/retry and a progress bar.

2) TNM_DEM_Pull_by_Fire.py
   Queries TNM for "Digital Elevation Model (DEM) 1 meter" GeoTIFF tiles using a
   fire perimeter shapefile (polygon or bounding box), paginates results, caches
   files globally to avoid duplicates, and organizes outputs by shapefile name.

3) TNM_DEM_Pull_by_HUC.py
   Same DEM workflow as above but per HUC12 polygon; creates one output folder
   per HUC12 and uses the same pagination and caching behavior.

4) TNM_lidar_pull.py
   Queries TNM for "Lidar Point Cloud (LPC)" products using AOI polygons; performs
   bbox sanity checks for lon/lat, extracts diverse download links (LAZ/LAS/ZIP/FGDB),
   and streams downloads with a global cache + hard-link/copy into AOI folders.

5) RockyWeb_lidar_pull.py
   Fallback LiDAR downloader that lists LAZ files from a known RockyWeb project
   path, then filters tiles by AOI overlap (via PDAL "info" bounding boxes) before
   download. Use only when TNM is down; slower and requires known paths/CRS.

6) TNM_DEM_Mosaic.py
   Builds mosaics from DEM tiles (downloaded via TNM workflows). Grouping is based
   on dataset keys parsed from filenames, harmonizes majority CRS, reprojects
   minority tiles, builds in-memory VRT, writes compressed tiled GeoTIFF, and (optionally)
   builds overviews.

7) OT_DEM_Mosaic.py
   Similar mosaicking but targeted for DEM tiles pulled from OpenTopography. Finds
   .tif recursively, chooses majority CRS, reprojects minority tiles, builds VRT,
   and writes compressed tiled GeoTIFF (+ overviews optional).

8) OT_Bulk_Known_Collection.py
   Lists and downloads all objects beneath a known OpenTopography S3 prefix (e.g.,
   shortname) from either "pc-bulk" (point cloud) or "raster" buckets, preserving
   hierarchy, skipping already-downloaded files.

--------------------------------------------------------------------------------
System Requirements & Recommended Environment
--------------------------------------------------------------------------------
- Python ≥ 3.9 (tested with 3.10/3.11 recommended).
- Core libraries (pip/conda): requests, tqdm, bs4 (BeautifulSoup), pyshp (shapefile),
  shapely, pyproj, fiona (for RockyWeb AOI reading), GDAL (for mosaics), PDAL (for
  RockyWeb tile bbox). 
- GDAL/PDAL are easiest via conda-forge; ensure "gdal" and "pdal" command-line tools
  are available on PATH before running mosaic/fallback scripts. 

Tip: Avoid geopandas at all costs

--------------------------------------------------------------------------------
Configuration Keys (per script)
--------------------------------------------------------------------------------
Hazard_Assessment_Download.py
- PARENT_URL: USGS parent directory listing for fires.
- OUTPUT_DIR: local folder for zipped shapefiles.
- MAX_RETRIES, RETRY_SLEEP: network resilience parameters.

TNM_DEM_Pull_by_Fire.py
- SHAPEFILE_PATH: path to fire perimeter shapefile (.shp).
- OUTPUT_DIR: root output for downloads (also hosts "_cache").
- DATASET_NAME: "Digital Elevation Model (DEM) 1 meter".
- PROD_FORMATS: "GeoTIFF".
- USE_POLYGON_WHEN_POSSIBLE: send polygon WKT if feasible; else bbox.
- SIMPLIFY_*: geometry simplification thresholds (reduce URL size).
- DRY_RUN: print URLs only, do not download.

TNM_DEM_Pull_by_HUC.py
- SHAPEFILE_PATH: path to HUC12 polygons.
- OUTPUT_DIR, DATASET_NAME, PROD_FORMATS, USE_POLYGON_WHEN_POSSIBLE, SIMPLIFY_*, DRY_RUN:
  same semantics as fire-based script.
- HUC field guesses: tries typical field names for folder naming.

TNM_lidar_pull.py
- SHAPEFILE_PATH: AOI polygons shapefile.
- OUTPUT_DIR: root for LPC downloads (also hosts "_cache").
- DATASET_NAME: "Lidar Point Cloud (LPC)".
- SKIP_IF_NOT_LONLAT: skip projected bbox mistakes safely.
- DRY_RUN: list target URLs, no download.

RockyWeb_lidar_pull.py (fallback)
- SHAPEFILE_PATH: AOI shapefile (reprojected to match LAZ CRS in code - be careful may need to update).
- BASE_URL: RockyWeb LAZ directory for a specific project.
- OUTPUT_DIR: destination for selected LAZ tiles.

TNM_DEM_Mosaic.py
- ROOT_DIR: folder with DEM tiles (grouped by dataset naming).
- OUTPUT_DIR: destination for mosaic GeoTIFFs.

OT_DEM_Mosaic.py
- ROOT_DIR: folder containing OT DEM tiles (searched recursively).
- OUTPUT_TIF: final mosaic file path (single output).

OT_Bulk_Known_Collection.py
- ENDPOINT: OpenTopography S3 endpoint.
- BUCKET: "pc-bulk" or "raster".
- PREFIX: dataset shortname/prefix (e.g., "CA25_Lamb/").
- OUTROOT: local root to mirror remote hierarchy.

--------------------------------------------------------------------------------
Proposed End-to-End Workflow
--------------------------------------------------------------------------------
(1) Prepare AOI data
    - If needed, fetch USGS hazard assessment shapefiles for multiple fires:
      Run Hazard_Assessment_Download.py to build a local archive.
    - Create/collect the fire perimeter shapefile or HUC12 shapefile(s).
    - Ensure the AOI shapefile has a valid .prj; scripts will transform to WGS84
      (EPSG:4326) for TNM queries or warn if coordinates look projected.

(2) Acquire elevation (DEM) data
    - For a single fire perimeter: run TNM_DEM_Pull_by_Fire.py.
    - For watershed sets (HUC12): run TNM_DEM_Pull_by_HUC.py.
    - Both scripts paginate TNM results, use polygon WKT if feasible (fallback to bbox),
      and write to organized folders with a global "_cache" to avoid duplicates.

(3) Acquire LiDAR (LPC) data
    - Preferred path: run TNM_lidar_pull.py over your AOI shapefile; it extracts
      common deliverables (LAZ/LAS/ZIP/FGDB) when available.
    - If TNM is down or incomplete, use RockyWeb_lidar_pull.py with a known project
      BASE_URL; it filters LAZ tiles by AOI overlap via PDAL "info".

(4) (Optional) Acquire DEMs from OpenTopography
    - When you know a dataset shortname/prefix: run OT_Bulk_Known_Collection.py to
      mirror the collection locally (raster or point cloud buckets).

(5) Mosaic DEM tiles
    - TNM-sourced tiles: run TNM_DEM_Mosaic.py to build dataset-specific mosaics
      (majority CRS harmonization + VRT → compressed tiled GeoTIFF + overviews).
    - OpenTopography tiles: run OT_DEM_Mosaic.py for a single output mosaic with
      the same harmonization approach.

(6) QA/QC & Organization
    - Visually verify mosaics, check CRS, nodata handling, and overviews.
    - Keep the "_cache" folder; it significantly reduces re-downloads across runs.

ASCII Workflow Sketch
---------------------
[AOI prep] ──> [TNM DEM Pull (fire/HUC)] ──┐
                                           ├─> [DEM Mosaic (TNM)]
[TNM LiDAR Pull] ──────────────────────────┘
[RockyWeb LiDAR Fallback] (only if needed)
[OT Bulk (known collection)] ──────────────> [DEM Mosaic (OT)]

--------------------------------------------------------------------------------
Quick-Start Examples
--------------------------------------------------------------------------------
# 1) DEM for a fire perimeter
- Edit TNM_DEM_Pull_by_Fire.py: set SHAPEFILE_PATH and OUTPUT_DIR.
- Run: `python TNM_DEM_Pull_by_Fire.py`
- Mosaic: set ROOT_DIR to the DEM folder and run `python TNM_DEM_Mosaic.py`.

# 2) DEM for HUC12s
- Edit TNM_DEM_Pull_by_HUC.py: set SHAPEFILE_PATH and OUTPUT_DIR. .py)
- Run: `python TNM_DEM_Pull_by_HUC.py`
- Mosaic per dataset: `python TNM_DEM_Mosaic.py` (root points at your DEM folder).

# 3) LiDAR deliverables for an AOI
- Edit TNM_lidar_pull.py: set SHAPEFILE_PATH and OUTPUT_DIR.
- Run: `python TNM_lidar_pull.py`
- If TNM is unavailable, configure RockyWeb_lidar_pull.py (BASE_URL) and run it.

# 4) OpenTopography bulk pull (known dataset)
- Edit OT_Bulk_Known_Collection.py: set BUCKET ("pc-bulk" or "raster"), PREFIX, OUTROOT.
- Run: `python OT_Bulk_Known_Collection.py`
- Mosaic: `python OT_DEM_Mosaic.py` with ROOT_DIR and OUTPUT_TIF set.

--------------------------------------------------------------------------------
Best Practices & Tips
--------------------------------------------------------------------------------
- CRS & .prj files: keep .prj with your shapefile; scripts transform to WGS84 for TNM
  queries and warn on suspicious bboxes.
- DRY_RUN first: set DRY_RUN=True to list URLs before downloading large datasets.
- Geometry simplification: large polygons can cause long URLs; tune SIMPLIFY_* to
  reduce WKT size and prefer POST automatically where implemented.
- Caching: the "_cache" folder is shared across runs/polygons to prevent redundant
  downloads. Do not delete it unless you want to re-download.
- Mosaics: the scripts pick a majority CRS and reproject minority tiles; if all tiles
  are consistent, reprojection is skipped and performance improves.
- Overviews: building overviews may fail in some environments; mosaics still work
  without them.
- RockyWeb fallback: requires a known project BASE_URL and correct AOI CRS (update
  EPSG in code). Expect slower fetching and more manual setup.

--------------------------------------------------------------------------------
Troubleshooting
--------------------------------------------------------------------------------
- "No items returned" from TNM:
  Check CRS: ensure AOI transformed to WGS84; verify bbox ranges look like lon/lat.
- Very large polygons / long URLs:
  Enable polygon simplification or rely on POST in the DEM-by-fire script.
- GDAL/PDAL not found:
  Install via conda-forge and confirm `gdalinfo` / `pdal info` run in your shell.
- Mosaic CRS conflicts:
  Scripts auto-harmonize to majority CRS; double-check inputs and let them reproject
  minority tiles.
- Interrupted downloads:
  Resume logic exists in Hazard_Assessment_Download; for TNM/OT pulls, re-run scripts—
  caching and skip logic prevent redundant work.

--------------------------------------------------------------------------------
Folder Organization (suggested)
--------------------------------------------------------------------------------
data/
  hazards/                  # Hazard shapefiles (from Shapefiles.zip)
  dem/                      # DEM tiles and mosaics
    _cache/                 # Shared TNM cache (do not delete)
  lidar/                    # LiDAR downloads (TNM or RockyWeb)
    _cache/                 # Shared TNM cache for LPC)
  opentopo/                 # OT bulk pulls (mirrored hierarchy)

--------------------------------------------------------------------------------
Attribution & Notes
--------------------------------------------------------------------------------
- Data sources: USGS National Map (TNM), USGS RockyWeb (LPC staging), OpenTopography.
- Tools: GDAL, PDAL, requests/tqdm/bs4, pyshp, shapely, pyproj, fiona.
- Respect source terms of use and cite USGS/OT in publications and deliverables.

--------------------------------------------------------------------------------
Changelog (high-level)
--------------------------------------------------------------------------------
- 2026-01: Initial set of TNM pull scripts for DEM/LPC; OT bulk and mosaics; RockyWeb
  fallback; hazard shapefile utility.

- CURRENT WORK
	-Dictionary creation for individual fire query by name for single-fire downloads
	-Automatic definition of HUC12s based on fire for seemless fire -> watershed workflow
	-Clean packaging for smooth command-line interface
	-Improved CRS handling from RockyWeb so it can be an auto-fallback during production
================================================================================
End of README
