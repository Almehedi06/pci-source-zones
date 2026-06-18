# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:35:18 2026

@author: lguido
"""

import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# ==== SETTINGS ====

# Base URL for the OpenTopography S3 endpoint
ENDPOINT = "https://opentopography.s3.sdsc.edu"

# Which bucket to access. Use "pc-bulk" for point clouds, "raster" for raster datasets
BUCKET = "pc-bulk"

# Dataset prefix (shortname) for the specific dataset you want
PREFIX = "CA25_Lamb/"

# Root directory to save all downloaded files
OUTROOT = Path("E:/OpenTopography/output")  # Change this to your preferred path
OUTROOT.mkdir(parents=True, exist_ok=True)  # Create directory if it does not exist

# Create a session for persistent connections
session = requests.Session()

# ==== FUNCTIONS ====

def list_objects(bucket, prefix):
    """
    List all objects under a given prefix using S3 ListObjectsV2.
    
    Handles pagination automatically using the continuation token.
    Returns a list of tuples: (object key, size in bytes).
    
    Warning: Some OpenTopography datasets are huge. Proceed with caution if you like your free time.
    """
    objects = []
    continuation = None

    while True:
        params = {
            "list-type": "2",  # ListObjectsV2
            "prefix": prefix,  # Only list objects starting with this prefix
        }
        if continuation:
            params["continuation-token"] = continuation

        # Make the GET request to list objects
        r = session.get(
            f"{ENDPOINT}/{bucket}",
            params=params,
            timeout=60,
        )
        r.raise_for_status()  # Stop immediately if something goes wrong

        # Parse XML response
        root = ET.fromstring(r.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}  # Namespace for S3 XML

        # Extract object keys and sizes
        for contents in root.findall("s3:Contents", ns):
            key = contents.find("s3:Key", ns).text
            size = int(contents.find("s3:Size", ns).text)
            objects.append((key, size))

        # Check if there are more results (pagination)
        is_truncated = root.find("s3:IsTruncated", ns).text == "true"
        if not is_truncated:
            break  # No more pages

        # Get continuation token for next page
        continuation = root.find("s3:NextContinuationToken", ns).text

    return objects


# ==== MAIN ====

# List all files under the dataset prefix
print("Listing objects...")
objects = list_objects(BUCKET, PREFIX)
print(f"Found {len(objects)} objects")  # Could be thousands — keep coffee handy

# Loop through each object and download if needed
for key, size in objects:
    # Relative path inside dataset
    relpath = key.replace(PREFIX, "")
    if not relpath:
        continue  # Skip the "directory" placeholder

    # Full path to save locally
    outpath = OUTROOT / relpath
    outpath.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist

    # Skip if file already exists with correct size
    if outpath.exists() and outpath.stat().st_size == size:
        print(f"Skipping {relpath} (already exists)")
        continue

    # Download URL
    url = f"{ENDPOINT}/{BUCKET}/{key}"
    print(f"Downloading {relpath}")

    # Stream download with progress bar
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(outpath, "wb") as f, tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=relpath,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

print("\nDownload complete")
