# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:26:48 2026

@author: lguido
"""

import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

# Maximum number of retry attempts for a single fire folder download
MAX_RETRIES = 5

# Number of seconds to wait between retry attempts
RETRY_SLEEP = 5

# ==== SETTINGS ====

# Parent directory containing subfolders for individual fires
# Each fire folder may (or may not) contain a Shapefiles.zip archive
PARENT_URL = "https://landslides.usgs.gov/static/landslides-realtime/fires/"

# Local directory where all downloaded shapefiles will be stored
# Using an external drive here to keep internal storage happy
OUTPUT_DIR = r"D:\DebrisFlowHazards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== FUNCTIONS ====

def list_fire_folders(parent_url):
    """
    Scrape the parent directory page and return a list of URLs
    corresponding to individual fire folders.

    This function assumes a simple directory-style HTML listing,
    which is true for the USGS hazard assessment pages.
    """
    resp = requests.get(parent_url)
    resp.raise_for_status()

    # Parse the HTML to extract links
    soup = BeautifulSoup(resp.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]

    # Keep only directory-style links (end with '/')
    # and ignore the parent directory reference ('../')
    fire_folders = [parent_url + l for l in links if l.endswith('/') and l != '../']

    return fire_folders


def download_shapefiles(fire_url, output_dir):
    """
    Download the Shapefiles.zip archive from a given fire folder,
    if it exists.

    Features:
    - Checks whether the file exists on the server before downloading
    - Supports resuming partially completed downloads
    - Retries on common network failures (because the internet is fickle)
    """
    shapefile_url = fire_url + "Shapefiles.zip"

    # Construct a local filename using the fire folder name
    filename = os.path.join(
        output_dir,
        os.path.basename(fire_url.strip('/')) + "_Shapefiles.zip"
    )

    # Use a HEAD request to confirm the file exists and get its size
    r = requests.head(shapefile_url, timeout=30)
    if r.status_code != 200:
        print(f"No shapefile found for {fire_url}")
        return

    # Total file size in bytes (used for progress bar and resume logic)
    total_size = int(r.headers.get('content-length', 0))

    # Attempt the download up to MAX_RETRIES times
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Check if a partial file already exists locally
            existing_size = os.path.getsize(filename) if os.path.exists(filename) else 0

            # If the file is already fully downloaded, skip it
            if existing_size >= total_size > 0:
                print(f"File already downloaded: {filename}")
                return

            # If resuming, request only the remaining bytes
            headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

            # Stream the download so we do not load the entire file into memory
            with requests.get(
                shapefile_url,
                stream=True,
                headers=headers,
                timeout=30
            ) as r_get:
                r_get.raise_for_status()

                # Append if resuming, otherwise start a new file
                mode = 'ab' if existing_size > 0 else 'wb'

                # tqdm provides a progress bar that plays nicely with large files
                with open(filename, mode) as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=os.path.basename(filename),
                    initial=existing_size,
                    ascii=True
                ) as pbar:
                    for chunk in r_get.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # If we reach this point, the download succeeded
            # Exit the retry loop
            return

        except (ChunkedEncodingError, ConnectionError, Timeout):
            print(
                f"\n Download interrupted for {os.path.basename(filename)} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            # If this was the last attempt, raise the error
            if attempt == MAX_RETRIES:
                print("Max retries reached — failing.")
                raise
            else:
                # Brief pause before trying again
                time.sleep(RETRY_SLEEP)

# ==== MAIN ====

# Retrieve all fire folder URLs from the parent directory
fire_folders = list_fire_folders(PARENT_URL)
print(f"Found {len(fire_folders)} fire folders.")

# Loop through each fire and attempt to download its shapefile archive
for fire in fire_folders:
    download_shapefiles(fire, OUTPUT_DIR)

print("All downloads complete")

