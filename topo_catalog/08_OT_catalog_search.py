# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:14:00 2025

@author: lguido

Purpose: Queries the OpenTopography catalog API for each perimeter polygon, saving a list of discovered datasets per polygon.
Automation: Automated querying and result saving.
Inputs: WKT CSV of perimeters.
Outputs: JSON file of catalog query results.

"""

import pandas as pd
import requests
import json
import time
from tqdm import tqdm  # For progress tracking
import re

# File paths
csv_path = r"...USGS_Perim_Files\merged_wkt.csv"
output_json = r"...USGS_Perim_Files\otcatalog_results.json"

# Read CSV file
df = pd.read_csv(csv_path)

# Ensure 'WKT' column exists
if "WKT" not in df.columns:
    raise ValueError("CSV does not contain a 'WKT' column.")

# Prepare results storage
results = []

# Function to clean WKT data
def transform_wkt(wkt):
    # Remove the "POLYGON Z" part and the parentheses, and split by commas
    wkt_clean = re.sub(r"POLYGON Z \(\(", "", wkt)  # Remove "POLYGON Z (("
    wkt_clean = re.sub(r"\)\)$", "", wkt_clean)     # Remove the closing parentheses at the end
    
    # Remove the Z (height) values (i.e., values after each coordinate pair) and spaces
    wkt_clean = re.sub(r"\s0,", "", wkt_clean)       # Remove all instances of " 0" (height info)
    wkt_clean = re.sub(r"\s+", ",", wkt_clean)        # Remove all spaces
    
    # Group every two numbers (X, Y) and join them with commas, as requested
    transformed = wkt_clean

    return transformed

# Function to split the WKT into manageable chunks for large polygons
def split_wkt(wkt, max_length=2048):
    coords = wkt.split(",")
    chunks = []
    chunk = ""
    for i in range(0, len(coords), 2):
        pair = f"{coords[i]},{coords[i+1]}"
        if len(chunk + pair) > max_length:
            chunks.append(chunk)
            chunk = pair  # Start a new chunk
        else:
            chunk += "," + pair if chunk else pair  # Add the first pair
    if chunk:
        chunks.append(chunk)
    return chunks

# Loop through each WKT polygon with progress tracking
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing WKT polygons"):
    original_wkt = row["WKT"]
    
    # Transform WKT format
    transformed_wkt = transform_wkt(original_wkt)

    # Split the WKT if necessary to avoid exceeding URL length
    wkt_chunks = transformed_wkt

    # API request
    url = "https://portal.opentopography.org/API/otCatalog"
    datasets_all = []
    
    for chunk in wkt_chunks:
        params = {"wkt": chunk, "datasetType": "DEM"}  # Change datasetType if needed

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()  # Raise error if request fails
            data = response.json()

            # Save only relevant dataset info
            datasets = data.get("datasets", [])
            datasets_all.extend(datasets)

        except requests.exceptions.RequestException as e:
            print(f"Error on index {index}: {e}")

        # Small delay to avoid overwhelming API (optional)
        time.sleep(1)

    # Store the result for this polygon
    results.append({
        "index": index,
        "original_wkt": original_wkt,
        "transformed_wkt": transformed_wkt,
        "datasets": datasets_all
    })

# Save results as JSON
with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"Processing complete. Results saved to {output_json}")
