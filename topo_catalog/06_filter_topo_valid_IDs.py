# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:45:10 2025

@author: lguido


Purpose: Filters topographic footprint shapefiles to retain only features with valid IDs as listed in corresponding CSVs.
Automation: Fully automated if CSVs and shapefiles are correct.
Inputs: Topo CSVs and corresponding shapefiles.
Outputs: Filtered shapefiles.

"""

import fiona
import csv

# Define file paths for CSVs and shapefiles
file_pairs = [
    ("...USGS_Perim_Files/Compiling/WESM.csv",
     "...USGS_Perim_Files/Compiling/TopoShapes/WESM_Clipped_Overlap.shp",
     "metadata_l"),

    ("...USGS_Perim_Files/Compiling/FESM.csv",
     "...USGS_Perim_Files/Compiling/TopoShapes/FESM_Clipped_Overlap.shp",
     "project_id"),

    ("...USGS_Perim_Files/Compiling/OT_3DEP.csv",
     "...USGS_Perim_Files/Compiling/TopoShapes/OT_3DEP_Clipped_Overlap.shp",
     "name"),

    ("...USGS_Perim_Files/Compiling/OT_NOAA.csv",
     "...USGS_Perim_Files/Compiling/TopoShapes/OT_NOAA_Clipped_Overlap.shp",
     "name"),

    ("...USGS_Perim_Files/Compiling/OT_PC.csv",
     "...USGS_Perim_Files/Compiling/TopoShapes/OT_PC_Clipped_Overlap.shp",
     "name"),

    ("...USGS_Perim_Files/Compiling/OT_Rasters.csv",
     "...USGS_Perim_Files/Compiling/TopoShapes/OT_Raster_Clipped_Overlap.shp",
     "name"),
]

def get_valid_entries(csv_path, key_column):
    """Read the CSV and extract valid entries based on the specified key column."""
    valid_entries = set()
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row[key_column]:  # Ensure the key is not empty
                valid_entries.add(row[key_column])
    return valid_entries

def filter_shapefile(shp_path, valid_entries, key_column):
    """Filter the shapefile to retain only features that match the valid CSV entries."""
    output_shp_path = shp_path.replace(".shp", "_Filtered.shp")

    with fiona.open(shp_path, "r") as src:
        meta = src.meta  # Preserve metadata (schema, CRS, driver)
        
        with fiona.open(output_shp_path, "w", **meta) as dst:
            for feature in src:
                if feature["properties"].get(key_column) in valid_entries:
                    dst.write(feature)

    print(f"Filtered shapefile saved: {output_shp_path}")

# Process each CSV and shapefile pair
for csv_path, shp_path, key_column in file_pairs:
    valid_entries = get_valid_entries(csv_path, key_column)
    filter_shapefile(shp_path, valid_entries, key_column)
