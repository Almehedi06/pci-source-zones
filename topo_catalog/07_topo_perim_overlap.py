# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:37:45 2025

@author: lguido

Purpose: Calculates which topographic dataset footprints overlap with each fire perimeter and extracts acquisition dates.
Automation: Automated spatial join and date extraction.
Inputs: Master perimeter shapefile, filtered topographic shapefiles, and CSVs with acquisition dates.
Outputs: Overlap CSV.

"""

import fiona
import pandas as pd
from shapely.geometry import shape

# File paths
perimeter_shp = r"...USGS_Perim_Files\Compiling\fire_master_shape.shp"
footprint_shps = {
    "WESM": "...USGS_Perim_Files/Compiling/TopoShapes/WESM_Clipped_Overlap_Filtered_SinglePolys.shp",
    "OT_3DEP": "...USGS_Perim_Files/Compiling/TopoShapes/OT_3DEP_Clipped_Overlap_Filtered_SinglePolys.shp",
    "OT_NOAA": "...USGS_Perim_Files/Compiling/TopoShapes/OT_NOAA_Clipped_Overlap_Filtered_SinglePolys.shp",
    "OT_PC": "...USGS_Perim_Files/Compiling/TopoShapes/OT_PC_Clipped_Overlap_Filtered_SinglePolys.shp",
    "OT_Raster": "...USGS_Perim_Files/Compiling/TopoShapes/OT_Raster_Clipped_Overlap_Filtered_SinglePolys.shp"
}
footprint_csvs = {
    "WESM": "...USGS_Perim_Files/Compiling/WESM.csv",
    "OT_3DEP": "...USGS_Perim_Files/Compiling/OT_3DEP.csv",
    "OT_NOAA": "...USGS_Perim_Files/Compiling/OT_NOAA.csv",
    "OT_PC": "...USGS_Perim_Files/Compiling/OT_PC.csv",
    "OT_Raster": "...USGS_Perim_Files/Compiling/OT_Rasters.csv"
}
# Matching columns for footprints and CSVs
id_columns = {"WESM": "metadata_l", "OT_3DEP": "name", "OT_NOAA": "name", "OT_PC": "name", "OT_Raster": "name"}

def load_csv_data(csv_path, id_col):
    """Loads CSV data and returns a dictionary mapping ID to (start_date, end_date)."""
    print(f"Loading CSV data from {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    return df.set_index(id_col)[["aq_start", "aq_end"]].to_dict(orient="index")

def find_overlapping_footprints(perimeter_geom, footprint_shp, id_col, csv_data):
    """Finds footprints overlapping with a perimeter and retrieves acquisition dates."""
    print(f"Processing footprint shapefile: {footprint_shp}")
    matched_ids, start_dates, end_dates = [], [], []
    with fiona.open(footprint_shp, "r") as footprint_src:
        for feat in footprint_src:
            footprint_geom = shape(feat["geometry"])
            if perimeter_geom.intersects(footprint_geom):
                footprint_id = feat["properties"].get(id_col)
                if footprint_id and footprint_id in csv_data:
                    matched_ids.append(footprint_id)
                    start_dates.append(csv_data[footprint_id]["aq_start"])
                    end_dates.append(csv_data[footprint_id]["aq_end"])
    print(f"Found {len(matched_ids)} overlapping footprints.")
    return matched_ids, start_dates, end_dates

def main():
    """Main function to process shapefiles and save output CSV."""
    print("Opening perimeter shapefile...")
    with fiona.open(perimeter_shp, "r") as perimeter_src:
        perimeter_data = []
        total_perimeters = len(perimeter_src)
        for i, perim_feat in enumerate(perimeter_src):
            print(f"Processing perimeter {i+1}/{total_perimeters}")
            perim_geom = shape(perim_feat["geometry"])
            row = perim_feat["properties"].copy()
            for key in footprint_shps.keys():
                print(f"Checking for overlaps with {key} data...")
                csv_data = load_csv_data(footprint_csvs[key], id_columns[key])
                ids, starts, ends = find_overlapping_footprints(perim_geom, footprint_shps[key], id_columns[key], csv_data)
                row[f"{key}"] = ids
                row[f"{key}_DATES_START"] = starts
                row[f"{key}_DATES_END"] = ends
            perimeter_data.append(row)
    print("Saving output CSV...")
    output_df = pd.DataFrame(perimeter_data)
    output_df.to_csv("...USGS_Perim_Files/Compiling/Overlap_WithDates.csv", index=False)
    print("Processing complete!")

if __name__ == "__main__":
    main()

