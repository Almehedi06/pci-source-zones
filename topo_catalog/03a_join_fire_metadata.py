# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:12:57 2025

@author: lguido

Purpose: Joins fire event metadata from a CSV to perimeter shapefiles, creating a master shapefile with attributes for each fire.
Automation: Automated if input files are correct; manual review recommended for metadata mismatches.
Inputs: Metadata CSV and perimeter shapefile.
Outputs: Enriched shapefile.

"""
import fiona
import pandas as pd
import shapefile  # pyshp
from shapely.geometry import shape, mapping

# File paths
shapefile_path = r"...USGS_Perim_Files\Compiling\FireShapes\merged_fresh.shp"
csv_path = r"...USGS_Perim_Files\Compiling\fire_master_list.csv"
output_shapefile_path = r"...USGS_Perim_Files\Compiling\fire_master_shape_whole.shp"

# Read CSV file
fire_data = pd.read_csv(csv_path)

# Ensure MERGE_SRC is properly formatted
fire_data["MERGE_SRC"] = fire_data["MERGE_SRC"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Explode list-type MERGE_SRC entries into individual rows
fire_data_exploded = fire_data.explode("MERGE_SRC")

# Read original shapefile
with fiona.open(shapefile_path, "r") as src:
    schema = {"geometry": src.schema["geometry"],
              "properties": {"MERGE_SRC": "str", "Fire_ID": "str", "Fire_Name": "str", "Start_Date": "str",
                              "End_Date": "str", "Manual_Verification": "str", "State_Name": "str",
                              "Year": "int", "WFIG_Override_Flag": "str", "Storm_Start": "str", "In_SciBase": "str"}}
    
    with fiona.open(output_shapefile_path, "w", driver="ESRI Shapefile",
                    crs=src.crs, schema=schema) as dst:
        for feature in src:
            merge_src_value = feature["properties"].get("MERGE_SRC", None)
            
            if merge_src_value:
                match = fire_data_exploded[fire_data_exploded["MERGE_SRC"] == merge_src_value]
                
                if not match.empty:
                    row = match.iloc[0]  # Take the first matching entry (assuming no duplicates needed)
                    new_properties = {
                        "MERGE_SRC": merge_src_value,
                        "Fire_ID": row["Fire_ID"],
                        "Fire_Name": row["Fire_Name"],
                        "Start_Date": row["Start_Date"],
                        "End_Date": row["End_Date"],
                        "Manual_Verification": row["Manual Verification"],
                        "State_Name": row["State_Name"],
                        "Year": int(row["Year"]),
                        "WFIG_Override_Flag": row["WFIG_Override_Flag"],
                        "Storm_Start": row["Storm_Start"],
                        "In_SciBase": row["In_SciBase"]
                    }
                else:
                    new_properties = {k: None for k in schema["properties"].keys()}
                    new_properties["MERGE_SRC"] = merge_src_value
                    
                dst.write({"geometry": feature["geometry"], "properties": new_properties})

print(f"New shapefile saved as {output_shapefile_path}")



