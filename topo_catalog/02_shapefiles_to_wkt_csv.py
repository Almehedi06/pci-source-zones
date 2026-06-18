# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:07:18 2025

@author: lguido

Purpose: Converts downloaded shapefiles into WKT format and exports a CSV for use in spatial queries and catalog searches.
Automation: Fully automated; requires consolidated shapefiles as input.
Inputs: Perimeter shapefiles.
Outputs: WKT CSV.

"""

import fiona
import shapely.wkt
import shapely.geometry
import pandas as pd

# Input and output paths
shapefile_path = r"...USGS_Perim_Files\merged_WGS84.shp"
csv_output_path = r"...USGS_Perim_Files\merged_wkt.csv"

# Read the shapefile
records = []
with fiona.open(shapefile_path, "r") as src:
    crs = src.crs  # Save the CRS in case it's needed later
    for feature in src:
        props = feature["properties"]  # Attribute table
        geom = shapely.geometry.shape(feature["geometry"])  # Convert to Shapely
        
        # Handle multipart geometries
        if isinstance(geom, shapely.geometry.MultiPolygon):
            for polygon in geom.geoms:
                polygon = polygon.buffer(0)  # Fix potential geometry issues
                polygon_filled = shapely.geometry.Polygon(polygon.exterior)  # Fill holes
                props_copy = props.copy()
                props_copy["WKT"] = polygon_filled.wkt
                records.append(props_copy)
        else:
            polygon = geom.buffer(0)  # Fix potential geometry issues
            polygon_filled = shapely.geometry.Polygon(polygon.exterior)  # Fill holes
            props["WKT"] = polygon_filled.wkt
            records.append(props)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(records)
df.to_csv(csv_output_path, index=False)

print(f"CSV saved: {csv_output_path}")

