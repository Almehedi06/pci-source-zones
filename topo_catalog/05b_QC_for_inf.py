# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:04:04 2024

@author: lguido


Purpose: Performs quality control on shapefiles by checking for infinite values in a specified field.
Automation: Automated check; manual review of flagged files required.
Inputs: CSV listing shapefiles.
Outputs: Console output of QC results.

"""

import pandas as pd
import numpy as np
import os
import fiona

# Function to check for 'inf' values in the shapefile
def check_shapefile_for_inf(shapefile_path):
    try:
        # Open the shapefile using fiona
        with fiona.open(shapefile_path) as src:
            # Check if 'Distance' field exists
            if 'Distance' not in src.schema['properties']:
                print(f"'Distance' column not found in {shapefile_path}.")
                return 0
            
            inf_count = 0
            
            # Iterate through each feature in the shapefile
            for feature in src:
                distance_value = feature['properties']['Distance']
                if distance_value == float('inf'):
                    inf_count += 1
            
            return inf_count
    except Exception as e:
        print(f"Error reading {shapefile_path}: {e}")
        return 0

# Main function to read the CSV and process shapefiles
def main(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Check for the 'Branch Points Path' column
    if 'Points Path' not in df.columns:
        print("Column 'Points Path' not found in the CSV.")
        return

    # Dictionary to store results
    shapefile_inf_counts = {}

    # Iterate through each shapefile path
    for index, row in df.iterrows():
        shapefile_path = row['Points Path']
        
        shapefile_name = os.path.basename(shapefile_path)

        # Check for 'inf' values in the shapefile
        inf_count = check_shapefile_for_inf(shapefile_path)

        # Store the result if there are 'inf' values
        if inf_count > 0:
            shapefile_inf_counts[shapefile_name] = inf_count

    # Print the results
    for shapefile, count in shapefile_inf_counts.items():
        print(f"{shapefile}: {count} inf values")
        
    print("Count Complete")

# Replace 'your_file.csv' with the path to your CSV file
if __name__ == "__main__":
    main("...DataCranking/Width_key.csv")