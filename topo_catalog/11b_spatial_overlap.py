# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:03:19 2025

@author: lguido
"""

import csv
import os

def extract_area_data(paired_overlap_path, elevation_folder_path, output_path):
    results = []

    # Read the Paired_Overlap.csv file
    with open(paired_overlap_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Loop through each row in the Paired_Overlap.csv file
        for row in reader:
            fire_id = row['Fire_ID']
            overlap_combinations = eval(row['Overlap_Combinations'])
            source_keys = eval(row['Source_Keys'])
            days_elapsed = eval(row['Days_Elapsed'])

            area_values = []  # List to store AREA values
            unmatched_pairs = []  # To track pairs that don't have AREA values

            # Loop through each pair of identifiers in the overlap combinations and corresponding source keys
            # Loop through each pair of identifiers in the overlap combinations and corresponding source keys
            for (id1, id2), (key1, key2) in zip(overlap_combinations, source_keys):
                # Construct the correct companion file name using the source keys
                companion_file_name = f"{key1}_{key2}_tab.csv"
                companion_file_path = os.path.join(elevation_folder_path, companion_file_name)
            
                # Check if the expected companion file exists
                if os.path.exists(companion_file_path):
                    found = False
                    with open(companion_file_path, newline='', encoding='utf-8') as companion_file:
                        companion_reader = csv.reader(companion_file)
                        for companion_row in companion_reader:
                            # Ensure correct matching of the pair order
                            if (companion_row[0].strip(), companion_row[1].strip()) in [(id1.strip(), id2.strip()), (id2.strip(), id1.strip())]:
                                area_values.append(companion_row[2])  # Append the AREA value
                                found = True
                                break
                    if not found:
                        # If no match is found for the pair, append 0 and log the unmatched pair
                        area_values.append('0')
                        unmatched_pairs.append((id1, id2))
                else:
                    # If the expected companion file does not exist, append 0 and log the missing file
                    print(f"Missing expected file: {companion_file_name}")  # Debugging
                    area_values.append('0')
                    unmatched_pairs.append(f"File not found: {companion_file_name}")

            # Log any discrepancies
            if len(area_values) != len(overlap_combinations):
                print(f"Warning: Mismatch found in Fire_ID {fire_id}:")
                print(f"Expected pairs: {len(overlap_combinations)}")
                print(f"Found area values: {len(area_values)}")
                print(f"Unmatched pairs or missing files: {unmatched_pairs}")
                print("Area values:", area_values)
                print("Overlap combinations:", overlap_combinations)

            # Append the collected data to the results list
            results.append({
                'Fire_ID': fire_id,
                'Overlap_Combinations': overlap_combinations,
                'Source_Keys': source_keys,
                'Days_Elapsed': days_elapsed,
                'Area_Values': area_values
            })

    # Write the updated results with AREA values to a new CSV
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Fire_ID', 'Overlap_Combinations', 'Source_Keys', 'Days_Elapsed', 'Area_Values']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write the rows to the output file
        for result in results:
            writer.writerow({
                'Fire_ID': result['Fire_ID'],
                'Overlap_Combinations': result['Overlap_Combinations'],
                'Source_Keys': result['Source_Keys'],
                'Days_Elapsed': result['Days_Elapsed'],
                'Area_Values': result['Area_Values']
            })

    print(f'Processing complete. Results saved to {output_path}')
# Example usage
paired_overlap_path = r"C:\Users\lguido\OneDrive - DOI\Desktop\USGS_Perim_Files\Compiling\Paired_Overlap.csv"
elevation_folder_path = r"C:\Users\lguido\OneDrive - DOI\Desktop\USGS_Perim_Files\Compiling\Elevation_Overlaps"
output_path = r"C:\Users\lguido\OneDrive - DOI\Desktop\USGS_Perim_Files\Compiling\Paired_Overlap_with_Area.csv"

extract_area_data(paired_overlap_path, elevation_folder_path, output_path)