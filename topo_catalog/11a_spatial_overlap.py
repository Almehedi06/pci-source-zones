# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:23:48 2025

@author: lguido
"""

import csv
from itertools import product
from datetime import datetime
import ast
import pandas as pd
from itertools import combinations

def process_fire_overlap(csv_path, output_path):
    results = []
    valid_classes = {
        "Before Fire": {"After Fire", "Overlapping - Ends Within Fire", "Overlapping - Covers Entire Fire", "Overlapping - Fully Within Fire", "Overlapping - Starts Within Fire"},
        "After Fire": {"After Fire", "Overlapping - Ends Within Fire", "Overlapping - Covers Entire Fire", "Overlapping - Fully Within Fire", "Overlapping - Starts Within Fire"},
        "Overlapping - Ends Within Fire": {"Overlapping - Ends Within Fire", "Overlapping - Covers Entire Fire", "Overlapping - Fully Within Fire", "Overlapping - Starts Within Fire", "After Fire"},
        "Overlapping - Covers Entire Fire": {"Overlapping - Ends Within Fire", "Overlapping - Covers Entire Fire", "Overlapping - Fully Within Fire", "Overlapping - Starts Within Fire", "After Fire"},
        "Overlapping - Fully Within Fire": {"Overlapping - Ends Within Fire", "Overlapping - Covers Entire Fire", "Overlapping - Fully Within Fire", "Overlapping - Starts Within Fire", "After Fire"},
        "Overlapping - Starts Within Fire": {"Overlapping - Ends Within Fire", "Overlapping - Covers Entire Fire", "Overlapping - Fully Within Fire", "Overlapping - Starts Within Fire", "After Fire"},
    }
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        

        for row in reader:
            fire_id = row['Fire_ID']
            possible_sets = []
            source_map = {}
            date_map = {}

            # Extract identifiers, classifications, and end dates
            for key in ['WESM', 'OT_3DEP', 'OT_NOAA', 'OT_PC', 'OT_Raster']:
                ids = eval(row[key]) if row[key] else []
                classes = eval(row[f'{key}_Fire_Class']) if row[f'{key}_Fire_Class'] else []
                
                # Convert the end dates using your conversion function
                dates = convert_dates(row[f'{key}_DATES_END']) if row[f'{key}_DATES_END'] else []

                # Pair identifiers with their classifications, sources, and dates
                for identifier, classification, date in zip(ids, classes, dates):
                    if classification != "Unknown":  # Ignore unknown classifications
                        possible_sets.append((identifier, classification))
                        source_map[identifier] = key  # Track the source dataset
                        if date:  # Only store the date if it's valid
                            date_map[identifier] = date  # Store parsed date
                            
                            
            # Debug: Print possible_sets and source_map
            print("Possible sets:")
            for ps in possible_sets:
                print(f"ID: {ps[0]}, Class: {ps[1]}")
            print("Source Map:")
            for key, value in source_map.items():
                print(f"ID: {key}, Source: {value}")

            # Generate valid pairwise combinations based on classification rules
            valid_combinations = []
            key_pairs = []
            days_elapsed_list = []
            # Use combinations to get all unique pairs from possible_sets
            for (id1, class1), (id2, class2) in combinations(possible_sets, 2):
                print(f"Evaluating combination: ({id1}, {id2}) with classes ({class1}, {class2})")
            
                # Check if the combination of classes is valid in either order
                if (class2 in valid_classes.get(class1, set())) or (class1 in valid_classes.get(class2, set())):
                    # Create a valid combination and ensure order is preserved
                    pair = (id1, id2)  # Preserve order as encountered
                    valid_combinations.append(pair)
            
                    # Add corresponding source keys to the key_pairs list
                    source_pair = (source_map[id1], source_map[id2])
                    key_pairs.append(source_pair)
            
                    # Calculate days elapsed and store it
                    if id1 in date_map and id2 in date_map:
                        days_elapsed = (date_map[id1] - date_map[id2]).days
                        days_elapsed_list.append(days_elapsed)
                        
            for i in range(len(valid_combinations)):
                print(f"Combination: {valid_combinations[i]}, Key Pair: {key_pairs[i]}, Days Elapsed: {days_elapsed_list[i]}")

            # Debug: Print the valid combinations and source keys
            print("Valid Combinations:")
            print(valid_combinations)
            print("Key Pairs:")
            print(key_pairs)
            
            print("Final Output Validation:")
            
            
            results.append({'Fire_ID': fire_id, 'Overlap_Combinations': list(valid_combinations), 'Source_Keys': key_pairs, 'Days_Elapsed': days_elapsed_list})
    
    # Write results to a new CSV
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Fire_ID', 'Overlap_Combinations', 'Source_Keys', 'Days_Elapsed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({'Fire_ID': result['Fire_ID'], 'Overlap_Combinations': result['Overlap_Combinations'], 'Source_Keys': result['Source_Keys'], 'Days_Elapsed': result['Days_Elapsed']})
    
    print(f'Processing complete. Results saved to {output_path}')
    

def convert_dates(value):
    if isinstance(value, str):  # Check if the entry is a string that looks like a list
        print(f"String value: {value}")  # Debug print
        try:
            # Handle the case where we have a string that looks like a list of Timestamps
            # e.g. "[Timestamp('2018-10-07 00:00:00')]"
            # Attempt to convert the string into an actual list of Timestamps
            value = value.replace("Timestamp", "")  # Remove 'Timestamp' text
            value = eval(value)  # Safely evaluate as a list of datetime-like objects
            print(f"Converted to list: {value}")
        except (ValueError, SyntaxError):
            print(f"Failed to convert: {value}")  # Debug print
            value = []  # Return empty list if it fails to evaluate
    if isinstance(value, list):  # If it's a list, process each entry individually
        return [pd.to_datetime(x) if isinstance(x, str) else pd.NaT for x in value]
    else:  # If it's not a list, just convert the single value
        return pd.to_datetime(value, errors='coerce')

# Example usage
process_fire_overlap(r"C:\Users\lguido\OneDrive - DOI\Desktop\USGS_Perim_Files\Compiling\Classified_Overlap.csv", r"C:\Users\lguido\OneDrive - DOI\Desktop\USGS_Perim_Files\Compiling\Paired_Overlap.csv")
