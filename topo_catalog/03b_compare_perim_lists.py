# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:46:16 2025

@author: lguido

Purpose: Compares lists of perimeter sources between two CSVs to identify discrepancies.
Automation: Automated comparison; manual interpretation of results required.
Inputs: Two CSVs with perimeter source listings.
Outputs: Console output of differences.

"""

import pandas as pd
import ast

# Load CSV files
csv1 = r"...USGS_Perim_Files\Compiling\merged_fresh.csv"  # Update with actual path
csv2 = r"C:\Users\lguido\Downloads\fire_and_flow(fire_and_flow).csv" # Update with actual path

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# Normalize MERGE_SRC column in df1 (assuming one entry per row)
files1 = set(df1['MERGE_SRC'].astype(str).tolist())

# Normalize MERGE_SRC column in df2 (assuming a list-like string)
df2['MERGE_SRC'] = df2['MERGE_SRC'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else [x])
files2 = set(file for sublist in df2['MERGE_SRC'] for file in sublist)

# Identify differences
only_in_csv1 = files1 - files2
only_in_csv2 = files2 - files1

# Print results
print("Files in CSV1 but not in CSV2:", only_in_csv1)
print("Files in CSV2 but not in CSV1:", only_in_csv2)
