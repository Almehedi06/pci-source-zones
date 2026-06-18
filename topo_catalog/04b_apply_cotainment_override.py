# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:36:22 2024

@author: lguido

Purpose: Applies manual containment date overrides from a curated CSV using fuzzy matching, updating the main fire event CSV.
Automation: Automated override application; override CSV must be manually maintained.
Inputs: Override CSV and updated fire event CSV.
Outputs: Fire event CSV with overrides applied.

"""

import pandas as pd
from fuzzywuzzy import process

# Paths to the input CSVs
override_path = r"...USGS_Perim_Files\WFIGS_Override.csv"
updated_path = r"...USGS_Perim_Files\updated_with_end_dates.csv"
output_path = r"...USGS_Perim_Files\pfdf_with_end.csv"

# Read the CSVs
override_df = pd.read_csv(override_path)
updated_df = pd.read_csv(updated_path)

# Convert names to lowercase for case-insensitive matching
override_df['poly_Incid_lower'] = override_df['poly_Incid'].str.lower()
updated_df['Fire_Name_lower'] = updated_df['Fire_Name'].str.lower()

# Perform fuzzy matching
matched_entries = []
non_matching_override = []

for poly_incid in override_df['poly_Incid_lower']:
    result = process.extractOne(poly_incid, updated_df['Fire_Name_lower'])
    if result:
        match, score = result[0], result[1]
        if score > 90:  # Adjust the threshold as necessary
            matched_entries.append({
                "poly_Incid": poly_incid,
                "Fire_Name": match,
                "attr_Conta": override_df.loc[override_df['poly_Incid_lower'] == poly_incid, 'attr_Conta'].values[0],
                "End_Date": updated_df.loc[updated_df['Fire_Name_lower'] == match, 'End_Date'].values[0]
            })
        else:
            # Add to non-matching if the match score is too low
            non_matching_override.append(poly_incid)
    else:
        # Add to non-matching if no match is found
        non_matching_override.append(poly_incid)


# Convert matched and non-matched to DataFrames
matched_df = pd.DataFrame(matched_entries)
non_matching_df = pd.DataFrame(non_matching_override, columns=['poly_Incid'])

# Print the non-matching override entries
print("Non-matching override entries:")
print(non_matching_df)

# Replace End_Date in updated_df with attr_Conta for matching entries
for match in matched_entries:
    updated_df.loc[updated_df['Fire_Name_lower'] == match['Fire_Name'], 'End_Date'] = match['attr_Conta']

# Drop the lowercase helper columns
override_df.drop(columns=['poly_Incid_lower'], inplace=True)
updated_df.drop(columns=['Fire_Name_lower'], inplace=True)

# Save the updated DataFrame to a new CSV
updated_df.to_csv(output_path, index=False)

# Output the matched DataFrame
print("\nMatched entries:")
print(matched_df)
