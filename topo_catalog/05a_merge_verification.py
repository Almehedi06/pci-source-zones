# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:44:52 2025

@author: lguido

Purpose: Merges a verified event inventory with the main fire event CSV, flags duplicates, and normalizes metadata.
Automation: Automated merging and duplicate detection; manual curation of input inventories required.
Inputs: Main fire event CSV and verification inventory CSV.
Outputs: Final merged CSV.

"""
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load the working assessments CSV
assessments = pd.read_csv(r"...USGS_Perim_Files\pfdf_with_end.csv", dtype={'Fire_ID': str})

# Load the verified events CSV
verified_events = pd.read_csv(r"...USGS_Perim_Files\MasterVerificationInventory_20241126.csv", dtype={'FireID': str})

# Print the number of unique Fire_IDs in each input file
print(f"Unique Fire_IDs in working assessments: {assessments['Fire_ID'].nunique()}")
print(f"Unique Fire_IDs in verified events: {verified_events['FireID'].nunique()}")

# Group MERGE_SRC values by Fire_ID into lists
merge_src_dict = assessments.groupby('Fire_ID')['MERGE_SRC'].apply(list).to_dict()

# Drop duplicate Fire_ID entries in assessments to retain only unique Fire_IDs
assessments = assessments.drop_duplicates(subset=['Fire_ID']).reset_index(drop=True)

# Add the grouped MERGE_SRC column back to assessments
assessments["MERGE_SRC"] = assessments["Fire_ID"].map(merge_src_dict)

# Group StormStart values by Fire_ID into lists
stormstart_dict = verified_events.groupby('FireID')['StormStart'].apply(list).to_dict()

# Add the grouped Storm_Start column to assessments
assessments["Storm_Start"] = assessments["Fire_ID"].map(stormstart_dict)

# Fill NaN values with empty lists where there are no corresponding Fire_IDs in verified_events
assessments["Storm_Start"] = assessments["Storm_Start"].apply(lambda x: x if isinstance(x, list) else [])
assessments["MERGE_SRC"] = assessments["MERGE_SRC"].apply(lambda x: x if isinstance(x, list) else [])

# Get Fire_IDs that are only in verified_events but not in assessments
extra_fire_ids = set(verified_events['FireID']) - set(assessments['Fire_ID'])

# Create a mapping of additional information from verified_events
extra_info = verified_events.drop_duplicates(subset=['FireID']).set_index('FireID')

# Function to convert FireStartDate format (YYYYMMDD to MM/DD/YYYY)
def format_fire_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y%m%d').strftime('%m/%d/%Y')
    except:
        return None  # Return None if conversion fails

# Create a DataFrame for extra Fire_IDs from verified_events
extra_rows = pd.DataFrame({
    "Fire_ID": list(extra_fire_ids),
    "Storm_Start": [stormstart_dict[fid] for fid in extra_fire_ids],
    "MERGE_SRC": [[] for _ in extra_fire_ids],  # Empty list since these Fire_IDs aren't in assessments
    "Fire_Name": [extra_info.loc[fid, "FireName"] if fid in extra_info.index else None for fid in extra_fire_ids],
    "Year": [extra_info.loc[fid, "FireYear"] if fid in extra_info.index else None for fid in extra_fire_ids],
    "State_Name": [extra_info.loc[fid, "FireState"] if fid in extra_info.index else None for fid in extra_fire_ids],
    "Start_Date": [format_fire_date(extra_info.loc[fid, "FireStartDate"]) if fid in extra_info.index else None for fid in extra_fire_ids]
})

# Add empty columns matching the assessments CSV
for col in assessments.columns:
    if col not in extra_rows.columns:
        extra_rows[col] = None  # Fill other fields with NaN

# Append extra Fire_IDs to the main DataFrame
final_df = pd.concat([assessments, extra_rows], ignore_index=True)

# Print the number of unique Fire_IDs in the final merged CSV
print(f"Unique Fire_IDs in final fire_and_flow CSV: {final_df['Fire_ID'].nunique()}")

# Normalize Fire Names (lowercase and strip spaces for consistency)
final_df["Fire_Name_Norm"] = final_df["Fire_Name"].str.lower().str.strip()

# **Step 1: Detect exact duplicates (same Fire_Name, Year, State, and Start_Date but different Fire_IDs)**
duplicate_suspects = final_df[final_df.duplicated(subset=["Fire_Name_Norm", "Year", "State_Name", "Start_Date"], keep=False)]

if not duplicate_suspects.empty:
    print("\n🔥 Possible duplicate Fire_IDs detected (EXACT MATCHES):")
    print(duplicate_suspects[["Fire_ID", "Fire_Name", "Year", "State_Name", "Start_Date"]].sort_values(by=["Fire_Name", "Year"]))

# **Step 2: Use Fuzzy Matching for Near Duplicates**
potential_duplicates = []
fire_names = final_df["Fire_Name_Norm"].dropna().unique()

# Check each fire name against others
for idx, fire in enumerate(fire_names):
    matches = process.extract(fire, fire_names, scorer=fuzz.ratio, limit=10)
    
    for match in matches:
        matched_name, score = match
        
        # Only consider names with similarity above 85% (adjust as needed)
        if fire != matched_name and score > 85:
            fire_rows = final_df[(final_df["Fire_Name_Norm"] == fire)]
            matched_rows = final_df[(final_df["Fire_Name_Norm"] == matched_name)]
            
            for _, fire_row in fire_rows.iterrows():
                for _, matched_row in matched_rows.iterrows():
                    # Check if the other details match (Year, State, Start Date) but Fire_ID is different
                    if (
                        fire_row["Year"] == matched_row["Year"] and
                        fire_row["State_Name"] == matched_row["State_Name"] and
                        fire_row["Start_Date"] == matched_row["Start_Date"] and
                        fire_row["Fire_ID"] != matched_row["Fire_ID"]
                    ):
                        potential_duplicates.append({
                            "Fire_ID_1": fire_row["Fire_ID"],
                            "Fire_Name_1": fire_row["Fire_Name"],
                            "Fire_ID_2": matched_row["Fire_ID"],
                            "Fire_Name_2": matched_row["Fire_Name"],
                            "Year": fire_row["Year"],
                            "State_Name": fire_row["State_Name"],
                            "Start_Date": fire_row["Start_Date"],
                            "Similarity_Score": score
                        })

# Convert to DataFrame for analysis
potential_dupes_df = pd.DataFrame(potential_duplicates)

if not potential_dupes_df.empty:
    print("\n⚠️  Potential duplicate Fire_IDs (FUZZY MATCHES):")
    print(potential_dupes_df.sort_values(by=["Similarity_Score"], ascending=False))

# Save the result to CSV
final_df.to_csv(r"...USGS_Perim_Files\fire_and_flow.csv", index=False)



