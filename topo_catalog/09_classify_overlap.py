# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 13:32:59 2025

@author: lguido

Purpose: Classifies the acquisition timing of topographic datasets relative to fire events (before, during, after, overlapping) and summarizes statistics for further analysis.
Automation: Fully automated classification and summary; manual review of classifications recommended.
Inputs: Overlap CSV with dates.
Outputs: Classified CSV and summary statistics.

"""
import pandas as pd
import ast

# Load the CSV file
df = pd.read_csv(r"...\USGS_Perim_Files\Compiling\Overlap_WithDates_Whole.csv")

# Convert Start_Date and End_Date to datetime format
df["Start_Date"] = pd.to_datetime(df["Start_Date"], format="%m/%d/%Y", errors='coerce')
df["End_Date"] = pd.to_datetime(df["End_Date"], format="%m/%d/%Y", errors='coerce')

# Define dataset start and end columns
dataset_start_cols = ["WESM_DATES_START", "OT_3DEP_DATES_START", "OT_NOAA_DATES_START", "OT_PC_DATES_START", "OT_Raster_DATES_START"]
dataset_end_cols = ["WESM_DATES_END", "OT_3DEP_DATES_END", "OT_NOAA_DATES_END", "OT_PC_DATES_END", "OT_Raster_DATES_END"]

def parse_dates(date_str):
    """Convert list-like string of dates to a sorted list of datetime objects."""
    try:
        date_list = ast.literal_eval(date_str)  # Convert string representation of list to actual list
        if not isinstance(date_list, list):
            date_list = [date_list]
        parsed_dates = sorted(pd.to_datetime(date_list, errors='coerce'))  # Convert & sort
        return [d for d in parsed_dates if pd.notna(d)]  # Remove NaT values
    except (ValueError, SyntaxError):
        return []

# Apply parsing to dataset start and end columns
for col in dataset_start_cols + dataset_end_cols:
    df[col] = df[col].astype(str).apply(parse_dates)

def classify_data_timing(start_dates, end_dates, fire_start, fire_end):
    """
    Classify each dataset acquisition relative to the fire event.
    Returns a list of classifications.
    """
    if pd.isna(fire_start) or pd.isna(fire_end):
        return ["Unknown"]

    # Ensure we have valid start and end dates
    if not start_dates or not end_dates:
        return ["Unknown"]

    classifications = []
    
    # Pair up start and end dates for each acquisition
    for start, end in zip(start_dates, end_dates):
        if end <= fire_start:
            classifications.append("Before Fire")
        elif start >= fire_end:
            classifications.append("After Fire")
        elif start < fire_start and end < fire_end:
            classifications.append("Overlapping - Ends Within Fire")
        elif start < fire_start and end > fire_end:
            classifications.append("Overlapping - Covers Entire Fire")
        elif start >= fire_start and end <= fire_end:
            classifications.append("Overlapping - Fully Within Fire")
        elif start > fire_start and end > fire_end:
            classifications.append("Overlapping - Starts Within Fire")
        else:
            classifications.append("Unknown")

    return classifications

# Apply classification for each dataset
for start_col, end_col in zip(dataset_start_cols, dataset_end_cols):
    classification_col = start_col.replace("_DATES_START", "_Fire_Class")
    df[classification_col] = df.apply(
        lambda row: classify_data_timing(row[start_col], row[end_col], row["Start_Date"], row["End_Date"]), axis=1
    )

# Add new columns to count each classification type
def count_classifications(row, fire_start, fire_end):
    """
    Count the occurrences of each classification type for a fire event across datasets.
    """
    classifications = []
    
    # For each dataset column, count the occurrences of classifications
    for start_col, end_col in zip(dataset_start_cols, dataset_end_cols):
        classification_col = start_col.replace("_DATES_START", "_Fire_Class")
        classifications.extend(row[classification_col])

    # Count the number of occurrences for each classification
    before_fire_count = classifications.count("Before Fire")
    after_fire_count = classifications.count("After Fire")
    overlapping_end_within = classifications.count("Overlapping - Ends Within Fire")
    overlapping_covers_entire = classifications.count("Overlapping - Covers Entire Fire")
    overlapping_fully_within = classifications.count("Overlapping - Fully Within Fire")
    overlapping_starts_within = classifications.count("Overlapping - Starts Within Fire")
    
    return pd.Series([
        before_fire_count, 
        after_fire_count, 
        overlapping_end_within, 
        overlapping_covers_entire, 
        overlapping_fully_within, 
        overlapping_starts_within
    ])

# Apply the counting function to each row
df[['Before Fire Count', 'After Fire Count', 
    'Overlapping - Ends Within Fire Count', 
    'Overlapping - Covers Entire Fire Count',
    'Overlapping - Fully Within Fire Count',
    'Overlapping - Starts Within Fire Count']] = df.apply(
    lambda row: count_classifications(row, row["Start_Date"], row["End_Date"]), axis=1
)


# Save results
#df.to_csv(r"...USGS_Perim_Files\Compiling\Classified_Overlap.csv", index=False)

print("Classification complete! Results saved to 'Classified_Overlap.csv'.")


# Summing up the counts for each category
sums = df[['Before Fire Count', 'After Fire Count', 
           'Overlapping - Ends Within Fire Count', 
           'Overlapping - Covers Entire Fire Count',
           'Overlapping - Fully Within Fire Count',
           'Overlapping - Starts Within Fire Count']].sum()

# Counting the number of fires with at least one entry in each count column
fires_with_entries = {
    "Before Fire": (df['Before Fire Count'] > 0).sum(),
    "After Fire": (df['After Fire Count'] > 0).sum(),
    "Overlapping - Ends Within Fire": (df['Overlapping - Ends Within Fire Count'] > 0).sum(),
    "Overlapping - Covers Entire Fire": (df['Overlapping - Covers Entire Fire Count'] > 0).sum(),
    "Overlapping - Fully Within Fire": (df['Overlapping - Fully Within Fire Count'] > 0).sum(),
    "Overlapping - Starts Within Fire": (df['Overlapping - Starts Within Fire Count'] > 0).sum()
}

# Counting the number of fires with at least one "Before Fire" and one "After Fire"
fires_with_before_and_after = ((df['Before Fire Count'] > 0) & (df['After Fire Count'] > 0)).sum()

fires_with_before_and_overlapping_after = ((df['Before Fire Count'] > 0) & (df['After Fire Count'] == 0) & 
                                           ((df['Overlapping - Starts Within Fire Count'] > 0) | 
                                            (df['Overlapping - Ends Within Fire Count'] > 0) | 
                                            (df['Overlapping - Fully Within Fire Count'] > 0) | 
                                            (df['Overlapping - Covers Entire Fire Count'] > 0))).sum()

fires_with_overlapping_before_and_complete_after = ((df['After Fire Count'] > 0) & (df['Before Fire Count'] == 0) & 
                                                    ((df['Overlapping - Ends Within Fire Count'] > 0) | 
                                                     (df['Overlapping - Starts Within Fire Count'] > 0) |
                                                     (df['Overlapping - Fully Within Fire Count'] > 0) | 
                                                     (df['Overlapping - Covers Entire Fire Count'] > 0))).sum()

# Print the statistics
print("\nSum of each count column:")
print(sums)

print("\nNumber of fires with at least one entry in each count column:")
for classification, count in fires_with_entries.items():
    print(f"{classification}: {count}")

print(f"\nNumber of fires with at least one entry in both 'Before Fire' and 'After Fire': {fires_with_before_and_after}")

print(f"Number of fires with Before Fire and Overlapping After: {fires_with_before_and_overlapping_after}")

print(f"Number of fires with Overlapping Before and Complete After: {fires_with_overlapping_before_and_complete_after}")

# Count fires with no data at all (no entries in any count column)
fires_with_no_data = (df[['Before Fire Count', 'After Fire Count', 
                          'Overlapping - Ends Within Fire Count', 
                          'Overlapping - Covers Entire Fire Count',
                          'Overlapping - Fully Within Fire Count',
                          'Overlapping - Starts Within Fire Count']].sum(axis=1) == 0).sum()

# Count fires with only Before data (Before Fire > 0 and no other counts)
fires_with_only_before = ((df['Before Fire Count'] > 0) & 
                          (df[['After Fire Count', 'Overlapping - Ends Within Fire Count', 
                               'Overlapping - Covers Entire Fire Count', 
                               'Overlapping - Fully Within Fire Count', 
                               'Overlapping - Starts Within Fire Count']].sum(axis=1) == 0)).sum()

# Count fires with only After data (After Fire > 0 and no other counts)
fires_with_only_after = ((df['After Fire Count'] > 0) & 
                         (df[['Before Fire Count', 'Overlapping - Ends Within Fire Count', 
                              'Overlapping - Covers Entire Fire Count', 
                              'Overlapping - Fully Within Fire Count', 
                              'Overlapping - Starts Within Fire Count']].sum(axis=1) == 0)).sum()

# Count fires with only Overlapping data (overlapping counts > 0 and no Before or After counts)
fires_with_only_overlapping = ((df[['Overlapping - Ends Within Fire Count', 
                                    'Overlapping - Covers Entire Fire Count', 
                                    'Overlapping - Fully Within Fire Count', 
                                    'Overlapping - Starts Within Fire Count']].sum(axis=1) > 0) & 
                               (df[['Before Fire Count', 'After Fire Count']].sum(axis=1) == 0)).sum()

# Print the additional statistics
print(f"Number of fires with no data at all: {fires_with_no_data}")
print(f"Number of fires with only Before Fire data: {fires_with_only_before}")
print(f"Number of fires with only After Fire data: {fires_with_only_after}")
print(f"Number of fires with only Overlapping data: {fires_with_only_overlapping}")

def categorize_fire(row):
    if row['Before Fire Count'] > 0 and row['After Fire Count'] > 0:
        return "fire_ids_with_before_and_after"
    elif row['Before Fire Count'] > 0 and row['After Fire Count'] == 0 and (
        row['Overlapping - Starts Within Fire Count'] > 0 or 
        row['Overlapping - Ends Within Fire Count'] > 0 or 
        row['Overlapping - Fully Within Fire Count'] > 0 or 
        row['Overlapping - Covers Entire Fire Count'] > 0):
        return "fire_ids_with_before_and_overlapping_after"
    elif row['After Fire Count'] > 0 and row['Before Fire Count'] == 0 and (
        row['Overlapping - Ends Within Fire Count'] > 0 or 
        row['Overlapping - Starts Within Fire Count'] > 0 or 
        row['Overlapping - Fully Within Fire Count'] > 0 or 
        row['Overlapping - Covers Entire Fire Count'] > 0):
        return "fire_ids_with_overlapping_before_and_complete_after"
    elif (row[['Before Fire Count', 'After Fire Count', 
               'Overlapping - Ends Within Fire Count', 
               'Overlapping - Covers Entire Fire Count',
               'Overlapping - Fully Within Fire Count',
               'Overlapping - Starts Within Fire Count']].sum() == 0):
        return "fire_ids_with_no_data"
    elif row['Before Fire Count'] > 0 and row[['After Fire Count', 
                                               'Overlapping - Ends Within Fire Count', 
                                               'Overlapping - Covers Entire Fire Count', 
                                               'Overlapping - Fully Within Fire Count', 
                                               'Overlapping - Starts Within Fire Count']].sum() == 0:
        return "fire_ids_with_only_before"
    elif row['After Fire Count'] > 0 and row[['Before Fire Count', 
                                              'Overlapping - Ends Within Fire Count', 
                                              'Overlapping - Covers Entire Fire Count', 
                                              'Overlapping - Fully Within Fire Count', 
                                              'Overlapping - Starts Within Fire Count']].sum() == 0:
        return "fire_ids_with_only_after"
    elif row[['Overlapping - Ends Within Fire Count', 
              'Overlapping - Covers Entire Fire Count', 
              'Overlapping - Fully Within Fire Count', 
              'Overlapping - Starts Within Fire Count']].sum() > 0 and row[['Before Fire Count', 'After Fire Count']].sum() == 0:
        return "fire_ids_with_only_overlapping"
    else:
        return "Unclassified"

# Apply the categorization function to create the new "Category" column
df["Category"] = df.apply(categorize_fire, axis=1)

# Save the modified DataFrame
#df.to_csv(r"...USGS_Perim_Files\Compiling\Classified_Overlap.csv", index=False)

print("Classification complete! Results saved to 'Classified_Overlap.csv'.")

###QC this CSV with outprint before re-doing histogram and other plots! 