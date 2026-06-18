# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:15:51 2025

@author: lguido

Purpose: Attempts to fill missing containment (end) dates for fires using web search and regex extraction from online sources.
Automation: Partially automated; may require manual review for ambiguous or missing results.
Inputs: Fire metadata CSV (with missing dates).
Outputs: CSV with updated containment dates.

"""

import pandas as pd
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import re
import time
from urllib.parse import urlparse
from fuzzywuzzy import process

# Function to extract containment dates from webpage content with various formats

def extract_containment_date(content):
    date_patterns = [
    r"([A-Z][a-z]+ \d{1,2}, \d{4})", # Month Day, Year (e.g., October 15, 2023)
    r"([A-Z][a-z]+ \d{1,2}, \d{4} at \d{1,2}(:\d{2})? (AM|PM))", # Month Day, Year at Time
    r"(\d{1,2} [A-Z][a-z]+ \d{4})", # Day Month Year (e.g., 15 October 2023)
    r"(\d{2}/\d{2}/\d{4})", # Numeric format: MM/DD/YYYY (e.g., 10/15/2023)
    r"(\d{2}-\d{2}-\d{4})", # Numeric format with dashes: MM-DD-YYYY (e.g., 10-15-2023)
    r"(\d{4}/\d{2}/\d{2})", # Numeric format: YYYY/MM/DD (e.g., 2023/10/15)
    r"(\d{2}/\d{2}/\d{2})" # Numeric format with two-digit year: MM/DD/YY (e.g., 10/15/23)
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, content)
        if matches:
            return matches[0]
    return None

# Function to determine if a URL is .gov or Wikipedia

def is_special_url(url):
    domain = urlparse(url).netloc
    return domain.endswith('.gov') or 'wikipedia.org' in domain

# Function to fetch containment date

def fetch_containment_date(fire_name, state_name, year):
    query = f"{fire_name} fire {state_name} {year} containment date"
    try:
        search_results = list(search(query, stop=10))  # Retrieve top 10 results
        print(f"Search results for {fire_name} fire {state_name} {year}: {search_results}")
        
        for url in search_results:
            print(f"Checking URL: {url}")
            try:
                # Adjust timeout for special URLs
                timeout = 200 if is_special_url(url) else 10
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                
                # Parse webpage content
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text()
                
                # Extract containment date
                date = extract_containment_date(content)
                if date:
                    print(f"Found containment date: {date}")
                    return date
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
        
        return None
    except Exception as e:
        print(f"Error fetching data for {fire_name} fire {state_name} {year}: {e}")
        return None

# Load the input CSV
input_path = r"...USGS_Perim_Files\updated_cleaned_merged_atts.csv"
df = pd.read_csv(input_path)

df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
df['Year'] = df['Start_Date'].dt.year.astype('Int64')
search_cache = {}

# Iterate through rows to find containment dates
for idx, row in df.iterrows():
    if pd.isna(row['End_Date']):  # Process only rows with missing End_Date
        fire_name = row['Fire_Name']
        state_name = row['State_Name']
        year = row['Year']
        
        # Use a tuple as the key to check the cache
        cache_key = (fire_name, state_name, year)
        if cache_key in search_cache:
            print(f"Using cached result for: {cache_key}")
            df.at[idx, 'End_Date'] = search_cache[cache_key]
        else:
            print(f"Searching containment date for: {fire_name}, {state_name}, {year}")
            containment_date = fetch_containment_date(fire_name, state_name, year)
            if containment_date:
                search_cache[cache_key] = containment_date
                df.at[idx, 'End_Date'] = containment_date
            time.sleep(1)  # Be polite and avoid overwhelming servers

df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce')
df['End_Date'] = df['End_Date'].dt.strftime('%m/%d/%Y')

# Save the updated DataFrame
output_path = r"...USGS_Perim_Files\updated_with_end_dates.csv"
df.to_csv(output_path, index=False)

print(f"Updated CSV with End_Date saved to {output_path}")

# Paths to input CSVs
override_path = r"...USGS_Perim_Files\WFIGS_Override_contain.csv"
updated_path = r"...USGS_Perim_Files\updated_with_end_dates.csv"
output_path = r"...USGS_Perim_Files\pfdf_with_end.csv"

override_df = pd.read_csv(override_path)
updated_df = pd.read_csv(updated_path)

override_df['poly_Incid_lower'] = override_df['poly_Incid'].str.lower()
updated_df['Fire_Name_lower'] = updated_df['Fire_Name'].str.lower()
updated_df['WFIG_Override_Flag'] = 'No'
updated_df['Search_End'] = ''
updated_df['Temporal_Error'] = ''

matched_entries = []
non_matching_override = []

for poly_incid in override_df['poly_Incid_lower']:
    result = process.extractOne(poly_incid, updated_df['Fire_Name_lower'])
    if result:
        match, score = result[0], result[1]
        if score > 90:
            matched_entries.append({
                "poly_Incid": poly_incid,
                "Fire_Name": match,
                "attr_Conta": override_df.loc[override_df['poly_Incid_lower'] == poly_incid, 'attr_Conta'].values[0],
                "End_Date": updated_df.loc[updated_df['Fire_Name_lower'] == match, 'End_Date'].values[0]
            })
        else:
            non_matching_override.append(poly_incid)
    else:
        non_matching_override.append(poly_incid)

matched_df = pd.DataFrame(matched_entries)
non_matching_df = pd.DataFrame(non_matching_override, columns=['poly_Incid'])
print("Non-matching override entries:")
print(non_matching_df)

for match in matched_entries:
    updated_df.loc[updated_df['Fire_Name_lower'] == match['Fire_Name'], 'Search_End'] = match['End_Date']
    updated_df.loc[updated_df['Fire_Name_lower'] == match['Fire_Name'], 'End_Date'] = match['attr_Conta']
    updated_df.loc[updated_df['Fire_Name_lower'] == match['Fire_Name'], 'WFIG_Override_Flag'] = 'Yes'
    
# Compute Temporal Error
updated_df['Start_Date'] = pd.to_datetime(updated_df['Start_Date'], errors='coerce')
updated_df['End_Date'] = pd.to_datetime(updated_df['End_Date'], errors='coerce')
updated_df.loc[updated_df['End_Date'] < updated_df['Start_Date'], 'Temporal_Error'] = 'Error'

override_df.drop(columns=['poly_Incid_lower'], inplace=True)
updated_df.drop(columns=['Fire_Name_lower'], inplace=True)
updated_df.to_csv(output_path, index=False)

print("\nMatched entries:")
print(matched_df)