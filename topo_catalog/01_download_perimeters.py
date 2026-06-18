# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:58:27 2024

@author: lguido

Purpose: Automatically downloads wildfire perimeter shapefiles from URLs listed in a CSV, extracts them, cleans up unnecessary files, and consolidates the outputs.
Automation: Fully automated if input CSV is prepared; manual intervention required only for failed downloads or manual additions.
Inputs: CSV with perimeter URLs.
Outputs: Extracted shapefiles and a status CSV.


"""

import os
import pandas as pd
import requests
import zipfile
import shutil

# Function to download and unzip a file
def download_and_extract(url, download_dir):
    """
    Downloads and extracts a ZIP file from the given URL to the specified directory.
    Returns True if successful, False otherwise.
    """
    local_zip = os.path.join(download_dir, os.path.basename(url))
    try:
        # Download the file
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an error for bad responses
        with open(local_zip, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {local_zip}")

        # Validate if the file is a ZIP
        if not zipfile.is_zipfile(local_zip):
            print(f"Warning: {local_zip} is not a valid ZIP file. Skipping.")
            os.remove(local_zip)  # Remove invalid file
            return False

        # Extract the ZIP file
        extracted_dir = os.path.join(download_dir, os.path.splitext(os.path.basename(url))[0])
        os.makedirs(extracted_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print(f"Extracted: {local_zip} to {extracted_dir}")

        # Clean up the downloaded ZIP file
        os.remove(local_zip)
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(local_zip):
            os.remove(local_zip)
        return False

    except zipfile.BadZipFile:
        print(f"Error: {local_zip} is not a valid ZIP file.")
        if os.path.exists(local_zip):
            os.remove(local_zip)
        return False

    except Exception as e:
        print(f"Unexpected error processing {url}: {e}")
        if os.path.exists(local_zip):
            os.remove(local_zip)
        return False

def clean_directory(target_dir, retain_file):
    """
    Cleans the target directory and its subdirectories by deleting all files
    that do not contain 'perim_feat' in their name, except for the specified retain_file.
    """
    files_deleted = 0
    for root, _, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip the retain_file
            if os.path.abspath(file_path) == os.path.abspath(retain_file):
                continue
            # Delete files that do not contain 'perim_feat' in the name
            if "perim_feat" not in file and "-perimeter" not in file:
                try:
                    os.remove(file_path)
                    files_deleted += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    print(f"Cleanup complete. Total files deleted: {files_deleted}")

def consolidate_files(source_dir, target_dir):
    """
    Moves all files from source_dir and its subdirectories into target_dir.
    Creates target_dir if it doesn't exist and removes empty subdirectories after moving files.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    files_moved = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)

            try:
                # Handle potential duplicate filenames by renaming
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(file)
                    count = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(target_dir, f"{base}_{count}{ext}")
                        count += 1
                
                # Move the file
                shutil.move(source_file, target_file)
                files_moved += 1
                print(f"Moved: {source_file} -> {target_file}")

            except Exception as e:
                print(f"Failed to move {source_file}: {e}")

    # Remove empty directories
    for root, dirs, _ in os.walk(source_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"Failed to remove directory {dir_path}: {e}")

    print(f"Consolidation complete. Total files moved: {files_moved}")
    

# Paths
csv_path = r"...Estab_Dir\pfdf_loc.csv"
output_dir = r"...USGS_Perim_Files"
output_csv = r"...Estab_Dir\pfdf_loc_with_status_20241126.csv"
target_dir = r"...USGS_Perim_Files\Shapefiles_Init"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV
df = pd.read_csv(csv_path)

# Initialize a column for download status
df['Download_Status'] = 'Pending'


# Process each URL in the CSV
for idx, row in df.iterrows():
    url = row['SHP_link']
    print(f"Processing entry {idx + 1}/{len(df)}: {url}")
    success = download_and_extract(url, output_dir)
    df.at[idx, 'Download_Status'] = 'Success' if success else 'Failed'

# Save the updated CSV
df.to_csv(output_csv, index=False)
print(f"Finished processing. Updated CSV saved to {output_csv}.")

# Execute the cleanup
clean_directory(output_dir, output_csv)

consolidate_files(output_dir, target_dir)


manual_download = r"...USGS_Perim_Files\Manual_Downloads"
manual_target = r"...USGS_Perim_Files\Manual_Init"

