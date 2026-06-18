# Workflow Automation Summary

## Automated Steps

- Downloading and extracting perimeter shapefiles from URLs in a CSV
- Converting shapefiles to WKT CSV for catalog queries
- Joining fire metadata to perimeter geometries
- Filling missing containment dates using web search (partially automated, but relies on internet and can require manual review if not found)
- Merging verification inventories and checking for duplicates
- Filtering topographic datasets by valid IDs
- Finding spatial overlaps between fire perimeters and topographic footprints
- Querying OpenTopography catalog for intersecting datasets
- Classifying dataset timing relative to fire events

---

## Manual/Non-Automated Steps

- Preparing and updating CSVs with correct columns and file paths (initial setup, corrections)
- Manual review/QC steps:
  - Checking for infinite values in shapefiles
  - Reviewing non-matching entries after fuzzy matching or overrides
  - Verifying problematic containment dates or metadata mismatches
- Manual download folder handling:
  - Some scripts refer to "Manual_Downloads" and similar directories, which may require user intervention if automated downloads fail
- Updating override and verification inventories:
  - Override CSVs and verification inventories must be manually curated and kept up-to-date

---

## Updates and Current Practices

- **Containment Dates:**  
  Manual verification of containment (end) dates is no longer necessary. Containment date information is now standardized and included in the USGS assessments; future workflows can pull this data directly from those sources.

- **In-Progress Analyses:**  
  Files with the `"11_"` prefix are currently in progress and being used to complete further spatial overlap analyses. These scripts are subject to change and represent ongoing work in the spatial analysis portion of the workflow.

- **Checking for New Overlaps and Updating Topo Footprints:**  
  You can check for new overlaps by querying the OpenTopography (OT) API. When updating, you may re-download topographic footprint metadata to check for new datasets.  
  If desired, this process can be automated: a script can exclude any datasets already present in your files and continue with only new datasets, seamlessly incorporating them into your existing workflow and files.