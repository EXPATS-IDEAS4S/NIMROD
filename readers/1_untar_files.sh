
#!/bin/bash

# Path to the .tar.gz file
TAR_GZ_FILE="/data/sat/msg/radar/nimrod/2023/metoffice-c-band-rain-radar_europe_20231231_5km-composite.dat.gz.tar"

# Destination directory for extracted files
DEST_DIR="/data/sat/msg/radar/nimrod/netcdf/2023/"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Extract the .tar.gz file
echo "Extracting $TAR_GZ_FILE..."
tar -xzf "$TAR_GZ_FILE" -C "$DEST_DIR" #TODO this is not working

# Change to the destination directory
cd "$DEST_DIR"

# Find and decompress all .gz files
echo "Decompressing .gz files..."
find . -type f -name '*.gz' -exec gunzip {} +

echo "Extraction and decompression complete."





