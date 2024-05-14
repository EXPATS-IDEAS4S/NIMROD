import os
import tarfile
import gzip
import shutil
from glob import glob

month = '04'
year = '2023'

# Path to the tar file
tar_path = "/data/sat/msg/radar/nimrod/"+year+"/"+month+"/"


# Construct the path pattern to include subdirectories for days
file_pattern = "metoffice-c-band-rain-radar_europe_*_5km-composite.dat.gz.tar"

#open all MSG files in directory 
fnames = sorted(glob(tar_path+file_pattern))
print(fnames)

# Destination directory for extracted files
dest_dir = "/data/sat/msg/radar/nimrod/netcdf/"+year+"/"+month+"/"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

for fname in fnames:
    # Extract the tar archive
    print(f"Extracting files from {fname}...")
    with tarfile.open(fname, "r") as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete.")

    # Decompress any .gz files in the extracted files
    print("Decompressing .gz files...")
    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if file.endswith(".gz"):
                gz_path = os.path.join(root, file)
                decompressed_path = os.path.splitext(gz_path)[0]  # Remove .gz suffix
                try:
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(decompressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(gz_path)  # Optionally remove the .gz file after decompression
                    print(f"Decompressed {gz_path} to {decompressed_path}")
                except Exception as e:
                    print(f"Failed to decompress {gz_path}: {e}")

print("All operations complete.")







