"""
list of combination from Claudia Thesis
BTD 6.2-10.8 -> detect deep convective clouds --> higher more convection?
Ratio 0.6/1.6 -> distiguish ice and water clouds --> higher more rain
BTD 8.7-10.8 -> precence of clouds --> higher more deep clouds
BTD 10.8-12-0 -> cloud phase info --> lower more rain?
BTD 3.9-10.8 -> discenr cirrus and optical thick clouds --> lower thicked clouds

additional combinations from literature review of my thesis:
3.9-7.3 --> similar to 3.9-10.8
7.3-12 ---> separate convective clouds from non-precip cirrus -->
"""

import xarray as xr
import sys
import os
from glob import glob

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from compare.config_compare import msg_folder, msg_filepattern, channels

# List of all MSG data files
fnames_msg = sorted(glob(msg_folder + msg_filepattern))

#open cot data
path_cot = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/cloud_products/CPP_CMSAF/Processed/"
cot_filepattern = 'CPPin*405SVMSGI1UD.nc'

fnames_cot = sorted(glob(path_cot+cot_filepattern))

n_times = len(fnames_msg)

# Loop through the files to process each one
for t in range(n_times):
    # Open the MSG dataset
    msg_ds = xr.open_dataset(fnames_msg[t])
    msg_ds = msg_ds.rename({'end_time': 'time'})
    #print(msg_ds)
    print(msg_ds['time'].values[0])
    
    # Create a new dataset for combinations, copying coordinates from the original dataset
    combinations_ds = xr.Dataset(coords=msg_ds.coords)
    #print(combinations_ds)

    #open cot dataset
    cot_ds = xr.open_dataset(fnames_cot[t])
    #print(cot_ds)

    # Calculate combinations and add them to the new dataset
    combinations_ds['COT'] = cot_ds['ot']
    combinations_ds['VIS006:IR_016'] = msg_ds['VIS006'] / msg_ds['IR_016']
    combinations_ds['WV_062-IR_108'] = msg_ds['WV_062'] - msg_ds['IR_108']
    combinations_ds['IR_087-IR_108'] = msg_ds['IR_087'] - msg_ds['IR_108']
    combinations_ds['IR_108-IR_120'] = msg_ds['IR_108'] - msg_ds['IR_120']
    combinations_ds['IR_039-IR_108'] = msg_ds['IR_039'] - msg_ds['IR_108']
    combinations_ds['IR_039-WV_073'] = msg_ds['IR_039'] - msg_ds['WV_073']
    combinations_ds['WV_073-IR_120'] = msg_ds['WV_073'] - msg_ds['IR_120']

    # Define the new path and filename
    new_path = os.path.join(msg_folder, 'combination')
    os.makedirs(new_path, exist_ok=True)  # Ensure the directory exists

    new_filename = os.path.basename(fnames_msg[t])
    full_new_filename = os.path.join(new_path, new_filename)

    # Save the new dataset
    combinations_ds.to_netcdf(full_new_filename)
    #print(combinations_ds)

    # Close the original dataset
    msg_ds.close()
    cot_ds.close()

    #exit()

print("Process completed. Combination files saved.")
