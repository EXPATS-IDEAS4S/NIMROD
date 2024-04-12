import xarray as xr
from glob import glob
import numpy as np
import sys
from datetime import datetime
import os

#import own methods
sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.plot_functions import plot_cma
from process.process_functions import regrid_data
from process.config_process import lat_max, lon_max, lat_min, lon_min
from process.config_process import reg_lats, reg_lons

#path for images
path_fig = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/Fig/"

#get filenames for all cma
path_cma = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/cloud_products/CMA_NWCSAF/"
cma_filepattern = 'S_NWC_CMA_MSG4_FLOOD-GER-2021-VISIR_*.nc'
fnames_cma = sorted(glob(path_cma+cma_filepattern))
output_folder = path_cma+'Processed/'

#get filenames for all msg parallac corrected  files
path_msg = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/Parallax_Corrected/"
msg_filepattern = "MSG4-SEVI-MSG15-0100-NA-*.nc"
fnames_msg = sorted(glob(path_msg+msg_filepattern))

#get regular grids
lat_points = np.load(reg_lats)
lon_points = np.load(reg_lons)
msg_lat_grid_reg, msg_lon_grid_reg = np.meshgrid(lat_points,lon_points,indexing='ij')

#n files
if len(fnames_cma) == len(fnames_msg):
    print('CMA and MSG have same number of files')
    n_files = len(fnames_cma)
    print(n_files)
else:
    print('something went wrong, number of files mismatch')
    exit()


for n in range(n_files):
    if n>=0:
        #open datasets
        ds_cma = xr.open_dataset(fnames_cma[n])['cma']
        ds_msg = xr.open_dataset(fnames_msg[n])
        
        #get time
        time = str(ds_msg['end_time'].values[0]).split('.')[0]
        print(time)
        time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

        #check shape and order of lat/lon grids
        lon_cma = np.flip(ds_cma['lon'].values)
        lon_msg = ds_msg['lon_grid'].values.squeeze()
        #print(lon_cma,'\n', lon_msg)
        lat_cma = np.flip(ds_cma['lat'].values)
        lat_msg = ds_msg['lat_grid'].values.squeeze()
        #print(lat_cma, '\n', lat_msg)

        #adapt cma matrix to the shape of lat/lon grid from parallax corrected MSG
        cma = np.flip(ds_cma.values)

        #plot CMA, apply msg grif to CMA being caruful if the points correspond
        #plot_cma(cma,lat_cma,lon_cma,time,'CMA',[lon_min,lon_max,lat_min,lat_max],path_fig ) 
        #plot_cma(cma,lat_msg,lon_msg,time,'CMA-parallax',[lon_min,lon_max,lat_min,lat_max],path_fig )  
        
        #regrid to a regular grid (NN method to maintain binary form)
        cma_regrid = regrid_data(lat_msg.flatten(),lon_msg.flatten(),cma.flatten(),msg_lat_grid_reg, msg_lon_grid_reg)
        #plot_cma(cma_regrid,msg_lat_grid_reg,msg_lon_grid_reg,time,'CMA-regrid',[lon_min,lon_max,lat_min,lat_max],path_fig ) 
        
        #save the processed CMA

        #create an empty xarray Dataset 
        ds = xr.Dataset()
        cma_da = xr.DataArray(
            cma_regrid,
            dims=("y", "x"),
            coords={"lat": ("y", lat_points), "lon": ("x", lon_points)},
            name='cma')

        # combine DataArrays of rain rate into xarray object
        ds["cma"] = cma_da
        
        # Add a new dimension for the start time coordinate
        ds = ds.expand_dims('time', axis=0)
        ds['time'] = [time]
        #print(ds)

        # Check if the directory exists
        if not os.path.exists(output_folder):
             # Create the directory if it doesn't exist
             os.makedirs(output_folder)

        # Define a new filename for saving regridded data
        save_filename = os.path.join(output_folder, os.path.basename(fnames_cma[n]))

        # Save in netCDF format
        ds.to_netcdf(save_filename)
        print('product saved to', save_filename,'\n')

