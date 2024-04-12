import xarray as xr
from glob import glob
import numpy as np
import sys
from datetime import datetime
import os
import pyproj
import matplotlib.colors as colors

#import own methods
sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.plot_functions import plot_cot
from process.process_functions import regrid_data
from compare.comparison_function import get_max_min
from process.config_process import lat_max, lon_max, lat_min, lon_min
from process.config_process import reg_lats, reg_lons

#path for images
path_fig = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/Fig/"

#get filenames for all cma
path_cot = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/cloud_products/CPP_CMSAF/"
cot_filepattern = 'CPPin*405SVMSGI1UD.nc'
fnames_cot = sorted(glob(path_cot+cot_filepattern))
output_folder = path_cot+'Processed/'

#get filenames for all msg parallac corrected  files
path_msg = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/Parallax_Corrected/"
msg_filepattern = "MSG4-SEVI-MSG15-0100-NA-*.nc"
fnames_msg = sorted(glob(path_msg+msg_filepattern))

#get regular grids
lat_points = np.load(reg_lats)
lon_points = np.load(reg_lons)
msg_lat_grid_reg, msg_lon_grid_reg = np.meshgrid(lat_points,lon_points,indexing='ij')

#n files
if len(fnames_cot) == len(fnames_msg):
    print('CMA and MSG have same number of files')
    n_files = len(fnames_cot)
    print(n_files)
else:
    print('something went wrong, number of files mismatch')
    exit()

#check data integrity
ds_cot_all = xr.open_mfdataset(fnames_cot, combine='nested', concat_dim='time', parallel=True)
print('n of tot points',len(ds_cot_all['cot'].values.flatten()))
print('n of non Nan points',sum(~np.isnan(ds_cot_all['cot'].values.flatten()))) #40% NAN!-->not availble during nighttime
vmin , vmax = get_max_min(ds_cot_all,'cot')
print('min and max', vmin,vmax)


for n in range(n_files):
    if n>=0:
        #open datasets
        ds_cot = xr.open_dataset(fnames_cot[n])
        #print(ds_cot.variables)
        ds_msg = xr.open_dataset(fnames_msg[n])
        
        #get time
        #CPPin20210713213000405SVMSGI1UD.nc
        time = fnames_cot[n].split('/')[-1].split('.')[0][5:19]
        #time = str(ds_msg['end_time'].values[0]).split('.')[0]
        time = datetime.strptime(time, "%Y%m%d%H%M%S")
        readable_time = time.strftime('%Y-%m-%d, %H:%M')
        print(readable_time)

        # Extract variables
        time = ds_cot['time'].values[0]
        x = ds_cot['x'].values #zonal angles satellite and point of measurement (rad)
        y = ds_cot['y'].values #meridional angle ''
        h = ds_cot['subsatellite_alt'].values.squeeze() #satellite hight
        cot = ds_cot['cot'].values.squeeze() 
        #print('COT', np.shape(cot), cot)
        #print("x: ", np.shape(x), x)
        #print("y: ", np.shape(y), y)
        #print('h', h)
        #print('time', time)

        ### Project coordinates in a lat/lon grid ###

        # Define the source projection (geostationary) with parameters from your file. TODO Get parameters from metadata
        geos_proj = pyproj.Proj(proj='geos', h=h, lon_0=0.0, sweep='y', a=6378169.0, b=6356583.8)

        # Define the target projection as WGS84 (latitude and longitude in degrees)
        wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')

        # Create a transformer
        transformer = pyproj.Transformer.from_proj(geos_proj, wgs84_proj, always_xy=True)

        # Create a mesh grid
        x_mesh, y_mesh = np.meshgrid(x, y)

        # Convert x and y from radians to meters using the satellite height
        # Note: Perform this operation directly on the mesh grid
        x_meters_mesh = x_mesh * h
        y_meters_mesh = y_mesh * h

        # Flatten the mesh grid arrays to 1D arrays for transformation
        x_meters_flat = x_meters_mesh.flatten()
        y_meters_flat = y_meters_mesh.flatten()

        # Transform coordinates to lat/lon
        lon_flat, lat_flat = transformer.transform(x_meters_flat, y_meters_flat)

        # Now lon_flat and lat_flat are 1D arrays; you can reshape them back to 2D if needed
        lon = lon_flat.reshape(x_mesh.shape)
        lat = lat_flat.reshape(y_mesh.shape)

        # Print your results as needed
        #print("Transformed Longitude (degrees East):", np.shape(lon), lon)
        #print("Transformed Latitude (degrees North):", np.shape(lat), lat)

        # Mask NaN values and regrid Data
        cot_flat = cot.flatten()
        #print(len(cot_flat))
        valid = ~np.isnan(cot_flat)
        print('\nvalid point',sum(valid))
        if sum(valid)>0:
            cot_valid = cot_flat[valid]
            lon_valid = lon_flat[valid]
            lat_valid = lat_flat[valid]

            cot_regrid = regrid_data(lat_valid,lon_valid,cot_valid,msg_lat_grid_reg,msg_lon_grid_reg,'linear') 
        else:
            cot_regrid = np.full(msg_lat_grid_reg.shape, np.nan)
        #print('\nregridded cot',np.shape(cot_regrid), cot_regrid)

        #plot map
        cmap = 'plasma'
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        plot_cot(cot_regrid,msg_lat_grid_reg,msg_lon_grid_reg,time,'COT',[lon_min,lon_max,lat_min,lat_max],cmap,norm,path_fig+'COT/')

        
        #save the processed COT

        #create an empty xarray Dataset 
        ds = xr.Dataset()
        cot_da = xr.DataArray(
            cot_regrid,
            dims=("y", "x"),
            coords={"lat": ("y", lat_points), "lon": ("x", lon_points)},
            name='cot')

        # combine DataArrays of rain rate into xarray object
        ds["cot"] = cot_da
        
        # Add a new dimension for the start time coordinate
        ds = ds.expand_dims('time', axis=0)
        ds['time'] = [time]
        #print(ds)

        # Check if the directory exists
        if not os.path.exists(output_folder):
             # Create the directory if it doesn't exist
             os.makedirs(output_folder)

        # Define a new filename for saving regridded data
        save_filename = os.path.join(output_folder, os.path.basename(fnames_cot[n]))

        # Save in netCDF format
        ds.to_netcdf(save_filename)
        print('product saved to', save_filename,'\n')
