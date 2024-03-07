"""
Match the Nimrod data to MSG grid

@author: Daniele Corradini
"""
import numpy as np
import os
import sys
from glob import glob
import xarray as xr

#import own methods
sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.plot_functions import plot_rain_data
from readers.read_functions import read_msg_lat_lon, read_radar_data_with_lat_lon
from process_functions import crop_radar_data, regrid_data, check_grid

#import variables from config file
from config_process import msg_folder, reg_lats, reg_lons, msg_filepattern
from config_process import radar_folder, output_folder, radar_filepattern, fig_folder
from config_process import ncols_radar, nrows_radar, lon_max, lon_min, lat_max, lat_min
from config_process import save, regular_grid

extent = [lon_min, lon_max, lat_min, lat_max] #[left, right, bottom ,top]

# Read MSG lat/lon data
if regular_grid:
    lat_points = np.load(reg_lats)
    lon_points = np.load(reg_lons)
    msg_lat_grid, msg_lon_grid = np.meshgrid(lat_points,lon_points,indexing='ij')
else:  
    fnames_msg = glob(msg_folder+msg_filepattern)  
    msg_lat, msg_lon = read_msg_lat_lon(fnames_msg[0])
    msg_lon_grid, msg_lat_grid = np.flip(msg_lon[0,:,:]), np.flip(msg_lat[0,:,:]) #TODO is this needed?, maybe not becasue the grid function will set at MSG grid anyway

#check MSG grid
#check_grid(msg_lat_grid,msg_lon_grid,None,'MSG')    

#check the format of the radar rain data
radar_files = sorted(glob(radar_folder+radar_filepattern))
#print(radar_files)
time, radar_lat, radar_lon, rain_rate = read_radar_data_with_lat_lon(radar_files[0])
#print('time',np.shape(time),time)
#check_grid(radar_lat,radar_lon,rain_rate,'radar flat')

# reshape radar dat to the grid
rain_rate_grid = rain_rate.reshape(nrows_radar, ncols_radar)
radar_lon_grid = radar_lon.reshape(nrows_radar, ncols_radar)
radar_lat_grid = radar_lat.reshape(nrows_radar, ncols_radar)
#check_grid(radar_lat_grid,radar_lon_grid,rain_rate_grid,'radar grid')

#check with a plot before and after the crop
plot_rain_data(rain_rate_grid,radar_lat_grid,radar_lon_grid, time, 'original_grid', extent, fig_folder)

# #define area of work for the project for cropping the data 
# cropped_data, cropped_lat, cropped_lon = crop_radar_data(rain_rate_grid, radar_lat_grid, radar_lon_grid, lat_min, lat_max, lon_min, lon_max)
#check_grid(cropped_lat, cropped_lon, cropped_data, 'cropped radar')

# #check with a plot before and after the crop
# plot_data(cropped_data, cropped_lat, cropped_lon, time, 'orginial_grid_cropped', False)


# Loop through each radar file
for radar_file in radar_files:
    #radar_file = radar_folder+'nimrod_rain_data_eu_20210714_1200.nc'
    #full_path = os.path.join(radar_folder, radar_file)
    time, radar_lat, radar_lon, rain_rate = read_radar_data_with_lat_lon(radar_file)
    print(time)
    #print(radar_lat, radar_lon, rain_rate, msg_lat_grid, msg_lon_grid )

    # Regrid the data
    regridded_data = regrid_data(radar_lat, radar_lon, rain_rate, msg_lat_grid, msg_lon_grid)
    #check_grid(msg_lat_grid,msg_lon_grid,regridded_data,'radar regrid')

    #plot the regrid
    plot_rain_data(regridded_data, msg_lat_grid, msg_lon_grid, time, 'regridded', extent, fig_folder)
    #exit()

    if save:
        #create an empty xarray Dataset 
        ds = xr.Dataset()

        if regular_grid:
            rain_da = xr.DataArray(
                regridded_data,
                dims=("y", "x"),
                coords={"lat": ("y", lat_points), "lon": ("x", lon_points)},
                name='rain_rate')
        else:
            lon_da = xr.DataArray(msg_lon_grid, dims=("y", "x"), name="lon_grid")
            lat_da = xr.DataArray(msg_lat_grid, dims=("y", "x"), name="lat_grid")
            rain_da = xr.DataArray(regridded_data, dims=("y", "x"), name="rain_rate")

            # combine DataArrays of grids into xarray object
            ds["lon_grid"] = lon_da
            ds["lat_grid"] = lat_da
        
        # combine DataArrays of rain rate into xarray object
        ds["rain_rate"] = rain_da
        
        # Add a new dimension for the start time coordinate
        ds = ds.expand_dims('time', axis=0)
        ds['time'] = [time]

        # Check if the directory exists
        if not os.path.exists(output_folder):
             # Create the directory if it doesn't exist
             os.makedirs(output_folder)

        # Define a new filename for saving regridded data
        save_filename = os.path.join(output_folder, 'regridded_' + os.path.basename(radar_file))

        # Save in netCDF format
        ds.to_netcdf(save_filename)
        print('product saved to', save_filename,'\n')

