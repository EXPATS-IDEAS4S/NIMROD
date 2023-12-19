import numpy as np
from netCDF4 import Dataset, num2date, date2num
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random


# Function to plot data with a custom colormap and Cartopy
def plot_data(rain_data, lat, lon, title, save):
    """
    rain_data (M,N), lat (M,N), lon (M,N)
    """
    # Create a custom colormap
    cmap = plt.cm.gist_ncar.copy()
    #cmap.set_bad('grey', 1.)  # Set NaNs to grey TODO check this
    #cmap.set_under('black', 1.)  # Set zeros to black TODO check this

    # Create a plot with Cartopy
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, edgecolor='white', linestyle=':')

    # Plot the data with the custom colormap
    norm = mcolors.PowerNorm(gamma=0.4)
    mesh = ax.contourf(lon, lat, rain_data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))
    
    # Add colorbar with reduced size
    cbar = plt.colorbar(mesh, label='Rain Rate (mm/h)', shrink=0.5)

    plt.title(f'Rain Rate Plot - '+title)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')

    # Save the plot
    if save:
        plot_filename = '/home/daniele/Documenti/PhD_Cologne/Case Studies/Germany_Flood_2021/rain_products/nimrod/images/rain_rate_'+title+'.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        plt.close()
    #else:
    #    plt.show()

    #plt.close()

# Function to read the lat/lon grid from the MSG file
def read_msg_lat_lon(msg_file):
    with Dataset(msg_file, 'r') as nc:
        lat = nc.variables['lat'][:] 
        lon = nc.variables['lon'][:] 
    return lat, lon

# Function to read radar data from a single time step NetCDF file
def read_radar_data_with_lat_lon(radar_file):
    with Dataset(radar_file, 'r') as nc:
        time_var = nc.variables['time']
        time = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

        # Read latitude and longitude data
        latitudes = nc.variables['latitude'][:]
        longitudes = nc.variables['longitude'][:]

        rain_rate = nc.variables['rain_rate'][:]
        # Since there's only one time step per file, we take the first (and only) element
        rain_rate = rain_rate[0, :]

    return time, latitudes, longitudes, rain_rate


def crop_radar_data(radar_data, radar_lat, radar_lon, lat_min, lat_max, lon_min, lon_max):
    # Assuming radar_lat and radar_lon are 2D arrays of the same shape as radar_data
    #Since the grid is irregular, the idea is to find the largest rectangular subset 
    #of the grid that fits entirely within the specified boundaries. 
    #This means you need to find the indices for the rows and columns where all values in 
    #those rows and columns fall within your lat/lon boundaries.

    # Find the rows and columns that are entirely within the lat/lon boundaries
    valid_rows = np.all((radar_lat >= lat_min) & (radar_lat <= lat_max), axis=1)
    valid_cols = np.all((radar_lon >= lon_min) & (radar_lon <= lon_max), axis=0)

    # Use these to crop the radar data
    cropped_radar_data = radar_data[valid_rows, :][:, valid_cols]

    # Crop the latitude and longitude arrays
    cropped_radar_lat = radar_lat[valid_rows, :][:, valid_cols]
    cropped_radar_lon = radar_lon[valid_rows, :][:, valid_cols]

    return cropped_radar_data, cropped_radar_lat, cropped_radar_lon




# Function to regrid the radar data to the MSG grid
def regrid_data(radar_lat, radar_lon, radar_data, msg_lat_grid, msg_lon_grid):
    # Source points (radar data) - already flattened
    radar_points = np.array([radar_lat, radar_lon]).T

    # Create a 2D mesh grid for the MSG data
    #msg_lon_grid, msg_lat_grid = np.meshgrid(msg_lon, msg_lat)
    msg_points = np.array([msg_lat_grid.flatten(), msg_lon_grid.flatten()]).T

    # Perform the regridding
    regridded_data = griddata(radar_points, radar_data, msg_points, method='linear')

    # Reshape the regridded data back to 2D (if necessary)
    regridded_data_reshaped = regridded_data.reshape(msg_lat_grid.shape)

    return regridded_data_reshaped

# File paths
msg_file = '/home/daniele/Documenti/PhD_Cologne/Case Studies/Germany_Flood_2021/MSG/HRSEVIRI_20220714_20210715_EXPATSdomain_proj/HRSEVIRI_20210714T120010Z_20210714T121243Z_epct_1297b012_PC.nc'
radar_folder = '/home/daniele/Documenti/PhD_Cologne/Case Studies/Germany_Flood_2021/rain_products/nimrod/nc_files'  # Folder containing radar files
output_folder = '/home/daniele/Documenti/PhD_Cologne/Case Studies/Germany_Flood_2021/rain_products/nimrod/nc_files_RegridToMSG/'

# Read MSG lat/lon data
msg_lat, msg_lon = read_msg_lat_lon(msg_file)
print(len(msg_lat),msg_lat)
print(len(msg_lon), msg_lon)
msg_lon_grid, msg_lat_grid = np.meshgrid(msg_lon, msg_lat)

#chech the format of the radar rain data
time, radar_lat, radar_lon, rain_rate = read_radar_data_with_lat_lon(radar_folder+'/nimrod_rain_data_eu_20210714_0000.nc')
print('time',np.shape(time),time)
print('radar_lat',np.shape(radar_lat), radar_lat)
print('radar_lon',np.shape(radar_lon), radar_lon)
print('rain_rate',np.shape(rain_rate),rain_rate)
# Count NaN values
nan_count = np.count_nonzero(np.isnan(rain_rate))
# Count values above zero
above_zero_count = np.count_nonzero(rain_rate > 0)
print(f"Number of NaN values: {nan_count}")
print(f"Number of values above zero: {above_zero_count}")

# List all radar files in the folder
radar_files = [f for f in os.listdir(radar_folder) if f.endswith('.nc')]

#number of columns and rows for radar data
ncols_radar = 620
nrows_radar = 700

# reshape radar dat to the grid
rain_rate_grid = rain_rate.reshape(nrows_radar, ncols_radar)
radar_lon_grid = radar_lon.reshape(nrows_radar, ncols_radar)
radar_lat_grid = radar_lat.reshape(nrows_radar, ncols_radar)
print('grid radar_lat',np.shape(radar_lat_grid), radar_lat_grid)
print('grid radar_lon',np.shape(radar_lon_grid), radar_lon_grid)
print('grid rain_rate',np.shape(rain_rate_grid),rain_rate_grid)

# define area of work for the project for cropping the data 
lon_min, lon_max, lat_min, lat_max = 5. , 9. , 48. , 52 #Germany Floods 2021 -->Different than EXPATS!
# cropped_data, cropped_lat, cropped_lon = crop_radar_data(rain_rate_grid, radar_lat_grid, radar_lon_grid, lat_min, lat_max, lon_min, lon_max)

# print('crop_radar_lat',np.shape(cropped_data), cropped_data)
# print('crop radar_lon',np.shape(cropped_lat), cropped_lat)
# print('crop rain_rate',np.shape(cropped_lon), cropped_lon)
# # Count NaN values
# nan_count = np.count_nonzero(np.isnan(cropped_data))
# # Count values above zero
# above_zero_count = np.count_nonzero(cropped_data > 0)
# print(f"Number of NaN values: {nan_count}")
# print(f"Number of values above zero: {above_zero_count}")

#check with a plot before and after the crop
plot_data(rain_rate_grid,radar_lat_grid,radar_lon_grid, 'original_grid', False)
#plot_data(cropped_data, cropped_lat, cropped_lon, 'orginial_grid_cropped', False)
plt.show()
plt.close()


# Loop through each radar file
for radar_file in radar_files:
    full_path = os.path.join(radar_folder, radar_file)
    time, radar_lat, radar_lon, rain_rate = read_radar_data_with_lat_lon(full_path)

    #crop data
    #cropped_data, cropped_lat, cropped_lon = crop_radar_data(rain_rate, radar_lat, radar_lon, lat_min, lat_max, lon_min, lon_max)

    # Regrid the data
    regridded_data = regrid_data(radar_lat, radar_lon, rain_rate, msg_lat_grid, msg_lon_grid)

    print('msg lat',np.shape(msg_lat))
    print('msg lon',np.shape(msg_lon))
    print('regrid radar',np.shape(regridded_data),regridded_data)
    # Count NaN values
    nan_count = np.count_nonzero(np.isnan(regridded_data))
    # Count values above zero
    above_zero_count = np.count_nonzero(regridded_data > 0)
    print(f"Number of NaN values: {nan_count}")
    print(f"Number of values above zero: {above_zero_count}")
    
    #plot the regrid
    plot_data(regridded_data, msg_lat_grid, msg_lon_grid, 'regridded', False)
    plt.show()
    plt.close()
    exit()

    #save in netcdf
    save=False

    if save:

        # Define a new filename for saving regridded data
        save_filename = os.path.join(output_folder, 'regridded_' + os.path.basename(radar_file))

        # Save the regridded data
        with Dataset(save_filename, 'w', format='NETCDF4') as nc:
            # Create dimensions
            nc.createDimension('time', 1)
            nc.createDimension('lat', len(msg_lat))
            nc.createDimension('lon', len(msg_lon))

            # Create variables
            times_nc = nc.createVariable('time', 'f4', ('time',))
            latitudes_nc = nc.createVariable('latitude', 'f4', ('lat',))
            longitudes_nc = nc.createVariable('longitude', 'f4', ('lon',))
            regridded_rain = nc.createVariable('regridded_rain_rate', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)

            # Assign data
            # Specify the units and calendar type for your time variable
            time_units = 'hours since 2000-01-01 00:00:00'
            calendar_type = 'gregorian'

            # Convert the cftime objects to numeric values
            numeric_time_values = date2num(time, units=time_units, calendar=calendar_type)

            # Now you can assign these numeric values to the NetCDF variable
            times_nc[:] = numeric_time_values
            #times_nc[:] = time
            latitudes_nc[:] = msg_lat
            longitudes_nc[:] = msg_lon
            regridded_rain[0, :, :] = regridded_data

        print(f"Regridded data saved as {save_filename}")

