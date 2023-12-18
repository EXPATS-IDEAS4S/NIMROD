import os
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyproj

#plot
plot = True

# Directory containing ASCII files and NC file
ascii_dir = '/home/daniele/Documenti/PhD_Cologne/Case Studies/Germany_Flood_2021/rain_products/nimrod/20210714_asc/'

# Folder to save the NetCDF file
output_folder = '/home/daniele/Documenti/PhD_Cologne/Case Studies/Germany_Flood_2021/rain_products/nimrod/'

# Ensure the output folder exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Domain boundaries in degrees
latmax = 62.7000
latmin = 33.0361
lonmax = 36.3671
lonmin = -23.4700

ncols = 620
nrows = 700

xtlcorner =  -23.404760360717773
ytlcorner = 62.688175201416016

#Polar Stereo
downward_long = 0 #-32767.0
std_lat = 60.0

cellsize = 5000.0

ps_proj = pyproj.Proj(proj='stere', lat_ts=std_lat, lat_0=-90, lon_0=downward_long)

# Convert the origin to projected coordinates
x_origin, y_origin = ps_proj(xtlcorner, ytlcorner)

# Generate grid coordinates
x_coords = x_origin + np.arange(ncols) * cellsize
y_coords = y_origin - np.arange(nrows) * cellsize  # Subtract because y decreases as you go down

# Create a meshgrid
xx, yy = np.meshgrid(x_coords, y_coords)

# Flatten the arrays to pass to pyproj
xx_flat = xx.flatten()
yy_flat = yy.flatten()

# Create the Polar Stereographic projection object
#ps_proj = pyproj.Proj(proj='stere', lat_ts=standard_latitude, lat_0=-90, lon_0=central_meridian)

# Perform the transformation
lons, lats = ps_proj(xx_flat, yy_flat, inverse=True)

# Reshape back to grid
lon_grid = lons.reshape(nrows, ncols)
lat_grid = lats.reshape(nrows, ncols)

# Convert grid coordinates back to lat/lon
#lons, lats = ps_proj(x_coords, y_coords, inverse=True)

# Function to read ASCII file and return data
def read_ascii_file(file_path):
    with open(file_path, 'r') as file:
        print(file)
        # Read header lines
        ncols = int(file.readline().split()[1]) #620 -lon
        #print('ncols',ncols)
        nrows = int(file.readline().split()[1]) #700 -lat
        #print('nrows', nrows)
        file.readline()  # Skip xllcorner and yllcorner lines
        file.readline()
        cellsize = float(file.readline().split()[1])
        #print('cellsize',cellsize)

        NODATA_value = (file.readline().split()[1])
        #print('nan',NODATA_value)

        # Read data
        data = np.loadtxt(file)

    # Identify indices where value is -0.0
    neg_zero_indices = np.bitwise_and(np.isclose(data, 0.0), np.signbit(data))

    # Replace -0.0 with NaN
    data[neg_zero_indices] = np.nan
    #print(data)

    # Replace NODATA values with NaN
    #data[data == NODATA_value] = np.nan
    return data

# Function to extract datetime from filename e.g. metoffice-c-band-rain-radar_europe_202107140000_5km-composite.asc
def extract_datetime(filename):
    date_str = filename.split('_')[2]
    return datetime.strptime(date_str, '%Y%m%d%H%M')
"""
# Convert meter interval to degrees
# One degree of latitude is approximately 111.32 km
lat_interval_deg = cellsize / 111320

# Length of a degree of longitude at 60 degrees latitude
deg_lon_at_60N = 111320 * np.cos(np.radians(60))
lon_interval_deg = cellsize / deg_lon_at_60N

# Create x y coords arrays
x_range = np.arange(xtlcorner, xtlcorner + (ncols)*lon_interval_deg, lon_interval_deg)
#y_range = np.arange(yllcorner, yllcorner + nrows*cellsize, cellsize)
y_range = np.arange(ytlcorner, ytlcorner - (nrows)*lat_interval_deg, -lat_interval_deg)

#create the x-y grid
lon_grid, lat_grid = np.meshgrid(x_range, y_range)  # Create a meshgrid for your grid coordinates

#define the start and the end projections
OSGB36 = pyproj.Proj("EPSG:32662") #EPSG:32662
WGS84 = pyproj.Proj("EPSG:32662")

# Transform the entire arrays
#lon_grid, lat_grid = pyproj.transform(OSGB36, WGS84, lon_grid, lat_grid, always_xy=True)
"""
#lon_grid, lat_grid = np.meshgrid(lons, lats)

print(lat_grid)
print(lon_grid)

# Initialize lists to store data and times
all_data = []
times = []

# Process each ASCII file and extract date from filename
ascii_files = [f for f in os.listdir(ascii_dir) if f.endswith('.asc')]
for file in ascii_files:
    file_path = os.path.join(ascii_dir, file)
    data = read_ascii_file(file_path)
    all_data.append(data)
    times.append(extract_datetime(file))

# Convert lists to arrays
all_data = np.array(all_data)
times = np.array(times)

#plot one time step for checking
if plot:
    # Choose a timestamp index (0 for the first timestamp)
    timestamp_index = 0

    # Extract the data for the chosen timestamp
    data_to_plot = all_data[timestamp_index, :, :]

    # Create a custom colormap
    cmap = plt.cm.gist_ncar.copy() #plt.cm.viridis.copy()
    cmap.set_bad('grey', 1.)  # Set NaNs to grey
    cmap.set_under('black', 1.)  # Set zeros to black

    # Create a plot with Cartopy
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, edgecolor='white', linestyle=':')

    # Plot the data with the custom colormap
    norm = mcolors.PowerNorm(gamma=0.4)
    mesh = plt.contourf(lon_grid, lat_grid, data_to_plot,cmap=cmap,norm=norm,transform=ccrs.PlateCarree(),extend='max',alpha=0.4,levels=np.linspace(0,100,500)) 
    # Add colorbar with reduced size
    cbar = plt.colorbar(mesh, label='Rain Rate (mm/h)', shrink=0.5)

    plt.title(f'Rain Rate on {times[timestamp_index].strftime("%Y-%m-%d %H:%M")}')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')

    # Save the plot
    plt.show()
    #plot_filename = f'{output_folder}/rain_rate_plot_{times[timestamp_index].strftime("%Y%m%d%H%M")}.png'
    #plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"Plot saved as {plot_filename}")

# Extract year, month, and day from one of the times (assuming all same day)
date_str = times[0].strftime('%Y%m%d')


"""
# Create NetCDF file with date in filename
netcdf_filename = f'{output_folder}/nimrod_rain_data_eu_{date_str}.nc'
with Dataset(netcdf_filename, 'w', format='NETCDF4') as nc:
    # Create dimensions
    nc.createDimension('time', None)
    nc.createDimension('lat', len(latitudes))
    nc.createDimension('lon', len(longitudes))

    # Create variables
    times_nc = nc.createVariable('time', 'f4', ('time',))
    latitudes_nc = nc.createVariable('lat', 'f4', ('lat',))
    longitudes_nc = nc.createVariable('lon', 'f4', ('lon',))
    rain = nc.createVariable('rain_rate', 'f4', ('time', 'lat', 'lon',), fill_value=np.nan)

    # Set units and other attributes as necessary
    times_nc.units = 'hours since 2000-01-01 00:00:00'
    times_nc.calendar = 'gregorian'
    latitudes_nc.units = 'degrees_north'
    longitudes_nc.units = 'degrees_east'

    # Assign data
    times_nc[:] = date2num(times, units=times_nc.units, calendar=times_nc.calendar)
    latitudes_nc[:] = latitudes
    longitudes_nc[:] = longitudes
    rain[:, :, :] = all_data

print(f"NetCDF file saved as {netcdf_filename}")
"""
