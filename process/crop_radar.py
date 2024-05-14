import xarray as xr
import glob
import os


def crop_radar_files(radar_folder, file_pattern, lon_min, lon_max, lat_min, lat_max, output_folder=None):
    """
    Crop NetCDF radar files to a specified lat-lon extent and optionally save them.

    Parameters:
        radar_folder (str): Directory containing the radar files.
        file_pattern (str): Glob pattern to match the files.
        lon_min (float): Minimum longitude of the cropping area.
        lon_max (float): Maximum longitude of the cropping area.
        lat_min (float): Minimum latitude of the cropping area.
        lat_max (float): Maximum latitude of the cropping area.
        output_folder (str): Directory to save the cropped files. If None, files are not saved.

    Returns:
        list of xarray.Dataset: List of cropped datasets.
    """
    # Construct the full path for file pattern
    files = glob.glob(os.path.join(radar_folder, file_pattern))
    cropped_datasets = []
    
    for file in files:
        print(file)
        # Open the dataset
        ds = xr.open_dataset(file)
        
        # Crop the dataset using the latitude and longitude ranges
        cropped_ds = ds.where((ds.lon >= lon_min) & (ds.lon <= lon_max) & 
                              (ds.lat >= lat_min) & (ds.lat <= lat_max), drop=True)

        # Optionally save the cropped dataset
        if output_folder:
            output_filename = os.path.join(output_folder, os.path.basename(file))
            cropped_ds.to_netcdf(output_filename)
        
        cropped_datasets.append(cropped_ds)
        
    return cropped_datasets


radar_folder = '/data/sat/msg/radar/nimrod/netcdf/2023/04/'  
radar_filepattern = 'nimrod_rain_data_eu_*.nc'
output_folder = '/data/sat/msg/radar/nimrod/netcdf/2023/04_cropped/'  

lon_min, lon_max, lat_min, lat_max = 5., 16., 42., 51.5

# Crop and possibly save the files
cropped_data = crop_radar_files(radar_folder, radar_filepattern, lon_min, lon_max, lat_min, lat_max, output_folder)

print(cropped_data)