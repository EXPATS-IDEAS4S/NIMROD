"""
plotting function used
to compare MSG and NIMROD data
"""

#import libraries
import numpy as np
from netCDF4 import Dataset, num2date, date2num
import os
import imageio
import datetime
import xarray as xr
import pandas as pd


def get_max_min(ds, ch):
    """
    Calculate the minimum and maximum values of a specific channel within a dataset, 
    excluding any NaN values.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the channel from which to extract the values.
    ch : str
        The name of the channel (variable) within the dataset.

    Returns
    -------
    min : float
        The minimum value found in the channel, excluding NaN values.
    max : float
        The maximum value found in the channel, excluding NaN values.
    """

    ch_values = ds[ch][:]
    ch_values = ch_values.values.flatten()
    ch_values = ch_values[~np.isnan(ch_values)]
    max = np.amax(ch_values)
    min = np.amin(ch_values)

    return min, max


def select_daytime_files_from_hour(fnames_msg, start_hour, end_hour):
    """
    Filter a list of filenames by selecting only those that correspond to daytime hours,
    based on a specified start and end hour.

    This function assumes filenames contain a timestamp from which an hour can be extracted
    and compared against the specified daytime hours range.

    Parameters
    ----------
    fnames_msg : list of str
        The list of filenames to be filtered. Each filename must include a timestamp
        from which an hour can be extracted.
    start_hour : str
        The start hour of the daytime period, formatted as a 2-digit string (e.g., '07' for 7 AM).
    end_hour : str
        The end hour of the daytime period, formatted as a 2-digit string (e.g., '18' for 6 PM).

    Returns
    -------
    fnames_msg_day : list of str
        A filtered list of filenames that correspond only to the files within the specified
        daytime hours range.

    Notes
    -----
    The function assumes filenames follow a specific pattern where the timestamp is located
    after the last dash ('-') and before the first dot ('.') in the filename. The hour must be
    in the 9th and 10th position of the timestamp segment for proper extraction.
    """
    fnames_msg_day = []

    for file in fnames_msg:
        # Extracting the timestamp part from the filename
        time = file.split('/')[-1].split('-')[-1].split('.')[0]

        # Extracting the hour part from the timestamp
        hour = time[8:10]

        # Check if the hour is in daytime 
        if start_hour <= hour <= end_hour: 
            fnames_msg_day.append(file)
    
    return fnames_msg_day


def mask_nighttime_values(ds, time_var_name, vis_channels, start_hour, end_hour):
    """
    Masks (replaces with NaN) the values in VIS channels of an xarray Dataset for times 
    outside the specified daytime hours range.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing MSG channel values with a 'time' dimension.
    time_var_name : str
        The name of the coordinate time, e.g 'time' or 'end_time'
    vir_channels : list of str
        List of the visible channels names
    start_hour : int
        The beginning of the daytime period in 24-hour format (e.g., 7 for 7 AM).
    end_hour : int
        The end of the daytime period in 24-hour format (e.g., 18 for 6 PM).

    Returns
    -------
    xarray.Dataset
        The dataset with values outside of the specified daytime hours replaced with NaN
        across all channels.

    Notes
    -----
    This function assumes the dataset's 'time' coordinate is either a datetime64 object or 
    can be converted to one. It affects all variables in the dataset that depend on the 'time'
    dimension. Ensure that the 'time' coordinate accurately reflects the local time corresponding
    to the MSG observations for the start_hour and end_hour parameters to be applied correctly.
    """
    # Convert start_hour and end_hour to integers
    start_hour = int(start_hour)
    end_hour = int(end_hour)

    # Convert the 'time' coordinate to a pandas DatetimeIndex to extract the hour
    time_index = pd.DatetimeIndex(ds[time_var_name].values)

    # Identify the indices where the hour is outside the specified daytime range
    night_time_indices = np.where(~((time_index.hour >= start_hour) & (time_index.hour <= end_hour)))[0]

    # Loop through all data variables in the dataset
    for var in ds.data_vars:
        if var in vis_channels:
            # Check if 'time' is a dimension of the current variable
            if time_var_name in ds[var].dims:
                # Replace values at nighttime indices with NaN
                ds[var][:, ...][night_time_indices, ...] = np.nan

    return ds


def filter_rain_rate(rain_rate_ds, threshold=0.1):
    """
    Filters out points where the rain rate is below the specified threshold
    and applies the same filter to corresponding MSG channel data.
    
    Parameters:
    - rain_rate_ds: xarray.Dataset containing the rain rate data.
    - threshold: float, the minimum rain rate value to include.
    
    Returns:
    - filtered_rain_rate_ds: xarray.Dataset with rain rate data filtered.
    """
    # Create a mask where the rain rate is greater than or equal to the threshold
    mask = rain_rate_ds['rain_rate'] >= threshold
    
    # Apply this mask to the rain rate dataset
    filtered_rain_rate_ds = rain_rate_ds.where(mask)#, drop=True)
    
    # Apply the same mask to each channel in the MSG dataset
    # We iterate over all variables assuming they are the channels to be filtered
    #filtered_msg_ds = msg_ds.where(mask)#, drop=True)
    
    return filtered_rain_rate_ds


def filter_by_cloud_mask(msg_ds, cma_ds):
    """
    Filters out points where the cloud mask indicates clear sky (value 0)
    and applies the same filter to corresponding rain rate data and other MSG channel data.
    
    Parameters:
    - msg_ds: xarray.Dataset containing the MSG channel data.
    - cma_ds: xarray.Dataset containing the cloud mask data
    
    Returns:
    - filtered_msg_ds: xarray.Dataset with MSG data filtered based on cloud mask.
    """
    # Create a mask where the cloud mask indicates clouds (value not equal to 0)
    #print(cma_ds['cma'].values)
    mask = cma_ds['cma'] == 1
    
    # Apply this mask to the rain rate dataset
    #filtered_rain_rate_ds = rain_rate_ds.where(mask)#, drop=True)
    
    # Apply the same mask to the MSG dataset
    # Assuming all variables in the msg_ds should be filtered by the cloud mask
    filtered_msg_ds = msg_ds.where(mask)#, drop=True)
    
    return filtered_msg_ds


