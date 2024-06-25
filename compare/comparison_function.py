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
from sklearn.metrics import confusion_matrix 




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
    if len(ch_values)>0:
        max = np.amax(ch_values)
        min = np.amin(ch_values)
    else:
        max = np.nan
        min = np.nan

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


def mask_nighttime_single_step(ds, time_var_name, vis_channels, start_hour, end_hour):
    """
    Masks (replaces with NaN) the values in VIS channels of an xarray Dataset if the time
    of the dataset falls outside the specified daytime hours range. This function is designed
    for datasets containing only one time step.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing MSG channel values with a 'time' dimension.
    time_var_name : str
        The name of the coordinate time, e.g 'time' or 'end_time'
    vis_channels : list of str
        List of the visible channels names.
    start_hour : int
        The beginning of the daytime period in 24-hour format (e.g., 7 for 7 AM).
    end_hour : int
        The end of the daytime period in 24-hour format (e.g., 18 for 6 PM).

    Returns
    -------
    xarray.Dataset
        The dataset with values outside of the specified daytime hours replaced with NaN
        across all VIS channels if the dataset's time falls outside the daytime hours.
    """
    # Convert start_hour and end_hour to integers
    start_hour = int(start_hour)
    end_hour = int(end_hour)

    # Convert the 'time' coordinate to a pandas DatetimeIndex to extract the hour
    dataset_time = pd.to_datetime(ds[time_var_name].values)

    # Check if the dataset time falls outside the specified daytime hours
    if not (start_hour <= dataset_time.hour <= end_hour):
        # Loop through all data variables in the dataset
        for var in ds.data_vars:
            if var in vis_channels:
                # Replace values with NaN
                ds[var][:, ...] = np.nan

    return ds


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


def filter_rain_rate(rain_rate_ds, min_threshold, max_threshold):
    """
    Filters the input xarray.Dataset containing rain rate data to include only data points
    where the rain rate is within the specified minimum and maximum thresholds, inclusive.
    
    Parameters:
    - rain_rate_ds (xarray.Dataset): The dataset containing the rain rate data. 
      It is expected to have a variable named 'rain_rate' representing the rain rate values.
    - min_threshold (float): The minimum rain rate value to include.
    - max_threshold (float): The maximum rain rate value to include.
    
    Returns:
    - xarray.Dataset: A dataset with the rain rate data filtered according to the specified thresholds.
    """
    # Ensure that the input thresholds are valid
    if min_threshold > max_threshold:
        raise ValueError("The minimum threshold cannot be greater than the maximum threshold.")

    # Create a mask where the rain rate is between the minimum and maximum thresholds (inclusive)
    mask = (rain_rate_ds['rain_rate'] >= min_threshold) & (rain_rate_ds['rain_rate'] <= max_threshold)
    
    # Apply this mask to the rain rate dataset, keeping only the data points that match the condition
    filtered_rain_rate_ds = rain_rate_ds.where(mask)
    
    return filtered_rain_rate_ds




def filter_elevation(msg_ds, rain_ds, oro_ds, elev_class):
    """
    Filters datasets based on elevation ranges and returns the filtered rain and MSG datasets.
    
    Parameters:
    - msg_ds: xarray.Dataset containing MSG channel data.
    - rain_ds: xarray.Dataset containing the rain rate data.
    - oro_ds: xarray.Dataset containing orography (elevation) data.
    - elev_class: tuple or list with two elements specifying the minimum and maximum elevation
                  range to include (min_elevation, max_elevation).
    
    Returns:
    - filtered_rain_ds: xarray.Dataset with rain data filtered based on the elevation range.
    - filtered_msg_ds: xarray.Dataset with MSG data filtered based on the same elevation range.
    """
    # Ensure elevation range is specified correctly
    if not isinstance(elev_class, (list, tuple)) or len(elev_class) != 2:
        raise ValueError("elev_class must be a list or tuple with two elements (min, max elevation).")

    # Create a mask where the elevation is within the specified range
    mask = (oro_ds['orography'] > elev_class[0]) & (oro_ds['orography'] <= elev_class[1])
    
    # Apply this mask to the rain dataset
    filtered_rain_ds = rain_ds.where(mask)

    # Apply the same mask to the MSG dataset. Assuming all variables within msg_ds should be filtered.
    filtered_msg_ds = msg_ds.where(mask)
    
    return filtered_rain_ds, filtered_msg_ds



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


def generate_cloud_classes(boundaries):
    """
    Generate cloud classes as a list of tuples based on given boundaries.
    
    Parameters:
    - boundaries (list): A list of numeric boundary values for the cloud classes. 
      Each boundary value defines the start of a new class, except for the last value, 
      which defines the end of the last class.
      
    Returns:
    - cloud_classes (list of tuples): A list of tuples representing the cloud classes,
      where each tuple is (lower_bound, upper_bound) of a class.
    """
    cloud_classes = []
    for i in range(len(boundaries) - 1):
        cloud_classes.append((boundaries[i], boundaries[i + 1]))
    return cloud_classes




def calc_distribution_stats_by_rain_classes(rain_rate_ds, msg_ds, path_out, channels, rain_classes, rain_classes_name):
    """
    Calculates distribution statistics for various MSG channels across different rain classes.
    """
    stats_results = []
    
    for i, ch in enumerate(channels):
        ch_stats = {"channel": ch}
        for n, rain_class in enumerate(rain_classes):
            class_mask = (rain_rate_ds['rain_rate'] >= rain_class[0]) & (rain_rate_ds['rain_rate'] < rain_class[1])
            class_data = msg_ds[ch].where(class_mask).values.flatten()
            class_data = class_data[~np.isnan(class_data)]

            if len(class_data) > 0:
                median = np.median(class_data)
                q1 = np.percentile(class_data, 25)
                q3 = np.percentile(class_data, 75)

                ch_stats[rain_classes_name[n]] = {"median": median, "q1": q1, "q3": q3}

        stats_results.append(ch_stats)

    # Save stats to a text file
    with open(path_out + 'channel_stats.txt', 'w') as file:
        for result in stats_results:
            file.write(f"Channel: {result['channel']}\n")
            for rain_class in rain_classes_name:
                if rain_class in result:
                    file.write(f"{rain_class}: Median = {result[rain_class]['median']}, Q1 = {result[rain_class]['q1']}, Q3 = {result[rain_class]['q3']}\n")
            file.write("\n")


def calculate_categorical_metrics(y_true, y_pred, labels):
    """
    Calculate categorical metrics for a classification model.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: A dictionary containing calculated categorical metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    true_positive = cm[1, 1] #hits (H)
    false_positive = cm[0, 1] # false alarms (F)
    false_negative = cm[1, 0] #misses (M)
    true_negative = cm[0, 0] # correct rejections (C)
    
    total_samples = true_positive + false_positive + false_negative + true_negative
    
    probability_of_detection = true_positive/(true_positive+false_negative) #H/(H+M)
    false_alarm_ratio = false_positive / (false_positive + true_positive) #or false positive rate #error in Giacomo paper?
    multiplicative_bias = (true_positive + false_positive) / (true_positive + false_negative)
    critical_success_index = true_positive / (true_positive + false_positive + false_negative)
    
    # Calculate the expected value by chance
    expected_by_chance = (true_positive + false_positive) * (true_positive + false_negative) / total_samples
    equitable_threat_score = (true_positive - expected_by_chance) / (true_positive + false_positive + false_negative - expected_by_chance)
    
    return {
        'Probability of Detection': probability_of_detection,
        'False Alarm Ratio': false_alarm_ratio,
        'Multiplicative Bias': multiplicative_bias,
        'Critical Success Index': critical_success_index,
        'Equitable Threat Score': equitable_threat_score
    }


def calculate_ets_across_thresholds(msg_ds, channel_name,vis_channels,ir_channels, y_true, thresholds, rain_threshold, labels=[0, 1]):
    ets_results = []

    for threshold in thresholds:
        print('\n channel threshold: ', threshold)
        # Generate y_pred based on the current threshold for the channel
        if channel_name in vis_channels:
            y_pred = np.where(msg_ds[channel_name] >= threshold, 1, 0)
        elif channel_name in ir_channels:
            y_pred = np.where(msg_ds[channel_name] > threshold, 0, 1)
        else:
            print('wrong channel name!')
        y_pred = y_pred.flatten()

        #find mask that exclude nan values for both arrays
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        #print('y pred ',len(y_pred))
        #print('y true ', len(y_true))

        # Calculate metrics
        metrics = calculate_categorical_metrics(y_true, y_pred, labels=labels)
        ets = metrics['Equitable Threat Score']
        
        # Append the threshold and ETS to the results
        ets_results.append((rain_threshold, channel_name, threshold, ets))
    
    # Convert results to a DataFrame and save to CSV
    df_ets = pd.DataFrame(ets_results, columns=['Rain_Threshold','Channel','MSG_Threshold', 'ETS'])

    return df_ets


def save_ets_thresholds_channels_trends(msg_ds, rain_ds, vis_channels, ir_channels, rain_thresholds, elev_class, threshold_n_bin, path_out):
    # definel all list of channels
    channels = vis_channels+ir_channels

    # Define the DataFrame with the columns you'll need
    df = pd.DataFrame(columns=['Rain_Threshold','Channel','MSG_Threshold','ETS'])

    #loop over the different rain classifications
    for rain_threshold in rain_thresholds:
        print('\nrain threshold: ', rain_threshold)
        if rain_threshold!=0.1:
            rain_max = get_max_min(rain_ds,'rain_rate')[1]
            rain_ds = filter_rain_rate(rain_ds,0.1,rain_max)
        
        #generate y_true on the current threshold for the rain rate
        y_true = np.where(rain_ds['rain_rate']>= rain_threshold, 1, 0)
        y_true = y_true.flatten()

        #read the medians of the rain classes distributions
        #msg_intervals = extract_medians(path_out+'channel_stats.txt',channels,rain_classes[n])
        

        for channel in channels:
            print('\nchannel: ', channel)
            #find interval for the threshold
            msg_min, msg_max = get_max_min(msg_ds,channel)

            #define the threshold
            thresholds = np.linspace(msg_min, msg_max, threshold_n_bin)

            #calculate the ets for every threshold
            df_ets = calculate_ets_across_thresholds(msg_ds,channel,vis_channels,ir_channels,y_true,thresholds, rain_threshold)

            df = pd.concat([df, df_ets], ignore_index=True)

    #save dataFrame
    df.to_csv(path_out+"ets_results_"+elev_class+".csv", index=False)



def extract_medians(file_path, channels, rain_classes):
    medians = []
    current_channel = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'Channel:' in line:
                channel_name = line.split(': ')[1].strip()
                if channel_name in channels:
                    current_channel = channel_name
            elif current_channel and any(rain_class in line for rain_class in rain_classes):
                parts = line.split(':')
                rain_class = parts[0].strip()
                if rain_class in rain_classes:
                    stats = parts[1].strip().split(',')
                    median_value = float(stats[0].split('=')[1].strip())
                    if rain_class == rain_classes[0]:
                        medians.append((median_value,))
                    else:
                        medians[-1] += (median_value,)
    
    return medians


def select_max_ets_thresholds(input_csv, output_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Group by 'Rain_Threshold' and 'Channel', then get the row with the max ETS for each group
    max_ets_rows = df.loc[df.groupby(['Rain_Threshold', 'Channel'])['ETS'].idxmax()]
    
    # Save the results to another CSV file
    max_ets_rows.to_csv(output_csv, index=False)


def save_metrics_thresholds_channels(msg_ds, rain_ds, vis_channels, ir_channels, rain_thresholds, elev_class, path_out):
    # definel all list of channels
    channels = vis_channels+ir_channels

    # Define array to store metrics
    results = []

    #loop over the different rain classifications
    for rain_threshold in rain_thresholds:
        print('\nrain threshold: ', rain_threshold)
        if rain_threshold!=0.1:
            rain_max = get_max_min(rain_ds,'rain_rate')[1]
            rain_ds = filter_rain_rate(rain_ds,0.1,rain_max)
        
        #generate y_true on the current threshold for the rain rate
        y_true = np.where(rain_ds['rain_rate']>= rain_threshold, 1, 0)
        y_true = y_true.flatten()       

        for channel in channels:
            print('\nchannel: ', channel)
            #define the threshold
            msg_threshold = find_max_ets_threshold(path_out+"ets_results_"+elev_class+".csv",channel,rain_threshold)

            # Generate y_pred based on the current threshold for the channel
            if channel in vis_channels:
                y_pred = np.where(msg_ds[channel] >= msg_threshold, 1, 0)
            elif channel in ir_channels:
                y_pred = np.where(msg_ds[channel] > msg_threshold, 0, 1)
            else:
                print('wrong channel name!')
            y_pred = y_pred.flatten()

            #find mask that exclude nan values for both arrays
            mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            #print('y pred ',len(y_pred))
            #print('y true ', len(y_true))

            # Calculate metrics
            metrics = calculate_categorical_metrics(y_true, y_pred, labels=[0,1])
            pod = metrics['Probability of Detection']
            far = metrics['False Alarm Ratio'] 
            bias = metrics['Multiplicative Bias']
            csi = metrics['Critical Success Index'] 
            ets = metrics['Equitable Threat Score']
            
            # Append the threshold and ETS to the results
            results.append((rain_threshold, channel, msg_threshold, pod, far, bias, csi, ets))
        
    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results, columns=['Rain_Threshold','Channel','MSG_Threshold', 'POD','FAR', 'BIAS','CSI','ETS'])     

    #save dataFrame
    df.to_csv(path_out+"dichotomy_metrics_results_"+elev_class+".csv", index=False)



def find_max_ets_threshold(input_csv, channel, rain_threshold):
    # Load the CSV
    df = pd.read_csv(input_csv)
    
    # Filter the DataFrame for the given channel and rain threshold
    filtered_df = df[(df['Channel'] == channel) & (df['Rain_Threshold'] == rain_threshold)]
    
    # Find the row with the maximum ETS
    max_ets_row = filtered_df[filtered_df['ETS'] == filtered_df['ETS'].max()]
    
    # If multiple rows have the same max ETS, take the first one
    if not max_ets_row.empty:
        msg_threshold = max_ets_row.iloc[0]['MSG_Threshold']
        return msg_threshold
    else:
        print('Max ETS not found!')
        return None
    


def save_spatial_metrics(msg_ds,rain_ds, rain_threshold, vis_channels, ir_channels, path_out):
    """
    Plot the specified metric of spatial data over time on a geographical map.

    Parameters:
    - data (xarray.DataArray): The spatial data (must include lat and lon dimensions).
    - metric (str): The metric to calculate and plot.
    - time_step (str or None): Specific time step to plot. If None, uses all time steps.
    """
    channels = vis_channels+ir_channels

    #compute y true
    if rain_threshold!=0.1:
        rain_ds = filter_rain_rate(rain_ds)
        
    #generate y_true on the current threshold for the rain rate
    y_true = np.where(rain_ds['rain_rate']>= rain_threshold, 1, 0)
    #y_true = y_true.flatten()  
    
    #get channels value
    for channel in channels:
        print('\nchannel: ', channel)
        #define the threshold
        msg_threshold = find_max_ets_threshold(path_out+"ets_results_whole_distribution.csv",channel,rain_threshold)

        # Generate y_pred based on the current threshold for the channel
        if channel in vis_channels:
            y_pred = np.where(msg_ds[channel] >= msg_threshold, 1, 0)
        elif channel in ir_channels:
            y_pred = np.where(msg_ds[channel] > msg_threshold, 0, 1)
        else:
            print('wrong channel name!')

        # Initialize an empty DataArray for ETS results
        results_pod = xr.full_like(msg_ds.isel(time=0), fill_value=np.nan)
        results_far = xr.full_like(msg_ds.isel(time=0), fill_value=np.nan)
        results_bias = xr.full_like(msg_ds.isel(time=0), fill_value=np.nan)
        results_csi = xr.full_like(msg_ds.isel(time=0), fill_value=np.nan)
        results_ets = xr.full_like(msg_ds.isel(time=0), fill_value=np.nan)

        # Loop through each spatial point
        for lat in msg_ds.lat:
            for lon in msg_ds.lon:
                obs_time_series = y_true.sel(lat=lat, lon=lon)
                forecast_time_series = y_pred.sel(lat=lat, lon=lon)

                #find mask that exclude nan values for both arrays
                mask = (~np.isnan(obs_time_series)) & (~np.isnan(forecast_time_series))
                obs_time_series = obs_time_series[mask]
                forecast_time_series = forecast_time_series[mask]

                # Calculate metric for the current pixel
                metrics = calculate_categorical_metrics(obs_time_series, forecast_time_series, labels=[0,1])
                results_pod.loc[dict(lat=lat, lon=lon)] = metrics['Probability of Detection']
                results_far.loc[dict(lat=lat, lon=lon)] =  metrics['False Alarm Ratio'] 
                results_bias.loc[dict(lat=lat, lon=lon)] = metrics['Multiplicative Bias']
                results_csi.loc[dict(lat=lat, lon=lon)] = metrics['Critical Success Index'] 
                results_ets.loc[dict(lat=lat, lon=lon)] = metrics['Equitable Threat Score']

    results_pod.to_netcdf(path_out+'pod_pixelwise_dichotomous_analysis.nc')
    results_far.to_netcdf(path_out+'far_pixelwise_dichotomous_analysis.nc')
    results_bias.to_netcdf(path_out+'bias_pixelwise_dichotomous_analysis.nc')
    results_csi.to_netcdf(path_out+'csi_pixelwise_dichotomous_analysis.nc')
    results_ets.to_netcdf(path_out+'ets_pixelwise_dichotomous_analysis.nc')
        

def save_spatial_metrics_to_dataset(msg_ds, rain_ds, rain_threshold, vis_channels, ir_channels, path_out):
    """
    Computes spatial metrics between MSG satellite channel data and observed rain rates, 
    saving the results in a netCDF file.

    This function processes each specified MSG channel against observed rain rates to 
    calculate several metrics, including Probability of Detection (POD), False Alarm Ratio (FAR), 
    Bias, Critical Success Index (CSI), and Equitable Threat Score (ETS). The calculations are 
    performed pixel-wise across the spatial extent of the datasets, excluding the time dimension. 
    The results are aggregated into a new xarray dataset mirroring the original MSG dataset's spatial 
    dimensions and saved as a netCDF file.

    Parameters:
    - msg_ds (xarray.Dataset): MSG dataset containing satellite channel data.
    - rain_ds (xarray.Dataset): Dataset containing observed rain rate data.
    - rain_threshold (float): Threshold for defining rain events in the observed data.
    - vis_channels (list): List of visible channel names to process.
    - ir_channels (list): List of infrared channel names to process.
    - path_out (str): Path to save the output netCDF file containing the metrics.

    Returns:
    - None: The function saves the results directly to a file and does not return any value.
    """
    channels = vis_channels + ir_channels
    metrics_names = ['pod', 'far', 'bias', 'csi', 'ets']

    # Initialize an empty dataset to hold results
    metrics_ds = xr.Dataset()

    #compute y true
    if rain_threshold!=0.1:
        rain_ds = filter_rain_rate(rain_ds)
        
    #generate y_true on the current threshold for the rain rate
    y_true = np.where(rain_ds['rain_rate']>= rain_threshold, 1, 0)
    
    for channel in channels:
        print('\nchannel: ', channel)
        #define the threshold
        msg_threshold = find_max_ets_threshold(path_out+"ets_results_whole_distribution.csv",channel,rain_threshold)

        # Generate y_pred based on the current threshold for the channel
        if channel in vis_channels:
            y_pred = np.where(msg_ds[channel] >= msg_threshold, 1, 0)
        elif channel in ir_channels:
            y_pred = np.where(msg_ds[channel] > msg_threshold, 0, 1)
        else:
            print('wrong channel name!')

        # Initialize dictionaries to hold metric results for current channel
        channel_metrics = {name: xr.full_like(msg_ds[channel].isel(time=0, drop=True), fill_value=np.nan) for name in metrics_names}
        #print(channel_metrics)
        # Loop to calculate metrics and assign to the appropriate DataArray
        lats = rain_ds.coords['lat'].values
        lons = rain_ds.coords['lon'].values
        for y,lat in enumerate(lats):
            for x,lon in enumerate(lons):
                #print('lat-lon:',lat,lon)
                # Calculate array to compare
                obs_time_series = y_true[:,y,x]
                pred_time_series = y_pred[:,y,x]
                #print(len(obs_time_series),obs_time_series)
                #print(len(pred_time_series),pred_time_series)

                #find mask that exclude nan values for both arrays
                mask = (~np.isnan(obs_time_series)) & (~np.isnan(pred_time_series))
                obs_time_series = obs_time_series[mask]
                pred_time_series = pred_time_series[mask]
                

                # Calculate metric for the current pixel
                metrics = calculate_categorical_metrics(obs_time_series, pred_time_series, labels=[0,1])
                
                # Example dummy metrics assignment
                metrics = {'pod': metrics['Probability of Detection'], 'far': metrics['False Alarm Ratio'], 
                           'bias':  metrics['Multiplicative Bias'], 'csi': metrics['Critical Success Index'], 
                           'ets': metrics['Equitable Threat Score']}
                
                for name in metrics_names:
                    #print(channel_metrics[name])
                    channel_metrics[name].loc[dict(x=x, y=y)] = metrics[name]
        
        # After processing all points, add the metrics DataArrays to the dataset under current channel name
        for name in metrics_names:
            metrics_ds[f'{channel}_{name}'] = channel_metrics[name]
    
    # Save the dataset to a netCDF file
    metrics_ds.to_netcdf(path_out + 'metrics_results_pixelwise_rain-th-'+str(rain_threshold)+'.nc')