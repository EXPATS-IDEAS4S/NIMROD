"""
plotting function used
to MSG and NIMROD data
in the quality check
"""

#import libraries
import numpy as np
from netCDF4 import Dataset, num2date, date2num
import os
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import imageio
import datetime
import sys
import xarray as xr
import pandas as pd
from sklearn.metrics import confusion_matrix 
from matplotlib.patches import Patch
import re

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
#from readers.read_functions import 
from figures.plot_functions import set_map_plot, plot_rain_data, set_map_plot_nocolorbar
from compare.comparison_function import find_max_ets_threshold, get_max_min, calculate_categorical_metrics


def plot_msg_channels_comb(msg_ds, elevation_ds, path_out, extent, channels, cmaps, units, min_values, max_values):
    """
    Plots the rain rate alongside various visible (VIS) and infrared (IR) channels from MSG data on a multi-panel figure.

    This function creates a 3x4 grid of subplots. The first subplot displays the rain rate, followed by three VIS channels and eight IR channels from MSG data. 
    Each subplot includes coastlines, borders, and gridlines for better geographical context. The plot extents are defined by specified longitude and latitude limits.

    Parameters:
    
    msg_ds (xr.Dataset):  MSG channel data.
    elevation_ds (xr.Dataset): orography data
    path_out (str): Path for saving the output plot.
    extent (list of float) lonmin, lonmax, latmin, latmax
    channels (list): List of channel names .

    The function saves the resulting plot as a PNG file at the specified output path.
    """
    
    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    # set up elevation
    elevation = elevation_ds['orography'].values  
    el_min, el_max = get_max_min(elevation_ds, 'orography')
    contour_levels = np.linspace(el_min, el_max, num=10)  # Adjust number of levels as needed

    #extract coordinates
    lats = msg_ds.coords['lat'].values
    lons = msg_ds.coords['lon'].values
    #lat_grid, lon_grid = np.meshgrid(lats,lons,indexing='ij')

    time = str(msg_ds.coords['time'].values[0]).split('.')[0]
    print(time)   

    #get channels value
    for i, ch in enumerate(channels):
        ch_values = msg_ds[ch].values.squeeze()
        #ch_values = np.flip(ch_values[0,:,:]) 
        vmin = min_values[i]
        vmax = max_values[i]

        #set cmap, label and normalize the colorbar
        cmap = cmaps[i]
        unit = units[i]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        set_map_plot(axs[i],norm,cmap,extent,'Channel '+ch,unit)        
        axs[i].contourf(lons, lats, ch_values, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap)
        axs[i].set_facecolor('#dcdcdc')
        axs[i].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Adjust layout
    fig.suptitle(f"MSG and Rain Rate Data for Time: {time}", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'multichannels_map_'+str(time).replace(' ','_')+'.png', bbox_inches='tight')#, transparent=True)
    print('\nFig saved in',path_out+'multichannels_map_'+str(time).replace(' ','_')+'.png')
    plt.close()




def plot_distributions_comb(msg_ds_day, path_out, channels, units, min_values, max_values, binwidths):
    """
    Plots the distibution of rain rate alongside various visible (VIS) and infrared (IR) channels from MSG data on a multi-panel figure.

    This function creates a 3x4 grid of subplots. The first subplot displays the rain rate, followed by three VIS channels and eight IR channels from MSG data. 

    Parameters:
    msg_ds (xr.Dataset): 
    path_out (str): Path for saving the output plot.
    channels (list): List of channel names 

    The function saves the resulting plot as a PNG file at the specified output path.
    """

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    axs = axs.flatten()

    #get channels value
    for i, ch in enumerate(channels):
        ch_values = msg_ds_day[ch][:]
        axs[i].hist(ch_values.values.flatten(), bins=np.arange(min_values[i],max_values[i]+binwidths[i],binwidths[i]), color='blue', alpha=0.7, density=True)
        axs[i].set_title('Channel '+ch, fontsize=10)
        axs[i].set_xlabel(units[i])

    # Adjust layout
    fig.suptitle(f"MSG combinations Density Functions", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'distribution_histo.png', bbox_inches='tight')#, transparent=True)
    plt.close()





def plot_channel_daily_trends_comb(channels, ds, path_out, unit, ch_type):
    """
    Plot the mean temporal trend of each channel for a given list of times.

    Parameters:
    -----------
    channels : list of str
        List of channel names.
    ds: Dataset
        Xarray Dataset with the channels values
    path_out : str
        Directory to save the output figure.
    unit : str
        unit of the channels.

    Returns:
    --------
    None
    """

    # Select the data for the channels, and feature
    ds_time = ds[channels]

    #rename time coordinate and convert to datetime
    #ds_time = ds_time.rename({'end_time': 'time'})

    #resample to daily data and compute mean
    ds_time = ds_time.groupby('time.hour').mean('time')

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('average daily trend' , fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.94,left=0.05)  # Adjust the space between the title and the plot

    #set time in a day
    time_data = np.arange(0,24)

    # Loop over channels
    for i, channel in enumerate(channels):
        channel_data = ds_time[channel] 

        # Calculate the mean value of the channel over the specified time steps
        mean_data_xr = channel_data.mean(dim=('x', 'y'), skipna=True) 
        mean_data = mean_data_xr.compute().values          
        
        ax.plot(time_data,mean_data, linestyle='-', marker='.', linewidth=1, alpha=0.7, label=channel.split(' ')[-1])

    # Set the x-axis ticks and labels
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax.grid(visible= True, which='both', color ='grey', linestyle ='--', linewidth = 0.5, alpha = 0.8)

    # Set the axis labels and legend
    ax.set_xlabel('Hour (UTC)',fontsize=14)
    ax.set_ylabel(unit,fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save and show the figure
    fig.savefig(path_out+f'AvgDailyTrends_{ch_type}.png', bbox_inches='tight')#, transparent=True)
    plt.close()

    return None



def plot_spatial_percentile_comb(msg_ds, elevation_ds, percentiles, channels, extent, path_out, cmaps, units):
    """
    Plot the specified percentile of spatial data over time on a geographical map.

    Parameters:
    - data (xarray.DataArray): The spatial data (must include lat and lon dimensions).
    - percentile (int): The percentile to calculate and plot.
    - time_step (str or None): Specific time step to plot. If None, uses all time steps.
    """

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
        
    # get coordinates
    lats = msg_ds.coords['lat'].values
    lons = msg_ds.coords['lon'].values

    # set up elevation
    elevation = elevation_ds['orography'].values  
    el_min, el_max = get_max_min(elevation_ds, 'orography')
    contour_levels = np.linspace(el_min, el_max, num=10)  # Adjust number of levels as needed

    #get channels value
    for i, ch in enumerate(channels):
        ds_msg_rechunked = msg_ds.chunk({'time': -1})
        ch_values = ds_msg_rechunked[ch].quantile(percentiles[i]/100.0, dim='time', method='linear').values.squeeze()
        #ch_values = np.flip(ch_values[0,:,:]) 
        vmin = ch_values.min()
        vmax = ch_values.max() 

        #set cmap, label and normalize the colorbar
        cmap = cmaps[i]
        unit = units[i]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        set_map_plot(axs[i],norm,cmap,extent,'Percentile '+str(percentiles[i])+' - Ch '+ch,unit)        
        axs[i].contourf(lons, lats, ch_values, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap)
        axs[i].set_facecolor('#dcdcdc')

        axs[i].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Adjust layout
    fig.suptitle("Data Percentiles", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    percentiles = np.unique(percentiles)
    fig.savefig(path_out+f'comb_map_percentiles_{percentiles}.png', bbox_inches='tight')#, transparent=True)
    plt.close()




def plot_distributions_by_rain_classes_comb(rain_rate_ds, msg_ds, path_out, channels, rain_classes, rain_classes_name, bin_widths, units):
    """
    Plots the distribution of brightness temperature or reflectance across different rain classes for various MSG channels.
    """

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    axs = axs.flatten()
    
    # Plot distributions for each channel divided by rain classes
    for i, ch in enumerate(channels):
        #color = 'blue' if ch in vis_channels else 'green'
        ch_min, ch_max = get_max_min(msg_ds,ch)
        bins = np.arange(ch_min, ch_max + bin_widths[i], bin_widths[i])
        
        # Plot distributions for each rain class
        for n, rain_class in enumerate(rain_classes):
            print('rain class', rain_class)
            class_mask = (rain_rate_ds['rain_rate'] >= rain_class[0]) & (rain_rate_ds['rain_rate'] < rain_class[1])
            #print(class_mask)
            #print(msg_ds[ch])
            print("True values in mask:", np.sum(class_mask.values))           
            # Applying mask and dropping NaNs explicitly
            class_data = msg_ds[ch].where(class_mask)
            print("Class data after masking:", class_data)
            class_data = class_data.values.flatten()
            #print("Non-NaN values count:", np.count_nonzero(~np.isnan(class_data)))
            
            # Avoid plotting empty data
            if len(class_data) > 0:
                axs[i].hist(class_data, bins=bins, alpha=0.7, density=True, label=rain_classes_name[n], histtype='step', linewidth=2)
                
                
        axs[i].set_title(f'Channel {ch}', fontsize=10)
        axs[i].set_xlabel(units[i])
        if i==2:
            axs[i].legend(title="Rain classes")
    
    # Adjust layout and save the plot
    fig.suptitle("MSG Channel Distributions by Rain Class", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    fig.savefig(path_out + 'distribution_by_rain_class.png', bbox_inches='tight')
    plt.close(fig)



def filter_dataset_by_variable(ds_to_filter, ds_mask, variable_name, min_threshold, max_threshold):
    """
    Filters the input xarray.Dataset to include only data points where the specified variable
    is within the given minimum and maximum thresholds, inclusive.
    
    Parameters:
    - ds (xarray.Dataset): The dataset to be filtered.
    - variable_name (str): The name of the variable to filter by.
    - min_threshold (float): The minimum value to include.
    - max_threshold (float): The maximum value to include.
    
    Returns:
    - xarray.Dataset: The filtered dataset.
    """
    # Ensure that the input thresholds are valid
    if min_threshold > max_threshold:
        raise ValueError("The minimum threshold cannot be greater than the maximum threshold.")

    # Ensure the variable exists in the dataset
    if variable_name not in ds_mask.variables:
        raise ValueError(f"The variable '{variable_name}' is not present in the dataset.")

    # Create a mask where the variable is between the minimum and maximum thresholds (inclusive)
    mask = (ds_mask[variable_name] >= min_threshold) & (ds_mask[variable_name] <= max_threshold)
    
    # Apply this mask to the dataset, keeping only the data points that match the condition
    filtered_ds = ds_to_filter.where(mask)
    
    return filtered_ds



def calculate_ets_across_thresholds(msg_ds, channel_name, channel_trend, y_true, thresholds, rain_threshold, labels=[0, 1]):
    ets_results = []

    for threshold in thresholds:
        print('\n channel threshold: ', threshold)
        # Generate y_pred based on the current threshold for the channel
        if channel_trend=='positive':
            y_pred = np.where(msg_ds[channel_name] >= threshold, 1, 0)
        elif channel_trend=='negative':
            y_pred = np.where(msg_ds[channel_name] > threshold, 0, 1)
        else:
            print('wrong channel trend name!')
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


def save_ets_thresholds_channels_trends_comb(msg_ds, rain_ds, channels, channel_trends, rain_thresholds, elev_class, threshold_n_bin, path_out):

    # Define the DataFrame with the columns you'll need
    df = pd.DataFrame(columns=['Rain_Threshold','Channel','MSG_Threshold','ETS'])

    #loop over the different rain classifications
    for rain_threshold in rain_thresholds:
        print('\nrain threshold: ', rain_threshold)
        if rain_threshold!=0.1:
            rain_max = get_max_min(rain_ds,'rain_rate')[1]
            rain_ds = filter_dataset_by_variable(rain_ds, rain_ds, 'rain_rate', 0.1, rain_max) 
            msg_ds = filter_dataset_by_variable(msg_ds, rain_ds, 'rain_rate', 0.1, rain_max)
        
        #generate y_true on the current threshold for the rain rate
        y_true = np.where(rain_ds['rain_rate']>= rain_threshold, 1, 0)
        y_true = y_true.flatten()

        #read the medians of the rain classes distributions
        #msg_intervals = extract_medians(path_out+'channel_stats.txt',channels,rain_classes[n])

        for i,channel in enumerate(channels):
            print('\nchannel: ', channel)
            #find interval for the threshold
            msg_min, msg_max = get_max_min(msg_ds,channel)

            #define the threshold
            thresholds = np.linspace(msg_min, msg_max, threshold_n_bin)

            #calculate the ets for every threshold
            df_ets = calculate_ets_across_thresholds(msg_ds, channel, channel_trends[i], y_true, thresholds, rain_threshold)

            df = pd.concat([df, df_ets], ignore_index=True)

    #save dataFrame
    df.to_csv(path_out+"ets_results_"+elev_class+".csv", index=False)



def plot_ets_trend_rain_threshold_comb(path_out, channels_units, elev_class):
    # Load the CSV file
    df = pd.read_csv(path_out+"ets_results_"+elev_class+".csv")

    # Set up the subplot grid
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))
    axs = axs.flatten()

    # Get unique channels and rain threshold
    channels = df['Channel'].unique()
    rain_thresholds = df['Rain_Threshold'].unique()

    # Use a qualitative colormap
    cmap = plt.get_cmap('Dark2')

    # Generate diverse colors
    colors = [cmap(i) for i in range(len(rain_thresholds))]

    # Loop through the channels and plot
    for i, channel in enumerate(channels):
        unit = channels_units[i]
        for j, rain_th in enumerate(rain_thresholds):
            channel_data = df[(df['Channel'] == channel) & (df['Rain_Threshold'] == rain_th)]
            axs[i].plot(channel_data['MSG_Threshold'], channel_data['ETS'], marker='.', linestyle='-', color=colors[j])#, label=str(rain_th))
        axs[i].set_title(channel)
        axs[i].set_xlabel(f'Channel Threshold in {unit}')
        axs[i].set_ylabel('ETS')
        axs[i].grid(True)
        #if i==3:
        #    axs[i].legend(loc='best', title="Rain threshold")


    # Create a custom legend for the whole figure
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=4, label=str(rain_thresholds[0])+' mm/h'),
                    plt.Line2D([0], [0], color=colors[1], lw=4, label=str(rain_thresholds[1])+' mm/h'),
                    plt.Line2D([0], [0], color=colors[2], lw=4, label=str(rain_thresholds[2])+' mm/h')]

    # Place the legend on the 12th subplot's axes
    axs[-1].legend(handles=legend_elements, loc='center', title="Rain Thresholds")


    # Adjust layout and save the plot
    fig.suptitle("ETS trend - "+elev_class, fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    fig.savefig(path_out + 'ets_trends_by_rain_class_'+elev_class+'.png', bbox_inches='tight')
    plt.close(fig)


def generate_labels(ds, variable_name, threshold, trend):
    """
    Generate binary labels based on a specified threshold and trend for a given variable in an xarray.Dataset.
    NaN values in the dataset are preserved in the output labels.
    
    Parameters:
    - ds (xarray.Dataset): The dataset containing the variable to generate labels for.
    - variable_name (str): The name of the variable in the dataset to be evaluated.
    - threshold (float): The threshold value used to generate binary labels.
    - trend (str): Specifies the trend for labeling. 
      - 'positive' indicates labels of 1 for values greater than or equal to the threshold, and 0 otherwise.
      - 'negative' indicates labels of 1 for values less than or equal to the threshold, and 0 otherwise.
    
    Returns:
    - numpy.ndarray: A flattened array of binary labels based on the specified threshold and trend, preserving NaN values.
    """
    data = ds[variable_name].values
    labels = np.full_like(data, np.nan)  # Create an array of NaNs with the same shape as the data
    
    if trend == 'positive':
        labels[data >= threshold] = 1
        labels[data < threshold] = 0
    elif trend == 'negative':
        labels[data <= threshold] = 1
        labels[data > threshold] = 0
    else:
        raise ValueError("Invalid trend name. Must be 'positive' or 'negative'.")

    return labels.flatten()



def save_metrics_thresholds_channels_comb(msg_ds, rain_ds, channels,channel_trends, rain_thresholds, elev_class, path_out):
    # Define array to store metrics
    results = []

    #loop over the different rain classifications
    for rain_threshold in rain_thresholds:
        print('\nrain threshold: ', rain_threshold)
        if rain_threshold!=0.1:
            rain_max = get_max_min(rain_ds,'rain_rate')[1]
            rain_ds = filter_dataset_by_variable(rain_ds, rain_ds, 'rain_rate', 0.1, rain_max) 
            msg_ds = filter_dataset_by_variable(msg_ds, rain_ds, 'rain_rate', 0.1, rain_max)
        
        #generate y_true on the current threshold for the rain rate
        y_true = generate_labels(rain_ds,'rain_rate',rain_threshold,'positive')      

        for i, channel in enumerate(channels):
            print('\nchannel: ', channel)
            #define the threshold
            msg_threshold = find_max_ets_threshold(path_out+"ets_results_"+elev_class+".csv",channel,rain_threshold)

            # Generate y_pred based on the current threshold for the channel
            y_pred = generate_labels(msg_ds,channel,msg_threshold,channel_trends[i])

            #find mask that exclude nan values for both arrays
            mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

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




def plot_conf_matrix_radar_channel_maps(msg_ds, rain_ds, elevation_ds, rain_thres, msg_threshold, channel_trend, path_out, extent, channel, cmap_msg, unit, ch_min, ch_max):
    """
    Plot the confusion matrix spatial maps.

    Parameters:
    - msg_ds (xarray.Dataset): The MSG dataset.
    - rain_ds (xarray.Dataset): The rain dataset.
    - ds_oro (xarray.Dataset): The orography dataset.
    - rain_thres (float): Rain threshold value.
    - msg_threshold (float): MSG threshold value.
    - trend (str): Trend type ('positive' or 'negative').
    - output_path (str): Path to save the output.
    - case_domain (str): Case domain description.
    - channel (str): The channel to analyze.
    - cmap (str): Colormap.
    - unit (str): Unit of the data.
    - min_val (float): Minimum value for normalization.
    - max_val (float): Maximum value for normalization.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    #get spatial and time coords
    time = str(rain_ds.coords['time'].values[0]).split('.')[0]
    print(time)
    lats = rain_ds.coords['lat'].values
    lons = rain_ds.coords['lon'].values
    lat_grid, lon_grid = np.meshgrid(lats,lons,indexing='ij')

    # set up elevation
    elevation = elevation_ds['orography'].values  
    el_min, el_max = get_max_min(elevation_ds, 'orography')
    contour_levels = np.linspace(el_min, el_max, num=10)  # Adjust number of levels as needed

    # Plot rain rate
    mask_rain = rain_ds['rain_rate'] >= rain_thres
    # Apply this mask to the rain rate dataset, keeping only the data points that match the condition
    rain_ds_masked = rain_ds.where(mask_rain)
    rain_rate = rain_ds_masked.variables['rain_rate'].values.squeeze() 
    cmap = plt.cm.gist_ncar.copy()
    norm = mcolors.PowerNorm(gamma=0.4)
    axs[0].contourf(lon_grid, lat_grid, rain_rate, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))
    set_map_plot(axs[0],norm,cmap,extent,'Rain Rate','Rain Rate (mm/h)')
    axs[0].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # plot msg channel
    if channel_trend == 'positive':
        mask_msg = msg_ds[channel] >= msg_threshold
    elif channel_trend == 'negative':
        mask_msg = msg_ds[channel] <= msg_threshold
    # Apply this mask to the rain rate dataset, keeping only the data points that match the condition
    msg_ds_masked = msg_ds.where(mask_msg)
    ch_values = msg_ds_masked[channel].values.squeeze()
    #ch_values = np.flip(ch_values[0,:,:]) 

    norm = mcolors.Normalize(vmin=ch_min, vmax=ch_max)
    set_map_plot(axs[1],norm,cmap_msg,extent,channel,unit)        
    axs[1].contourf(lons, lats, ch_values, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap_msg)
    #axs[1].set_facecolor('#dcdcdc')
    axs[1].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # plot the values of the confusion matrix
   
    # Generate labels
    y_true = generate_labels(rain_ds, 'rain_rate', rain_thres, 'positive')
    y_pred = generate_labels(msg_ds, channel, msg_threshold, channel_trend)
    
    # Initialize an array to store the classification results
    classification_map = np.full_like(y_true, 0, dtype=float)

    # Define classifications
    HIT = 1
    FALSE_ALARM = 2
    MISS = 3
    CORRECT_REJECTION = 4

    # Populate classification_map based on y_true and y_pred
    mask_not_nan = ~np.isnan(y_true) & ~np.isnan(y_pred)
    classification_map[mask_not_nan & (y_true == 1) & (y_pred == 1)] = HIT
    classification_map[mask_not_nan & (y_true == 0) & (y_pred == 1)] = FALSE_ALARM
    classification_map[mask_not_nan & (y_true == 1) & (y_pred == 0)] = MISS
    classification_map[mask_not_nan & (y_true == 0) & (y_pred == 0)] = CORRECT_REJECTION

    # Define custom colormap
    colors = ['silver', 'limegreen', 'lightcoral', 'gold' , 'lightcyan']  # White for NaN, green for hit, red for false alarm, blue for miss, gray for correct rejection
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Define boundaries for classification
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    legend_labels = ['No Data', 'Hit', 'False Alarm', 'Miss', 'Correct Rejection']

    # Reshape classification_map to match the original lats/lons shape
    classification_map_reshaped = classification_map.reshape(np.shape(lat_grid))
    print(np.unique(classification_map))
    
    # Customize the map
    set_map_plot_nocolorbar(axs[2], extent, 'Classification Results')

    # Plot classification results #TODO problem with the border of the FALSE ALARM that are signed as missing!
    cf = axs[2].contourf(lons, lats, classification_map_reshaped,levels=bounds, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
    cbar = fig.colorbar(cf, ax=axs[2], boundaries=bounds, ticks=range(5)) 
    cbar.ax.set_yticklabels(legend_labels)
    axs[2].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Removing axis for the fourth plot (empty plot)
    axs[3].axis('off')

    # Adjust layout
    fig.suptitle(f"Time: {time}", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'conf_matrix_maps/confusion_matrix_map_'+channel+'_'+str(time).replace(' ','_')+'.png', bbox_inches='tight')#, transparent=True)
    print('\nFig saved in',path_out+'conf_matrix_maps/confusion_matrix_map_'+str(time).replace(' ','_')+'.png')
    plt.close()



def plot_conf_matrix_maps(msg_ds, rain_ds, elevation_ds, rain_thres, msg_thresholds, channel_trends, path_out, channels, extent):
    """
    Plot the confusion matrix spatial maps.

    Parameters:
    - msg_ds (xarray.Dataset): The MSG dataset.
    - rain_ds (xarray.Dataset): The rain dataset.
    - ds_oro (xarray.Dataset): The orography dataset.
    - rain_thres (float): Rain threshold value.
    - msg_threshold (float): MSG threshold value.
    - trend (str): Trend type ('positive' or 'negative').
    - output_path (str): Path to save the output.
    - channel (str): The channel to analyze.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    #get spatial and time coords
    time = str(rain_ds.coords['time'].values[0]).split('.')[0]
    print(time)
    lats = rain_ds.coords['lat'].values
    lons = rain_ds.coords['lon'].values
    lat_grid, lon_grid = np.meshgrid(lats,lons,indexing='ij')

    # set up elevation
    elevation = elevation_ds['orography'].values  
    el_min, el_max = get_max_min(elevation_ds, 'orography')
    contour_levels = np.linspace(el_min, el_max, num=10)  # Adjust number of levels as needed

    # plot the values of the confusion matrix

    # Define classifications
    HIT = 1
    FALSE_ALARM = 2
    MISS = 3
    CORRECT_REJECTION = 4

    # Define custom colormap
    colors = ['silver', 'limegreen', 'lightcoral', 'gold' , 'lightcyan']  # White for NaN, green for hit, red for false alarm, blue for miss, gray for correct rejection
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Define boundaries for classification
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    legend_labels = ['No Data', 'Hit', 'False Alarm', 'Miss', 'Correct Rejection']
   
    # Generate labels for rain rate
    y_true = generate_labels(rain_ds, 'rain_rate', rain_thres, 'positive')

    for i, channel in enumerate(channels):
        # Generate labels for msg channels or derivatives
        y_pred = generate_labels(msg_ds, channel, msg_thresholds[i], channel_trends[i])
        
        # Initialize an array to store the classification results
        classification_map = np.full_like(y_true, 0, dtype=float)

        # Populate classification_map based on y_true and y_pred
        mask_not_nan = ~np.isnan(y_true) & ~np.isnan(y_pred)
        classification_map[mask_not_nan & (y_true == 1) & (y_pred == 1)] = HIT
        classification_map[mask_not_nan & (y_true == 0) & (y_pred == 1)] = FALSE_ALARM
        classification_map[mask_not_nan & (y_true == 1) & (y_pred == 0)] = MISS
        classification_map[mask_not_nan & (y_true == 0) & (y_pred == 0)] = CORRECT_REJECTION

        # Reshape classification_map to match the original lats/lons shape
        classification_map_reshaped = classification_map.reshape(np.shape(lat_grid))
        #print(np.unique(classification_map))
        
        # Customize the map
        set_map_plot_nocolorbar(axs[i], extent, channel)

        # Plot classification results #TODO problem with the border of the FALSE ALARM that are signed as missing!
        cf = axs[i].contourf(lons, lats, classification_map_reshaped,levels=bounds, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)
        axs[i].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Create an axis for the colorbar on the right
    cbar_ax = fig.add_axes([0.88, 0.10, 0.08, 0.8])  # [left, bottom, width, height]
    cbar_ax.axis('off')
    
    #create a colorbar for all the subpltos
    cbar = fig.colorbar(cf, ax=cbar_ax, boundaries=bounds, ticks=range(5), location='right', aspect=50)
    cbar.ax.set_yticklabels(legend_labels)

    # Adjust layout
    fig.suptitle(f"Time: {time}", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.1, right=0.85)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'conf_matrix_maps/confusion_matrix_map_'+str(channels)+'_'+str(time).replace(' ','_')+'.png', bbox_inches='tight')#, transparent=True)
    print('\nFig saved in',path_out+'conf_matrix_maps/confusion_matrix_map_'+str(time).replace(' ','_')+'.png')
    plt.close()



def save_conf_matrix_values(msg_ds, rain_ds, elevation_ds, rain_thres, msg_thresholds, channels, channel_trends, elev_classes_bounds, elev_classes_names, path_out):
    """
    Compute and save confusion matrix values for each channel and elevation category.

    Parameters:
    - msg_ds (xarray.Dataset): The MSG dataset.
    - rain_ds (xarray.Dataset): The rain dataset.
    - elevation_ds (xarray.Dataset): The elevation dataset.
    - rain_thres (float): Rain threshold value.
    - msg_thresholds (list of float): List of MSG threshold values for each channel.
    - channels (list of str): List of channels to analyze.
    - channel_trends (list of str): List of trend types ('positive' or 'negative') for each channel.
    - elev_classes_bounds (2D numpy array): Bounds of elevation classes.
    - elev_classes_names (list of str): Names of elevation classes.
    - path_out (str): Path to save the output CSV files.
    """

    #get time coords
    time = str(rain_ds.coords['time'].values[0]).split('.')[0]
    print(time)
    
    # Initialize a dictionary to store results
    results = {
        'Elevation_Class': [],
        'Channel': [],
        'Hit': [],
        'False_Alarm': [],
        'Miss': [],
        'Correct_Rejection': [],
        'No_Data': []
    }

    # Loop through elevation classes
    for j, el_class in enumerate(elev_classes_names):
        # Mask rain rate by elevation class
        masked_rain_ds = filter_dataset_by_variable(rain_ds, elevation_ds, 'orography', elev_classes_bounds[j][0], elev_classes_bounds[j][1])
        
        # Generate labels for rain rate
        y_true = generate_labels(masked_rain_ds, 'rain_rate', rain_thres, 'positive')

        # Filter dataset by elevation
        masked_msg_ds = filter_dataset_by_variable(msg_ds, elevation_ds, 'orography', elev_classes_bounds[j][0], elev_classes_bounds[j][1])

        # Loop through channels
        for i, channel in enumerate(channels):
            
            # Generate labels for msg channels or derivatives
            y_pred = generate_labels(masked_msg_ds, channel, msg_thresholds[i], channel_trends[i])

            if len(y_pred) == len(y_true):
                print('Labels have the same size.')
            else:
                raise ValueError('Lengths of y_pred and y_true are not equal. Exiting.')

            # Compute confusion matrix values
            mask_not_nan = ~np.isnan(y_true) & ~np.isnan(y_pred)
            HIT = np.sum(mask_not_nan & (y_true == 1) & (y_pred == 1))
            FALSE_ALARM = np.sum(mask_not_nan & (y_true == 0) & (y_pred == 1))
            MISS = np.sum(mask_not_nan & (y_true == 1) & (y_pred == 0))
            CORRECT_REJECTION = np.sum(mask_not_nan & (y_true == 0) & (y_pred == 0))
            No_data = len(y_pred) - np.sum(mask_not_nan)

            # Append results to dictionary
            results['Elevation_Class'].append(el_class)
            results['Channel'].append(channel)
            results['Hit'].append(HIT)
            results['False_Alarm'].append(FALSE_ALARM)
            results['Miss'].append(MISS)
            results['Correct_Rejection'].append(CORRECT_REJECTION)
            results['No_Data'].append(No_data)

    # Create a DataFrame from results
    df_results = pd.DataFrame(results)

    # Save results to CSV file
    csv_filename = f"confusion_matrix_results_{str(time).replace(' ','_')}.csv"
    csv_filepath = os.path.join(path_out, csv_filename)
    df_results.to_csv(csv_filepath, index=False)

    print(f"Confusion matrix results saved to {csv_filepath}")


def merge_csv_files(folder_path, output_file):
    """
    Merges all CSV files in the given folder into a single CSV file with an additional column for timestamps.
    
    Parameters:
    - folder_path (str): Path to the folder containing the CSV files.
    - output_file (str): Path to save the merged CSV file.
    """
    # List to store individual DataFrames
    df_list = []

    # Regular expression to extract timestamp from filename
    timestamp_pattern = re.compile(r'confusion_matrix_results_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.csv')

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extract timestamp from the filename
            match = timestamp_pattern.search(filename)
            if match:
                timestamp = match.group(1)  # Extract the timestamp part
            else:
                print(f"Skipping file {filename}: Timestamp not found in filename.")
                continue
            
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Add a new column for the timestamp
            df['timestamp'] = timestamp
            
            # Append the DataFrame to the list
            df_list.append(df)
    
    # Concatenate all DataFrames in the list
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved as {output_file}")


def calculate_metrics(merged_csv_file, output_csv_file):
    """
    Calculates the specified metrics for each elevation class and saves the results in a CSV file.
    
    Parameters:
    - merged_csv_file (str): Path to the merged CSV file containing confusion matrix counts.
    - output_csv_file (str): Path to save the calculated metrics as a CSV file.
    """
    # Read the merged CSV file
    df = pd.read_csv(merged_csv_file)

    # Group by elevation class and compute the sum of each category
    grouped = df.groupby('Elevation_Class').sum()

    # Initialize a dictionary to store the metrics
    metrics = {
        'elevation_class': [],
        'POD': [],
        'FAR': [],
        'BIAS': [],
        'CSI': [],
        'ETS': []
    }

    # Calculate the metrics for each elevation class
    for elev_class, group in grouped.iterrows():
        hits = group['Hit']
        misses = group['Miss']
        false_alarms = group['False_Alarm']
        correct_rejections = group['Correct_Rejection']
        total = hits + misses + false_alarms + correct_rejections

        # Calculate metrics
        pod = hits / (hits + misses) if (hits + misses) > 0 else float('nan')
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float('nan')
        bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else float('nan')
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else float('nan')
        hit_random = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else float('nan')
        ets = (hits - hit_random) / (hits + misses + false_alarms - hit_random) if (hits + misses + false_alarms - hit_random) > 0 else float('nan')

        # Store metrics
        metrics['elevation_class'].append(elev_class)
        metrics['POD'].append(pod)
        metrics['FAR'].append(far)
        metrics['BIAS'].append(bias)
        metrics['CSI'].append(csi)
        metrics['ETS'].append(ets)

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(output_csv_file, index=False)
    print(f"Metrics saved as {output_csv_file}")



def calculate_and_plot_metrics_over_time(merged_csv_file, channel, output_plot_file):
    """
    Calculates the specified metrics over time for a specific channel and plots the trend for different elevation classes and the total.

    Parameters:
    - merged_csv_file (str): Path to the merged CSV file containing confusion matrix counts.
    - channel (str): The specific channel to filter and calculate metrics for.
    - output_plot_file (str): Path to save the output plot.
    """
    # Read the merged CSV file
    df = pd.read_csv(merged_csv_file)

    # Filter the dataframe by the specified channel
    df = df[df['Channel'] == channel]

    # Convert the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Resample data by hours, summing up the counts for each hour
    df.set_index('timestamp', inplace=True)
    hourly_df = df.resample('H').sum()

    # Group by elevation class and resample by hour
    elevation_classes = df['Elevation_Class'].unique()
    metrics = {
        'elevation_class': [],
        'POD': [],
        'FAR': [],
        'BIAS': [],
        'CSI': [],
        'ETS': [],
        'timestamp': []
    }

    for elev_class in elevation_classes:
        class_df = df[df['Elevation_Class'] == elev_class].resample('H').sum()
        for timestamp, group in class_df.iterrows():
            hits = group['Hit']
            misses = group['Miss']
            false_alarms = group['False_Alarm']
            correct_rejections = group['Correct_Rejection']
            total = hits + misses + false_alarms + correct_rejections

            # Calculate metrics
            pod = hits / (hits + misses) if (hits + misses) > 0 else float('nan')
            far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float('nan')
            bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else float('nan')
            csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else float('nan')
            hit_random = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else float('nan')
            ets = (hits - hit_random) / (hits + misses + false_alarms - hit_random) if (hits + misses + false_alarms - hit_random) > 0 else float('nan')

            # Store metrics
            metrics['elevation_class'].append(elev_class)
            metrics['POD'].append(pod)
            metrics['FAR'].append(far)
            metrics['BIAS'].append(bias)
            metrics['CSI'].append(csi)
            metrics['ETS'].append(ets)
            metrics['timestamp'].append(timestamp)

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Metrics Trend Over Time by Elevation Class for Channel {channel}')

    for metric, ax in zip(['POD', 'FAR', 'BIAS', 'CSI', 'ETS'], axes.flatten()):
        for elev_class in elevation_classes:
            class_df = metrics_df[metrics_df['elevation_class'] == elev_class]
            ax.plot(class_df['timestamp'], class_df[metric], label=elev_class)
        ax.set_title(metric)
        ax.legend()
        if metric=='BIAS':
            ax.set_yscale('log')

    # Remove empty subplot
    fig.delaxes(axes[2, 1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_plot_file)
    #plt.show()



def save_conf_matrix_values_nc(msg_ds, rain_ds, rain_thres, msg_thresholds, channels, channel_trends, path_out):
    """
    Compute and save confusion matrix values for each channel in a NetCDF file.

    Parameters:
    - msg_ds (xarray.Dataset): The MSG dataset.
    - rain_ds (xarray.Dataset): The rain dataset.
    - rain_thres (float): Rain threshold value.
    - msg_thresholds (list of float): List of MSG threshold values for each channel.
    - channels (list of str): List of channels to analyze.
    - channel_trends (list of str): List of trend types ('positive' or 'negative') for each channel.
    - path_out (str): Path to save the output NetCDF file.
    """

    # Get time coordinate
    time = rain_ds.coords['time'].values[0]

    lats = rain_ds.coords['lat'].values
    lons = rain_ds.coords['lon'].values
    lat_grid, lon_grid = np.meshgrid(lats,lons,indexing='ij')
    
    # Initialize DataArrays for confusion matrix components
    conf_matrix_ds = xr.Dataset(coords=msg_ds.coords)

    # Generate labels for rain rate
    y_true = generate_labels(rain_ds, 'rain_rate', rain_thres, 'positive')

    # Loop through channels
    for i, channel in enumerate(channels):
        # Generate labels for msg channels or derivatives
        y_pred = generate_labels(msg_ds, channel, msg_thresholds[i], channel_trends[i])

        if len(y_pred) == len(y_true):
            print('Labels have the same size.')
        else:
            raise ValueError('Lengths of y_pred and y_true are not equal. Exiting.')

        # Initialize a DataArray to hold the confusion matrix values for the channel
        conf_matrix = xr.full_like(msg_ds[channel].squeeze(), 0, dtype=int)

        # Compute confusion matrix values
        mask_not_nan = ~np.isnan(y_true) & ~np.isnan(y_pred)
        HIT = (mask_not_nan & (y_true == 1) & (y_pred == 1)).astype(int)
        FALSE_ALARM = (mask_not_nan & (y_true == 0) & (y_pred == 1)).astype(int)
        MISS = (mask_not_nan & (y_true == 1) & (y_pred == 0)).astype(int)
        CORRECT_REJECTION = (mask_not_nan & (y_true == 0) & (y_pred == 0)).astype(int)

        HIT = HIT.reshape(np.shape(lat_grid))
        FALSE_ALARM = FALSE_ALARM.reshape(np.shape(lat_grid))
        MISS = MISS.reshape(np.shape(lat_grid))
        CORRECT_REJECTION = CORRECT_REJECTION.reshape(np.shape(lat_grid))
        
        # Assign values to the DataArray
        conf_matrix = xr.where(HIT, 1, conf_matrix)
        conf_matrix = xr.where(FALSE_ALARM, 2, conf_matrix)
        conf_matrix = xr.where(MISS, 3, conf_matrix)
        conf_matrix = xr.where(CORRECT_REJECTION, 4, conf_matrix)

        # Add the DataArray to the dataset
        conf_matrix_ds[channel] = conf_matrix

    # Set the time coordinate
    #conf_matrix_ds = conf_matrix_ds.expand_dims(time=[time])

    # Save the dataset to a NetCDF file
    nc_filename = f"confusion_matrix_results_{str(time).replace(' ','_')}.nc"
    nc_filepath = os.path.join(path_out, nc_filename)
    conf_matrix_ds.to_netcdf(nc_filepath)

    print(f"Confusion matrix results saved to {nc_filepath}")


def plot_distributions_conf_matrix(msg_ds, rain_ds, elevation_ds, conf_matrix_ds, channel, unit, min_value_ch, max_value_ch, output_plot_file):
    """
    Plot the distributions of rain, elevation, and MSG values for each confusion matrix category.

    Parameters:
    - msg_ds (xarray.Dataset): The MSG dataset.
    - rain_ds (xarray.Dataset): The rain dataset.
    - elevation_ds (xarray.Dataset): The elevation dataset.
    - conf_matrix_ds (xarray.Dataset): The confusion matrix dataset.
    - channel (str): The channel to analyze.
    - output_plot_file (str): Path to save the output plot.
    """
    # Replicate the elevation dataset for each time step
    elevation_ds = replicate_elevation_for_time(elevation_ds, msg_ds)

    #update rain rate dataset time coordinates to match msg dataset
    rain_ds = update_time_coords(msg_ds, rain_ds)

    # Define the confusion matrix categories
    categories = {
        1: 'Hit',
        2: 'False Alarm',
        3: 'Miss',
        4: 'Correct Rejection'
    }

    # Initialize the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Distributions of {channel} Values by Confusion Matrix Category',fontsize=12, fontweight='bold')

    #Define bins for rain
    min_value_rain = 9e-2, 
    max_value_rain = 80, 
    n_bin_rain = 50
    bins_rain = np.logspace(np.log10(min_value_rain), np.log10(max_value_rain), n_bin_rain).squeeze()
    #print(bins_rain)

    #Define bin for elevation
    min_value_el = -20
    max_value_el = 1400
    bin_width_el = 20
    bins_el = np.arange(min_value_el,max_value_el+bin_width_el,bin_width_el)
    #print(bins_el)

    #Define bin for channel
    bin_width_ch = 2
    bins_ch = np.arange(min_value_ch,max_value_ch+bin_width_ch,bin_width_ch)
    #print(bins_ch)

    #run over the different categories
    for cat_val, cat_name in categories.items():
        #find the mask values
        mask = conf_matrix_ds[channel] == cat_val

        # Plot the distribution of rain value
        masked_rain_ds = rain_ds.where(mask)
        rain_values = masked_rain_ds['rain_rate'].values.flatten()
        rain_values = rain_values[~np.isnan(rain_values)]

        # Handle zero and positive values separately
        non_rainy_values = rain_values[rain_values == 0]
        rainy_values = rain_values[rain_values > 0]
        print(f'dry pixels for {cat_name}: {len(non_rainy_values)}')

        if len(non_rainy_values) > 0:
            axes[0].hist(non_rainy_values, bins=np.arange(0,0.08,0.04), histtype='step')# density=True)
        if len(rainy_values) > 0:
            axes[0].hist(rainy_values, bins=bins_rain, histtype='step')#, density=True)

        axes[0].set_title('Rain Distribution')
        axes[0].set_xlabel('Rain Rate (mm/h)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')

        # Plot the distribution of elevation values
        masked_elevation_ds = elevation_ds.where(mask)
        elevation_values = masked_elevation_ds['orography'].values.flatten()
        elevation_values = elevation_values[~np.isnan(elevation_values)]
        axes[1].hist(elevation_values, bins=bins_el, histtype='step', label=cat_name)# density=True)
        axes[1].set_title('Elevation Distribution')
        axes[1].set_xlabel('Elevation (m)')

        # Plot the distribution of MSG values for the specified channel
        masked_msg_ds = msg_ds.where(mask)
        msg_values = masked_msg_ds[channel].values.flatten()
        msg_values = msg_values[~np.isnan(msg_values)]
        axes[2].hist(msg_values, bins=bins_ch, histtype='step', label=cat_name)#, density=True)
        axes[2].set_title(f'{channel} Distribution')
        axes[2].set_xlabel(unit)
    
    #plot legend only on the last plot
    axes[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_plot_file)


def replicate_elevation_for_time(elevation_ds, msg_ds):
    """
    Replicate the elevation dataset for each time step.

    Parameters:
    - elevation_ds (xarray.Dataset): The elevation dataset.
    - msg_ds (xarray.Dataset): The dataset containing time coordinates.

    Returns:
    - xarray.Dataset: The replicated elevation dataset.
    """
    time_coords = msg_ds.coords['time']

    elevation_data = elevation_ds['orography'].values
    replicated_data = np.repeat(elevation_data[np.newaxis, :, :], len(time_coords), axis=0)
    #print(replicated_data)

    # Define dimensions and coordinates for DataArray
    dims = ['time', 'y', 'x']
    coords = {
        'time': time_coords,
        'lat': elevation_ds.coords['lat'],
        'lon': elevation_ds.coords['lon']
    }

    # Convert replicated_data to DataArray
    replicated_da = xr.DataArray(replicated_data, dims=dims, coords=coords)

    # Create a new xarray Dataset with the same coordinates as msg_ds
    new_elev_ds = xr.Dataset(coords=msg_ds.coords)

    # Assign DataArray to 'orography' variable in new_elev_ds
    new_elev_ds['orography'] = replicated_da

    #print('new elevetion ds', new_elev_ds)

    return new_elev_ds


def update_time_coords(msg_ds, rain_ds):
    """
    Update the time coordinates of rain_ds to match those of msg_ds without changing the data values.

    Parameters:
    - msg_ds (xarray.Dataset): The MSG dataset with the desired time coordinates.
    - rain_ds (xarray.Dataset): The rain dataset to be updated.

    Returns:
    - xarray.Dataset: The updated rain dataset with time coordinates matching msg_ds.
    """
    # Ensure the time dimensions are of the same length
    if len(msg_ds['time']) != len(rain_ds['time']):
        raise ValueError("The length of time dimensions in msg_ds and rain_ds do not match.")

    # Update the time coordinates of rain_ds to match those of msg_ds
    rain_ds = rain_ds.assign_coords(time=msg_ds['time'])
    
    return rain_ds