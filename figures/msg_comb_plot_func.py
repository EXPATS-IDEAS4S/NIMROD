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

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
#from readers.read_functions import 
from figures.plot_functions import set_map_plot, plot_rain_data
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
        y_true = np.where(rain_ds['rain_rate']>= rain_threshold, 1, 0)
        y_true = y_true.flatten()       

        for i, channel in enumerate(channels):
            print('\nchannel: ', channel)
            #define the threshold
            msg_threshold = find_max_ets_threshold(path_out+"ets_results_"+elev_class+".csv",channel,rain_threshold)

            # Generate y_pred based on the current threshold for the channel
            if channel_trends[i]=='positive':
                y_pred = np.where(msg_ds[channel] >= msg_threshold, 1, 0)
            elif channel_trends[i]=='negative':
                y_pred = np.where(msg_ds[channel] > msg_threshold, 0, 1)
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