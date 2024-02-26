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



def plot_msg_channels_and_rain_rate(rain_rate_ds, msg_ds, path_out, lonmin, lonmax, latmin, latmax, vis_channels, ir_channels, min_values, max_values):
    """
    Plots the rain rate alongside various visible (VIS) and infrared (IR) channels from MSG data on a multi-panel figure.

    This function creates a 3x4 grid of subplots. The first subplot displays the rain rate, followed by three VIS channels and eight IR channels from MSG data. 
    Each subplot includes coastlines, borders, and gridlines for better geographical context. The plot extents are defined by specified longitude and latitude limits.

    Parameters:
    rain_rate_ds (str): Path to the NetCDF file containing the rain rate data.
    msg_ds (str): Path to the NetCDF file containing MSG channel data.
    path_out (str): Path for saving the output plot.
    lonmin (float): Minimum longitude for the plot extent.
    lonmax (float): Maximum longitude for the plot extent.
    latmin (float): Minimum latitude for the plot extent.
    latmax (float): Maximum latitude for the plot extent.
    vis_channels (list): List of visible channel names in the MSG data.
    ir_channels (list): List of infrared channel names in the MSG data.

    The function saves the resulting plot as a PNG file at the specified output path.
    """
    channels = vis_channels+ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    # Specify the units and calendar type for your time variable
    time_units = 'hours since 2000-01-01 00:00:00'
    calendar_type = 'gregorian'

    #get lon and lat grid and rain rate
    with Dataset(rain_rate_ds, 'r') as nc_rain:
        #print(nc.variables)
        lats = nc_rain['latitude'][:] 
        lons = nc_rain['longitude'][:]
        rain_rate = nc_rain['rain_rate'][:]
        #print(rain_rate)
        time = nc_rain['time']
        time = num2date(time[:], units=time_units, calendar=calendar_type)[0]
        
    # Plot rain rate
    cmap = plt.cm.gist_ncar.copy()
    norm = mcolors.PowerNorm(gamma=0.4)
    mesh = axs[0].contourf(lons, lats, rain_rate, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))
    axs[0].coastlines()
    axs[0].set_title('Rain Rate', fontsize=10)

    # Add colorbar with reduced size
    cbar = plt.colorbar(mesh, label='Rain Rate (mm/h)', shrink=0.8)

    # Adds coastlines and borders to the current axes
    axs[0].coastlines(resolution='50m', linewidths=0.5) 
    axs[0].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    #set axis thick labels
    gl = axs[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Set the extent to the specified domain
    axs[0].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    #get channels value
    with Dataset(msg_ds, 'r') as nc_msg:
        #print(nc_msg.variables)
        for i, ch in enumerate(channels):
            ch_values = nc_msg[ch][:]
            ch_values = np.flip(ch_values[0,:,:]) 
            vmin = min_values[i]
            vmax = max_values[i]

            if ch in vis_channels:
                cmap = 'ocean'
                unit = 'Reflectance (%)'
                # Extracting the timestamp part from the filename
                time_msg = msg_ds.split('/')[-1].split('-')[-1].split('.')[0]

                # Extracting the hour part from the timestamp
                hour = time_msg[8:10]

                # Check if hour is within the specified range
                if not ('04' <= hour < '17'):
                    # Fill ch_values with NaN if hour is outside the range '04' to '17'
                    ch_values = np.full(ch_values.shape, np.nan)
            elif ch in ir_channels:
                cmap = 'coolwarm'
                unit = 'Brightness Temp (K)'
            else:
                print('wrong channel name')

            # Create a Normalize object
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            # Create a ScalarMappable with the normalization and colormap
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
        
            mesh = axs[i+1].contourf(lons, lats, ch_values, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap)
            plt.colorbar(sm, label= unit, ax=axs[i+1], shrink=0.8)

            axs[i+1].set_facecolor('#dcdcdc')
            axs[i+1].coastlines()
            axs[i+1].set_title('Channel '+ch, fontsize=10)

            #set axis thick labels
            gl = axs[i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlines = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 10, 'color': 'black'}
            gl.ylabel_style = {'size': 10, 'color': 'black'}

            # Adds coastlines and borders to the current axes
            axs[i+1].coastlines(resolution='50m', linewidths=0.5) 
            axs[i+1].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

            # Set the extent to the specified domain
            axs[i+1].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    # Adjust layout
    fig.suptitle(f"MSG and Rain Rate Data for Time: {time}", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'maps/multichannels_map_'+str(time)+'.png', bbox_inches='tight')#, transparent=True)
    plt.close()




def create_gif_from_folder(folder_path, output_path, duration=0.5):
    """
    Creates a GIF from all the images in a specified folder.

    Parameters:
    folder_path (str): Path to the folder containing image files.
    output_path (str): Path where the GIF should be saved, including the filename and .gif extension.
    duration (float): Duration of each frame in the GIF.
    """
    # Get list of file names in the folder
    filenames = sorted([file for file in os.listdir(folder_path) if file.endswith('.png')])

    # Debug print
    print(f"Found {len(filenames)} .png files in {folder_path}")

    # Read images into a list
    images = []
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))

    # Debug print
    print(f"Creating GIF at {output_path}")

    # Save the images as a gif
    imageio.mimsave(output_path, images, duration=duration)


def get_max_min(ds,ch):
    ch_values = ds[ch][:]
    ch_values = ch_values.values.flatten()
    ch_values = ch_values[~np.isnan(ch_values)]
    max = np.amax(ch_values)
    min = np.amin(ch_values)

    return min, max


def plot_distributions(rain_rate_ds, msg_ds, msg_ds_day, path_out, vis_channels, ir_channels):
    """
    Plots the distibution of rain rate alongside various visible (VIS) and infrared (IR) channels from MSG data on a multi-panel figure.

    This function creates a 3x4 grid of subplots. The first subplot displays the rain rate, followed by three VIS channels and eight IR channels from MSG data. 

    Parameters:
    rain_rate_ds (str): Path to the NetCDF file containing the rain rate data.
    msg_ds (str): Path to the NetCDF file containing MSG channel data.
    path_out (str): Path for saving the output plot.
    vis_channels (list): List of visible channel names in the MSG data.
    ir_channels (list): List of infrared channel names in the MSG data.

    The function saves the resulting plot as a PNG file at the specified output path.
    """
    channels = vis_channels+ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axs = axs.flatten()

    #get lon and lat grid and rain rate
    rain_rate = rain_rate_ds['rain_rate'][:]
        
    # Plot rain rate histo
    min_value = 0
    max_value = 100
    bin_width = 2
    axs[0].hist(rain_rate.values.flatten(), bins=np.arange(min_value,max_value+bin_width,bin_width), color='red', alpha=0.7, density=True)
    axs[0].set_title('NIMROD data', fontsize=10)
    axs[0].set_xlabel('Rain Rate (mm/h)')
    axs[0].set_yscale('log')
    rain_rate_values = rain_rate.values.flatten()
    rain_rate_values = rain_rate_values[~np.isnan(rain_rate_values)]
    print('max rain rate', np.amax(rain_rate_values) )

    #get channels value
    for i, ch in enumerate(channels):
        if ch in vis_channels:
            ch_values = msg_ds_day[ch][:]
            unit = 'Reflectance (%)'
            color= 'blue'
            min_value = 0
            max_value = 100
            bin_width = 2
        elif ch in ir_channels:
            ch_values = msg_ds[ch][:]
            unit = 'Brightness Temp (K)'
            color = 'green'
            min_value = 200
            max_value = 310
            bin_width = 2
        else:
            print('wrong channel name')
    
        axs[i+1].hist(ch_values.values.flatten(), bins=np.arange(min_value,max_value+bin_width,bin_width), color=color, alpha=0.7, density=True)
        axs[i+1].set_title('Channel '+ch, fontsize=10)
        axs[i+1].set_xlabel(unit)

    # Adjust layout
    fig.suptitle(f"MSG and Rain Rate Density Functions", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'distribution_histo.png', bbox_inches='tight')#, transparent=True)
    plt.close()



def plot_distr_rain(rain_rate_ds, path_out, min_value, max_value, bin_width):
    """
    Plots the distibution of rain rate. 

    Parameters:
    rain_rate_ds (str): Path to the NetCDF file containing the rain rate data.
    path_out (str): Path for saving the output plot.

    The function saves the resulting plot as a PNG file at the specified output path.
    """
    # Load data from NetCDF
    rain_rate = rain_rate_ds['rain_rate'][:]

    # Ensure min_value and max_value are positive and greater than 0 for log scale
    min_value = max(min_value, 1e-6)
    max_value = max(max_value, min_value + 1e-6)

    # Create log-spaced bins
    bins = np.logspace(np.log10(min_value), np.log10(max_value), int((np.log10(max_value) - np.log10(min_value)) / bin_width))

    # Create a figure with subplots
    fig, axs = plt.subplots(figsize=(10, 8))

    # Plot histogram with log-spaced bins
    axs.hist(rain_rate.values.flatten(), bins=bins, color='red', alpha=0.7, density=True)

    axs.set_title('NIMROD data', fontsize=10)
    axs.set_xlabel('Rain Rate (mm/h)')
    axs.set_yscale('log')
    axs.set_xscale('log')
    rain_rate_values = rain_rate.values.flatten()
    rain_rate_values = rain_rate_values[~np.isnan(rain_rate_values)]
    print('min rain rate', np.amin(rain_rate_values[rain_rate_values>0]) ) #min rain rate to consider rain is 0.1!

    # Adjust layout
    fig.suptitle(f"Rain rate Density Functions at low rates", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'distribution_histo_rain.png', bbox_inches='tight')#, transparent=True)
    plt.close()


def plot_channel_daily_trends(channels, ds, path_out, unit):
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
    ds_time = ds_time.rename({'end_time': 'time'})

    #resample to daily data and compute mean
    ds_time = ds_time.groupby('time.hour').mean('time')

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('VIS channels average daily trend' , fontsize=16, fontweight='bold')
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

    #draw horizontal line and fill the area between the vertical lines
    plt.axhline(y=5,xmin=0,xmax=24, color='black', linestyle='-')

    # Set the x-axis ticks and labels
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax.grid(visible= True, which='both', color ='grey', linestyle ='--', linewidth = 0.5, alpha = 0.8)

    # Set the axis labels and legend
    ax.set_xlabel('Hour (UTC)',fontsize=14)
    ax.set_ylabel(unit,fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save and show the figure
    fig.savefig(path_out+'AvgDailyTrends.png', bbox_inches='tight')#, transparent=True)
    plt.close()

    return None



def plot_single_temp_trend(rain_rate_ds, msg_ds, msg_ds_day, path_out, vis_channels, ir_channels, start_date, end_date):
    """
    Plots the temporal trend of rain rate alongside various visible (VIS) and infrared (IR) channels from MSG data on a multi-panel figure.

    This function creates a 3x4 grid of subplots. The first subplot displays the rain rate, followed by three VIS channels and eight IR channels from MSG data. 

    Parameters:
    rain_rate_ds (str): Path to the NetCDF file containing the rain rate data.
    msg_ds (str): Path to the NetCDF file containing MSG channel data.
    path_out (str): Path for saving the output plot.
    vis_channels (list): List of visible channel names in the MSG data.
    ir_channels (list): List of infrared channel names in the MSG data.

    The function saves the resulting plot as a PNG file at the specified output path.
    """
    #get start and end time
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d%H%M")  
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d%H%M") 

    #get full list of channels
    channels = vis_channels+ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axs = axs.flatten()

    #get channels value
    for i, ch in enumerate(channels):
        if ch in vis_channels:
            ch_values = msg_ds_day[ch]
            times = ch_values['end_time'].values
            #print(sorted(times))
            unit = 'Reflectance (%)'
            #color= 'blue'
            ch_extreme_xr = ch_values.max(dim=('x', 'y'), skipna=True) 
            label_extreme = 'max'
        elif ch in ir_channels:
            ch_values = msg_ds[ch]
            times = ch_values['end_time'].values
            unit = 'Brightness Temp (K)'
            #color = 'green'
            ch_extreme_xr = ch_values.min(dim=('x', 'y'), skipna=True) 
            label_extreme = 'min'
        else:
            print('wrong channel name')

        ch_mean_xr = ch_values.mean(dim=('x', 'y'), skipna=True) 
        mean_ch = ch_mean_xr.compute().values
        extreme_ch = ch_extreme_xr.compute().values

        mean_ch_full, time_data_full = replace_missing_intervals(start_date, end_date, datetime.timedelta(minutes=15),mean_ch,times )
        extreme_ch_full, time_data_full = replace_missing_intervals(start_date, end_date, datetime.timedelta(minutes=15),extreme_ch,times )
        time_data_full = [time.strftime('%m-%d %H') for time in time_data_full]

        axs[i+1].plot(range(len(time_data_full)), mean_ch_full, linestyle='-', marker='.', linewidth=1, alpha=0.7, label='mean')
        axs[i+1].plot(range(len(time_data_full)), extreme_ch_full, linestyle='--', marker='.', linewidth=1, alpha=0.7, label=label_extreme)
        axs[i+1].set_title('Channel '+ch, fontsize=10)
        axs[i+1].set_ylabel(unit)
        axs[i+1].set_xlabel('Time (UTC)')

        # Find the indices where the hours change in the time list
        indices = [i for i in range(1, len(time_data_full)) if time_data_full[i].split(' ')[0].split('-')[1] != time_data_full[i-1].split(' ')[0].split('-')[1]]

        # Set the x-axis ticks and labels
        axs[i+1].set_xticks(indices)
        axs[i+1].set_xticklabels([time_data_full[i] for i in indices], rotation=45, ha='right')

    # Open and plot rain data using the same time step of MSG
    rain_rate = rain_rate_ds['rain_rate']

    # Selecting values greater than 0.0
    rain_rate_above_zero = rain_rate.where(rain_rate > 0.0)

    # Compute the mean and max values, skipping NaNs
    mean_rain_xr = rain_rate_above_zero.mean(dim=('x', 'y'), skipna=True)
    max_rain_xr = rain_rate_above_zero.max(dim=('x', 'y'), skipna=True)

    # Convert the results to NumPy arrays
    mean_rain = mean_rain_xr.compute().values
    max_rain = max_rain_xr.compute().values

    #plot
    axs[0].plot(range(len(times)),mean_rain, linestyle='-', marker='.', linewidth=1, alpha=0.7, label='mean')
    axs[0].plot(range(len(times)),max_rain, linestyle='--', marker='.', linewidth=1, alpha=0.7, label='max')

    # Set the axis ticks and labels, title
    axs[0].set_title('NIMROD data', fontsize=10)
    axs[0].set_xlabel('Time (UTC)')
    axs[0].set_ylabel('Rain Rate (mm/h)')
    axs[0].set_yscale('log')
    axs[0].set_xticks(indices)
    axs[0].set_xticklabels([time_data_full[i] for i in indices], rotation=45, ha='right')

    # Adjust layout
    fig.suptitle(f"MSG and Rain Rate Temporal Trends", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    #plt.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(path_out+'temp_trends_subplots.png', bbox_inches='tight')#, transparent=True)
    plt.close()


def plot_channel_trends(channels, ds_msg, ds_rain, start_date, end_date, path_out, vis_channels, ir_channels):
    """
    Plot the mean temporal trend of each channel for a given list of times.

    Parameters:
    -----------
    channels : list of str
        List of channel names.
    ds: Dataset
        Xarray Dataset with the channels values
    start_date: str
        date of the beginning yyyymmddhh
    start_date: str
        date of the end yyyymmddhh
    path_out : str
        Directory to save the output figure.

    Returns:
    --------
    None
    """
    #get start and end time
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d%H%M")  
    end_date = datetime.datetime.strptime(end_date, "%Y%mexpected_delta%d%H%M") 

    # Set up the figure
    fig, ax1 = plt.subplots(figsize=(18, 6))
    fig.suptitle('channels and rain rate temporal trend: '+str(start_date.strftime('%Y/%m/%d'))+' - '+str(end_date.strftime('%Y/%m/%d')), fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.94,left=0.05)  # Adjust the space between the title and the plot

    #create twinx axis for the VIS channels
    ax2 = ax1.twinx()

    #create other for the rain data but placed on the right end on the plot
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    # Loop over channels
    for channel in channels:
        channel_data = ds_msg[channel] 

        # Calculate the mean value of the channel over the specified time steps
        mean_data_xr = channel_data.mean(dim=('x', 'y'), skipna=True) 
        mean_data = mean_data_xr.compute().values
        time_data = channel_data['end_time'].values
        
        #TODO get the full time sequence (nan where time interval is missing). Important in the case of VIS channels
        mean_data_full, time_data_full = replace_missing_intervals_old(start_date, end_date, datetime.timedelta(minutes=15),mean_data,time_data )
        time_data_full = [time.strftime('%m-%d %H') for time in time_data_full]
        #print(len(time_data_full), time_data_full)

        if channel in vis_channels:
            # Plot the mean temporal trend for the channel
            ax2.plot(range(len(time_data_full)), mean_data, linestyle='-', marker='.', linewidth=1, alpha=0.7, label=channel.split(' ')[-1])
            ax2.set_ylabel('Reflectances (%)',fontsize=14)
            ax2.set_ylim(0,100)
        elif channel in ir_channels:
            # Plot the mean temporal trend for the channel
            ax3.plot(range(len(time_data_full)),mean_data, linestyle='-', marker='.', linewidth=1, alpha=0.7, label=channel.split(' ')[-1])
            ax3.set_ylabel('Brightness Temperature (%)',fontsize=14)
            ax3.set_ylim(100,300)
        else:
            print('wrong channel name!')

    #open and plot rain data using the same time step of msg
    rain_rate = ds_rain['rain_rate']
    mean_rain_xr = rain_rate.mean(dim=('x', 'y'), skipna=True) 
    mean_rain = mean_rain_xr.compute().values

    ax1.plot(range(len(time_data_full)),mean_rain, linestyle='-', marker='.', linewidth=1, alpha=0.7, label='NIMROD')
    ax1.set_ylabel('Rain Rate (mm/h)',fontsize=14)
    ax1.set_ylim(0,10)

    # Find the indices where the hours change in the time list
    indices = [i for i in range(1, len(time_data_full)) if time_data_full[i].split(' ')[1] != time_data_full[i-1].split(' ')[1]]

    # Set the x-axis ticks and labels
    ax1.set_xticks(indices)
    ax1.set_xticklabels([time_data_full[i] for i in indices], rotation=45, ha='right')
    #ax1.set_xticklabels(np.array(time_data_full)[indices], rotation=45, ha='right')

    # Set the x-axis ticks and labels
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    ax1.grid(visible= True, which='both', color ='grey', linestyle ='--', linewidth = 0.5, alpha = 0.8)

    # Set the axis labels 
    ax1.set_xlabel('Time (UTC)',fontsize=14)

    #set the legend
    handles, labels = [], []

    for ax in [ax1, ax2, ax3]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1.1, 0.5))
    #ax1.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    # Save and show the figure
    fig.savefig(path_out+'temporal_trend.png', bbox_inches='tight')#, transparent=True)
    plt.close()

    return None


def replace_missing_intervals(start_date, end_date, expected_delta, channel_data, time_list):
    """
    Replaces missing intervals in the input channel data with NaN values,
    based on start and end dates, expected timedelta, and actual time list. 
    It checks if the difference between consecutive time steps is greater than 20 minutes.
    """
    # Convert to datetime.datetime objects
    time_list_dt = sorted([datetime.datetime.utcfromtimestamp(np_datetime64.astype('datetime64[s]').astype('int')) for np_datetime64 in time_list])
    #time_list_dt = sorted([[dt.astype(datetime.datetime) for dt in time_list]])
    #print(time_list_dt)

    # Create a list of consecutive dates with the given time step
    consecutive_dates = [start_date + i * expected_delta for i in range((end_date - start_date) // expected_delta + 1)]

    # Initialize the new channel data array with NaN values
    channel_data_new = np.full(len(consecutive_dates), np.nan)

    # Threshold for missing interval (20 minutes in this case), converted to minutes
    missing_interval_threshold = 5  # minutes

    # Iterate over the expected consecutive dates
    channel_data_idx = 0
    for i, expected_date in enumerate(consecutive_dates):
        # Check if we have more data to compare
        if channel_data_idx < len(time_list_dt):
            current_time = time_list_dt[channel_data_idx]
            #print(expected_date)
            #print(current_time)

            # Calculate the time difference in minutes
            time_difference = abs((current_time - expected_date).total_seconds()) / 60

            # Check if the time difference is within the threshold
            if time_difference <= missing_interval_threshold:
                channel_data_new[i] = channel_data[channel_data_idx]
                channel_data_idx += 1
            else:
                # The current time does not match the expected date, indicating a missing interval
                channel_data_new[i] = np.nan
        else:
            # No more channel data available, fill the rest with NaN
            channel_data_new[i] = np.nan

    return channel_data_new, consecutive_dates




def replace_missing_intervals_old(start_date, end_date, delta, channel_data, time_list):
    """
    Replaces missing intervals in the input channel data with NaN values, based on start and end dates and a timedelta.

    Parameters
    ----------
    start_date : datetime
        The first date in the desired output time range.
    end_date : datetime
        The last date in the desired output time range.
    delta : timedelta
        The time step between consecutive dates.
    channel_data : ndarray
        1D array of channel data, with shape (n_times,).
    time_list : list
        A list of dates as strings.

    Returns
    -------
    channel_data_new : ndarray
        1D array of channel data with missing intervals replaced with NaN values.
    consecutive_dates: datetime list
        list of the dates without missing intervals
    """

    time_list_dt = [datetime.datetime.strptime(t, '%Y%m%d%H') for t in time_list]

    # Create a list of consecutive dates with the given time step
    date_int = start_date
    consecutive_dates = []
    while date_int <= end_date:
        consecutive_dates.append(date_int)
        date_int += delta  # Make sure date_int is a datetime object, not a string

    # Initialize the new channel data array with NaN values
    channel_data_new = np.full(len(consecutive_dates), np.nan)

    # Replace missing intervals with the original channel data
    for i, date in enumerate(time_list_dt):
        if date in consecutive_dates:
            idx = consecutive_dates.index(date)
            channel_data_new[idx] = channel_data[i]

    return channel_data_new, consecutive_dates