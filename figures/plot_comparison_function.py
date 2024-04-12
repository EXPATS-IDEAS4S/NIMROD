import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.colors as mcolors
import xarray as xr
import pandas as pd
import matplotlib.colors as colors
import cartopy.crs as ccrs

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.quality_check_functions import assign_label_to_channel, assign_cmap_to_channel
from compare.comparison_function import get_max_min, filter_rain_rate, find_max_ets_threshold, calculate_categorical_metrics, filter_elevation
from figures.plot_functions import set_map_plot

def plot_rain_rate_vs_msg_channels(rain_rate_ds, msg_ds, bin_width_x, bin_width_y, vir_channels, ir_channels, path_out, rain_class_name):
    """
    Plot scatter plots comparing rain rate with each of the 11 MSG channels.
    
    Parameters:
    - rain_rate_ds (y): xarray.Dataset containing the rain rate data.
    - msg_ds (x): xarray.Dataset containing the MSG channel data.
    - bin_width_x: float, desired bin width along the x-axis.
    - bin_width_y: float, desired bin width along the y-axis.
    """
    # Create a custom colormap that is white for zero and uses 'viridis' for other values
    viridis = plt.cm.get_cmap('viridis', 256)
    colors = np.vstack([[1, 1, 1, 0], viridis(np.arange(256))])  # Adding white color for zero
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_viridis', colors)

    # Complete list of channels
    channel_names = vir_channels+ir_channels
    
    # Create a 3x4 subplot grid
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    
    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Extract the rain rate data, ensuring it matches the shape of the channel data
    rain_rate_data = rain_rate_ds['rain_rate'].values.flatten()

    #Calculate the range and number of bins for y axis (rain rate) based on the desired bin width
    y_min, y_max = get_max_min(rain_rate_ds,'rain_rate')
    y_bins = np.arange(y_min, y_max + bin_width_y, bin_width_y)

    # Define logarithmically spaced bins for y axis (rain rate), considering non-zero minimum to avoid log(0)
    #y_bins = np.logspace(np.log10(y_min), np.log10(y_max), n_bin_y)
    
    # Loop through each channel and plot
    for i, channel_name in enumerate(channel_names):
        # Calculate the range and number of bins for x axis (channel values) based on the desired bin width
        x_min, x_max = get_max_min(msg_ds, channel_name)
        x_bins = np.arange(x_min, x_max + bin_width_x[i], bin_width_x[i])

        # Plotting the diagonal line
        #max_value = max(x_max, y_max)  
        #min_value = min(x_min, y_min) 
        #axs[i].plot([min_value, max_value], [min_value, max_value], 'r--')  # Red dashed line
        
        # Extract the data for the current channel
        channel_data = msg_ds[channel_name].values.flatten()
        
        # Create scatter plot in the corresponding subplot
        #axs[i].scatter(rain_rate_data, channel_data, alpha=0.5, marker='.', s=0.1)
        h, xedges, yedges, image = axs[i].hist2d(channel_data, rain_rate_data, bins=[x_bins, y_bins], cmap=custom_cmap, cmin=1)

        # Add a color bar to indicate the counts in bins
        cb = fig.colorbar(image, ax=axs[i])
        cb.set_label('counts')

        label = assign_label_to_channel(channel_name,vir_channels,ir_channels)

        #compute correlation excluding all the nan values
        valid_mask = ~np.isnan(channel_data) & ~np.isnan(rain_rate_data)  # Mask to filter out NaN values
        correlation_matrix = np.corrcoef(channel_data[valid_mask], rain_rate_data[valid_mask])

        # The Pearson correlation coefficient is at index [0, 1] and [1, 0] in the matrix
        pearson_corr = correlation_matrix[0, 1]

        # Or as a text annotation in the plot
        #axs[i].text(0.4, 0.93, f'ρ={pearson_corr:.2f}', transform=axs[i].transAxes, color='blue')

        # Optionally, set x and y labels
        axs[i].set_ylabel('Rain Rate (mm/h)')
        axs[i].set_xlabel(label)
        #axs[i].set_yscale('log')

        # Set subplot title to the channel name
        axs[i].set_title(f'{channel_name} - ρ={pearson_corr:.2f}')
        

    # Hide any unused subplots (if there's an extra one due to the grid size)
    for i in range(len(channel_names), len(axs)):
        axs[i].set_visible(False)

    # Adjust layout
    fig.suptitle(f"MSG channels vs {rain_class_name} Rain Rate 2D Histo", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    fig.savefig(path_out+'/MSG_NIMROD_2dhisto_'+rain_class_name+'.png', bbox_inches='tight')#, transparent=True)
    plt.close()
    


def plot_distributions_by_rain_classes(rain_rate_ds, msg_ds, path_out, vis_channels, ir_channels, rain_classes, rain_classes_name, bin_widths):
    """
    Plots the distribution of brightness temperature or reflectance across different rain classes for various MSG channels.
    """
    channels = vis_channels + ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axs = axs.flatten()
    
    # Plot rain rate histogram with vertical lines for rain classes
    rain_rate_values = rain_rate_ds['rain_rate'].values.flatten()
    rain_rate_values = rain_rate_values[~np.isnan(rain_rate_values)]
    #print(np.shape(rain_rate_values),rain_rate_values)
    #bins = np.logspace(0.01, np.log10( rain_rate_values.max()), 50)
    #bins = np.arange(rain_rate_values.min(), rain_rate_values.max() + bin_width, bin_width)
    # Create bins with a custom approach
    # Start with a small bin for zero to a small value
    bins = [0, bin_widths[0] / 10]
    # Add bins exponentially growing to cover the range of data
    current_bin = bins[-1]
    while current_bin < rain_rate_values.max():
        current_bin += bin_widths[0]
        bins.append(current_bin)
    # Convert to an array for histogramming
    bins = np.array(bins)
    axs[0].hist(rain_rate_values, bins=bins, color='red', alpha=0.7, density=True)
    axs[0].set_title('Rain Rate Distribution', fontsize=10)
    axs[0].set_xlabel('Rain Rate (mm/h)')
    axs[0].set_yscale('log')
    axs[0].set_xscale('symlog')
    axs[0].set_xlim(-0.1)
    axs[0].set_xticks(ticks=[0.1, 2.5, 10, 80], labels=['0.1', '2.5', '10', '80'])
    
    # Draw vertical lines and add text labels for rain classes
    for i, rain_class in enumerate(rain_classes):
        axs[0].axvline(x=rain_class[1], color='k', linestyle='--')
        # Calculate position for text label (midpoint of the class range)
        if i < len(rain_classes) - 1:
            text_x_pos = 10**((np.log10(rain_class[1]) + np.log10(rain_classes[i + 1][0])) / 2)
        else:
            # For the last class, position the text a bit differently to avoid placing it too far to the right
            text_x_pos = rain_class[1]
        axs[0].text(text_x_pos, 0.5, rain_classes_name[i], rotation=90, verticalalignment='bottom', horizontalalignment='right', transform=axs[0].get_xaxis_transform(), color='k')
    
    # Plot distributions for each channel divided by rain classes
    for i, ch in enumerate(channels):
        #channel_data = msg_ds[ch][:]
        unit = 'Reflectance (%)' if ch in vis_channels else 'Brightness Temp (K)'
        #color = 'blue' if ch in vis_channels else 'green'
        ch_min, ch_max = get_max_min(msg_ds,ch)
        bins = np.arange(ch_min, ch_max + bin_widths[i+1], bin_widths[i+1])
        
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
                axs[i+1].hist(class_data, bins=bins, alpha=0.7, density=True, label=rain_classes_name[n], histtype='step', linewidth=2)
                
                
        axs[i+1].set_title(f'Channel {ch}', fontsize=10)
        axs[i+1].set_xlabel(unit)
        if i==2:
            axs[i+1].legend(title="Rain classes")
    
    # Adjust layout and save the plot
    fig.suptitle("MSG Channel Distributions by Rain Class", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    fig.savefig(path_out + 'distribution_by_rain_class.png', bbox_inches='tight')
    plt.close(fig)


def plot_distributions_by_elevation_classes(rain_rate_ds, msg_ds, elev_ds, path_out, vis_channels, ir_channels, el_classes, el_classes_name, bin_widths):
    """
    Plots the distribution of brightness temperature or reflectance across different rain classes for various MSG channels.
    """
    channels = vis_channels + ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axs = axs.flatten()
    
    # Plot distributions for each channel divided by rain classes
    for i in range(len(channels)+1):
        if i==0:
            ch_min, ch_max = get_max_min(rain_rate_ds,'rain_rate')
            bins = np.arange(ch_min, ch_max + bin_widths[i], bin_widths[i])

            # Plot distributions for each rain class
            for n, el_class in enumerate(el_classes):
                print('rain class', el_class)
                class_mask = (elev_ds['orography'] >= el_class[0]) & (elev_ds['orography'] < el_class[1])
                #print(class_mask)
                #print(msg_ds[ch])
                print("True values in mask:", np.sum(class_mask.values))           
                # Applying mask and dropping NaNs explicitly
                class_data = rain_rate_ds['rain_rate'].where(class_mask)
                print("Class data after masking:", class_data)
                class_data = class_data.values.flatten()
                #print("Non-NaN values count:", np.count_nonzero(~np.isnan(class_data)))
                
                # Avoid plotting empty data
                if len(class_data) > 0:
                    axs[i].hist(class_data, bins=bins, alpha=0.7, density=True, label=el_classes_name[n], histtype='step', linewidth=2)      
                    
            axs[i].set_title('Rain Rate', fontsize=10)
            axs[i].set_xlabel('rain rate (mm/h)')
            axs[i].set_yscale('log')
        else:
            #channel_data = msg_ds[ch][:]
            unit = 'Reflectance (%)' if channels[i-1] in vis_channels else 'Brightness Temp (K)'
            #color = 'blue' if ch in vis_channels else 'green'
            ch_min, ch_max = get_max_min(msg_ds,channels[i-1])
            bins = np.arange(ch_min, ch_max + bin_widths[i], bin_widths[i])
            
            # Plot distributions for each rain class
            for n, el_class in enumerate(el_classes):
                print('rain class', el_class)
                class_mask = (elev_ds['orography'] >= el_class[0]) & (elev_ds['orography'] < el_class[1])
                #print(class_mask)
                #print(msg_ds[ch])
                print("True values in mask:", np.sum(class_mask.values))           
                # Applying mask and dropping NaNs explicitly
                class_data = msg_ds[channels[i-1]].where(class_mask)
                print("Class data after masking:", class_data)
                class_data = class_data.values.flatten()
                #print("Non-NaN values count:", np.count_nonzero(~np.isnan(class_data)))
                
                # Avoid plotting empty data
                if len(class_data) > 0:
                    axs[i].hist(class_data, bins=bins, alpha=0.7, density=True, label=el_classes_name[n], histtype='step', linewidth=2)
                    
                    
            axs[i].set_title(f'Channel {channels[i-1]}', fontsize=10)
            axs[i].set_xlabel(unit)
            if i==3:
                axs[i].legend(title="Elevation classes")
    
    # Adjust layout and save the plot
    fig.suptitle("MSG Channel Distributions by Elevation Class", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    fig.savefig(path_out + 'distribution_by_elev_class.png', bbox_inches='tight')
    plt.close(fig)


def plot_ets_trend_rain_threshold(path_out, vis_channels, elev_class):
    # Load the CSV file
    df = pd.read_csv(path_out+"ets_results_"+elev_class+".csv")

    # Set up the subplot grid
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
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
        unit = 'Reflectance (%)' if channel in vis_channels else 'Brightness Temp (K)'
        for j, rain_th in enumerate(rain_thresholds):
            channel_data = df[(df['Channel'] == channel) & (df['Rain_Threshold'] == rain_th)]
            axs[i].plot(channel_data['MSG_Threshold'], channel_data['ETS'], marker='.', linestyle='-', color=colors[j])#, label=str(rain_th))
        axs[i].set_title(channel)
        axs[i].set_xlabel(f'Channel Threshold in {unit}')
        axs[i].set_ylabel('ETS')
        axs[i].grid(True)
        #if i==3:
        #    axs[i].legend(loc='best', title="Rain threshold")

    # Hide any unused subplots (if there's an extra one due to the grid size)
    #for i in range(len(channels), len(axs)):
    #    axs[i].set_visible(False)

    # Turn off the axis for the 12th subplot
    axs[-1].axis('off')

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


def plot_ets_trend_elev_class(path_out,channels, vis_channels, rain_threshold, elev_classes):

    # Set up the subplot grid
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axs = axs.flatten()

    # Use a qualitative colormap
    cmap = plt.get_cmap('Dark2')

    # Generate diverse colors
    colors = [cmap(i) for i in range(len(elev_classes))]

    # Loop through the channels and plot
    for i, channel in enumerate(channels):
        unit = 'Reflectance (%)' if channel in vis_channels else 'Brightness Temp (K)'
        for j, el_clas in enumerate(elev_classes):
            # Load the CSV files
            df = pd.read_csv(path_out+"ets_results_"+el_clas+".csv")
            channel_data = df[(df['Channel'] == channel) & (df['Rain_Threshold'] == rain_threshold)]
            axs[i].plot(channel_data['MSG_Threshold'], channel_data['ETS'], marker='.', linestyle='-', color=colors[j])#, label=str(rain_th))
        axs[i].set_title(channel)
        axs[i].set_xlabel(f'Channel Threshold in {unit}')
        axs[i].set_ylabel('ETS')
        axs[i].grid(True)
        #if i==3:
        #    axs[i].legend(loc='best', title="Rain threshold")

    # Hide any unused subplots (if there's an extra one due to the grid size)
    #for i in range(len(channels), len(axs)):
    #    axs[i].set_visible(False)

    # Turn off the axis for the 12th subplot
    axs[-1].axis('off')

    # Create a custom legend for the whole figure
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=4, label=str(elev_classes[0])+' 0-200 m'),
                    plt.Line2D([0], [0], color=colors[1], lw=4, label=str(elev_classes[1])+' 200-600 m'),
                    plt.Line2D([0], [0], color=colors[2], lw=4, label=str(elev_classes[2])+' 600+ m')]

    # Place the legend on the 12th subplot's axes
    axs[-1].legend(handles=legend_elements, loc='center', title="Elevation Classes")


    # Adjust layout and save the plot
    fig.suptitle("ETS trend - "+str(rain_threshold), fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    fig.savefig(path_out + 'ets_trends_by_elev_class_'+str(rain_threshold)+'.png', bbox_inches='tight')
    plt.close(fig)


def plot_spatial_percentile(msg_ds,rain_ds, elevation_ds, percentiles, vis_channels, ir_channels, extent, path_out):
    """
    Plot the specified percentile of spatial data over time on a geographical map.

    Parameters:
    - data (xarray.DataArray): The spatial data (must include lat and lon dimensions).
    - percentile (int): The percentile to calculate and plot.
    - time_step (str or None): Specific time step to plot. If None, uses all time steps.
    """
    channels = vis_channels+ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
        
    # Plot rain rate
    lats = rain_ds.coords['lat'].values
    lons = rain_ds.coords['lon'].values
    lat_grid, lon_grid = np.meshgrid(lats,lons,indexing='ij')
    # Calculate the percentile value across time dimension
    ds_rain_rechunked = rain_ds.chunk({'time': -1})
    rain_rate = ds_rain_rechunked['rain_rate'].quantile(percentiles[1]/100.0, dim='time', method='linear').values.squeeze()

    # set up elevation
    elevation = elevation_ds['orography'].values  
    el_min, el_max = get_max_min(elevation_ds, 'orography')
    contour_levels = np.linspace(el_min, el_max, num=10)  # Adjust number of levels as needed

    #set up plot 
    cmap = plt.cm.gist_ncar.copy()
    vmin = rain_rate.min()
    vmax = rain_rate.max()
    #norm = mcolors.PowerNorm(gamma=0.4)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    axs[0].contourf(lon_grid, lat_grid, rain_rate, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())#, extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))
    set_map_plot(axs[0],norm,cmap,extent,'Percentile '+str(percentiles[1])+' - Rain Rate','Rain Rate (mm/h)')
    axs[0].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    #get channels value
    for i, ch in enumerate(channels):
        percentile = percentiles[1] if ch in vis_channels else percentiles[0]
        ds_msg_rechunked = msg_ds.chunk({'time': -1})
        ch_values = ds_msg_rechunked[ch].quantile(percentile/100.0, dim='time', method='linear').values.squeeze()
        #ch_values = np.flip(ch_values[0,:,:]) 
        vmin = ch_values.min()
        vmax = ch_values.max() 

        #set cmap, label and normalize the colorbar
        cmap = assign_cmap_to_channel(ch,vis_channels,ir_channels)
        unit = assign_label_to_channel(ch,vis_channels,ir_channels)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        set_map_plot(axs[i+1],norm,cmap,extent,'Percentile '+str(percentile)+' - Ch '+ch,unit)        
        axs[i+1].contourf(lons, lats, ch_values, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap)
        axs[i+1].set_facecolor('#dcdcdc')

        axs[i+1].contour(lons, lats, elevation, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Adjust layout
    fig.suptitle("MSG and Rain Rate Data Percentiles", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'multichannels_map_percentiles_'+str(percentiles)+'.png', bbox_inches='tight')#, transparent=True)
    plt.close()



    



def plot_spatial_metrics(msg_ds,rain_ds, metric, rain_threshold, vis_channels, ir_channels, extent, path_out):
    """
    Plot the specified metric of spatial data over time on a geographical map.

    Parameters:
    - data (xarray.DataArray): The spatial data (must include lat and lon dimensions).
    - metric (str): The metric to calculate and plot.
    - time_step (str or None): Specific time step to plot. If None, uses all time steps.
    """
    channels = vis_channels+ir_channels

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
        
    # Plot rain rate
    lats = rain_ds.coords['lat'].values
    lons = rain_ds.coords['lon'].values
    lat_grid, lon_grid = np.meshgrid(lats,lons,indexing='ij')
    # Calculate the percentile value across time dimension
    ds_rain_rechunked = rain_ds.chunk({'time': -1})
    rain_rate = ds_rain_rechunked['rain_rate'].max(dim='time').values.squeeze()

    #set up plot 
    cmap = plt.cm.gist_ncar.copy()
    #vmin = rain_rate.min()
    #vmax = rain_rate.max()
    norm = mcolors.PowerNorm(gamma=0.4)
    #norm = colors.Normalize(vmin=vmin, vmax=vmax)
    axs[0].contourf(lon_grid, lat_grid, rain_rate, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))
    set_map_plot(axs[0],norm,cmap,extent,'Max Rain Rate','Rain Rate (mm/h)')
    
    # Filter data variables that end with 'ets'
    metric_variables = {name: var for name, var in msg_ds.data_vars.items() if name.endswith('_'+metric)}

    # Combine the filtered variables into a new dataset
    metric_ds = xr.Dataset(metric_variables)
    #print(metric_ds)

    # Compute the max and min across all 'ets' variables
    vmax = metric_ds.to_array().max()
    vmin = metric_ds.to_array().min()

    print(vmin,vmax)

    #find norm to apply at colorbar
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    #get channels value
    for i, channel in enumerate(channels):
        print('\nchannel: ', channel)
        
        ch_values = msg_ds[channel+'_'+metric].values.squeeze()

        #set cmap, label and normalize the colorbar
        cmap = 'cividis' #assign_cmap_to_channel(ch,vis_channels,ir_channels)
        unit = metric #assign_label_to_channel(ch,vis_channels,ir_channels)
        
        set_map_plot(axs[i+1],norm,cmap,extent,'Ch '+channel,unit)        
        axs[i+1].contourf(lons, lats, ch_values, norm=norm, transform=ccrs.PlateCarree(), cmap=cmap)
        axs[i+1].set_facecolor('#dcdcdc')

    # Adjust layout
    fig.suptitle("Metric: "+metric+' - Rain Threshold: '+str(rain_threshold), fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    #plt.show()
    fig.savefig(path_out+'metrics_map_'+metric+'_'+str(rain_threshold)+'.png', bbox_inches='tight')#, transparent=True)
    plt.close()