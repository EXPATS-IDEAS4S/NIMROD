import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.colors as mcolors
import xarray as xr
import pandas as pd

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.quality_check_functions import assign_label_to_channel
from compare.comparison_function import get_max_min

def plot_rain_rate_vs_msg_channels(rain_rate_ds, msg_ds, bin_width_x, n_bin_y, vir_channels, ir_channels, path_out):
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
    #y_bins = np.arange(y_min, y_max + bin_width_y, bin_width_y)

    # Define logarithmically spaced bins for y axis (rain rate), considering non-zero minimum to avoid log(0)
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max), n_bin_y)
    
    # Loop through each channel and plot
    for i, channel_name in enumerate(channel_names):
        # Calculate the range and number of bins for x axis (channel values) based on the desired bin width
        x_min, x_max = get_max_min(msg_ds, channel_name)
        x_bins = np.arange(x_min, x_max + bin_width_x, bin_width_x)

        # Plotting the diagonal line
        #max_value = max(x_max, y_max)  
        #min_value = min(x_min, y_min) 
        #axs[i].plot([min_value, max_value], [min_value, max_value], 'r--')  # Red dashed line
        
        # Extract the data for the current channel
        channel_data = msg_ds[channel_name].values.flatten()
        
        # Create scatter plot in the corresponding subplot
        #axs[i].scatter(rain_rate_data, channel_data, alpha=0.5, marker='.', s=0.1)
        h, xedges, yedges, image = axs[i].hist2d(channel_data, rain_rate_data, bins=[x_bins, y_bins], cmap=custom_cmap)

        # Add a color bar to indicate the counts in bins
        cb = fig.colorbar(image, ax=axs[i])
        cb.set_label('counts')
        
        # Set subplot title to the channel name
        axs[i].set_title(f'{channel_name}')

        label = assign_label_to_channel(channel_name,vir_channels,ir_channels)

        #compute correlation excluding all the nan values
        valid_mask = ~np.isnan(channel_data) & ~np.isnan(rain_rate_data)  # Mask to filter out NaN values
        correlation_matrix = np.corrcoef(channel_data[valid_mask], rain_rate_data[valid_mask])

        # The Pearson correlation coefficient is at index [0, 1] and [1, 0] in the matrix
        pearson_corr = correlation_matrix[0, 1]

        # Or as a text annotation in the plot
        axs[i].text(0.4, 0.93, f'ρ={pearson_corr:.2f}', transform=axs[i].transAxes, color='blue')

        # Optionally, set x and y labels
        axs[i].set_ylabel('Rain Rate (mm/h)')
        axs[i].set_xlabel(label)
        axs[i].set_yscale('log')
        

    # Hide any unused subplots (if there's an extra one due to the grid size)
    for i in range(len(channel_names), len(axs)):
        axs[i].set_visible(False)

    # Adjust layout
    fig.suptitle(f"MSG channels and NIMROD Rain Rate 2D Histo", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    fig.savefig(path_out+'/MSG_NIMROD_2dhisto.png', bbox_inches='tight')#, transparent=True)
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
                axs[i+1].hist(class_data, bins=bins, alpha=0.7, density=True, label=rain_classes_name[n])
                
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


def plot_ets_trend(csv_file_path, path_out, vis_channels):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Set up the subplot grid
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(18, 10))
    axs = axs.flatten()

    # Get unique channels and rain threshold
    channels = df['Channel'].unique()
    rain_thresholds = df['Rain_Threshold'].unique()

    # Loop through the channels and plot
    for i, channel in enumerate(channels):
        unit = 'Reflectance (%)' if channel in vis_channels else 'Brightness Temp (K)'
        for rain_th in rain_thresholds:
            channel_data = df[(df['Channel'] == channel) & (df['Rain_Threshold'] == rain_th)]
            axs[i].plot(channel_data['MSG_Threshold'], channel_data['ETS'], marker='o', linestyle='-', label=str(rain_th))
        axs[i].set_title(channel)
        axs[i].set_xlabel(f'Channel Threshold [{unit}]')
        axs[i].set_ylabel('ETS')
        if i==2:
            axs[i].legend(loc='best', title="Rain threshold")

    # Hide any unused subplots (if there's an extra one due to the grid size)
    for i in range(len(channels), len(axs)):
        axs[i].set_visible(False)

    # Adjust layout and save the plot
    fig.suptitle("MSG Channel Distributions by Rain Class", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    fig.savefig(path_out + 'ets_trends_by_rain_class.png', bbox_inches='tight')
    plt.close(fig)