import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.quality_check_functions import assign_label_to_channel
from compare.comparison_function import get_max_min

def plot_rain_rate_vs_msg_channels(rain_rate_ds, msg_ds, bin_width_x, bin_width_y, vir_channels, ir_channels, path_out):
    """
    Plot scatter plots comparing rain rate with each of the 11 MSG channels.
    
    Parameters:
    - rain_rate_ds (y): xarray.Dataset containing the rain rate data.
    - msg_ds (x): xarray.Dataset containing the MSG channel data.
    - bin_width_x: float, desired bin width along the x-axis.
    - bin_width_y: float, desired bin width along the y-axis.
    """
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
    
    # Loop through each channel and plot
    for i, channel_name in enumerate(channel_names):
        # Calculate the range and number of bins for x axis (channel values) based on the desired bin width
        x_min, x_max = get_max_min(msg_ds, channel_name)
        x_bins = np.arange(x_min, x_max + bin_width_x, bin_width_x)
        
        # Extract the data for the current channel
        channel_data = msg_ds[channel_name].values.flatten()
        
        # Create scatter plot in the corresponding subplot
        #axs[i].scatter(rain_rate_data, channel_data, alpha=0.5, marker='.', s=0.1)
        h, xedges, yedges, image = axs[i].hist2d(channel_data, rain_rate_data, bins=[x_bins, y_bins], cmap='Blues')

        # Add a color bar to indicate the counts in bins
        cb = axs[i].colorbar(image)
        cb.set_label('counts')
        
        # Set subplot title to the channel name
        axs[i].set_title(f'{channel_name}')

        label = assign_label_to_channel(channel_name,vir_channels,ir_channels)
        
        # Optionally, set x and y labels
        axs[i].set_xlabel('Rain Rate (mm/h)')
        axs[i].set_ylabel(label)
        

    # Hide any unused subplots (if there's an extra one due to the grid size)
    for i in range(len(channel_names), len(axs)):
        axs[i].set_visible(False)

    # Adjust layout
    fig.suptitle(f"MSG channels and NIMROD Rain Rate 2D Histo", fontsize=12, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.4)  
    plt.tight_layout()
    fig.savefig(path_out+'/MSG_NIMROD_2dhisto.png', bbox_inches='tight')#, transparent=True)
    plt.close()
    

