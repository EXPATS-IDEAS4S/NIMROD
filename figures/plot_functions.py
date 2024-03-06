import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_data(rain_data, lat, lon, time, title, save=None):
    """
    Plots rainfall data on a map with a custom color map using Matplotlib and Cartopy.
    This function creates a figure showing the `rain_data` on a map, using a custom colormap for the rain rates
    and adding geographical features for context. The plot includes coastlines, country borders, and gridlines with labels. 
    The plot can optionally be saved to a PNG file if a path is provided through the `save` parameter. 

    Parameters:
    - rain_data (numpy.ndarray): A 2D array of rain rate data to be plotted, with dimensions matching `lat` and `lon`.
    - lat (numpy.ndarray): A 2D array of latitude values corresponding to `rain_data`.
    - lon (numpy.ndarray): A 2D array of longitude values corresponding to `rain_data`.
    - time (datetime or string): The time corresponding to the `rain_data` snapshot. If it's a datetime object, it will be formatted; if it's a string, it will be used as is.
    - title (str): The title to be displayed on the plot.
    - save (str, optional): The directory path where the plot image will be saved. If not provided, the plot will not be saved to a file.    
    """
    # Create a custom colormap
    cmap = plt.cm.gist_ncar.copy()

    # Create a plot with Cartopy
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linestyle=':')

    # Plot the data with the custom colormap
    norm = mcolors.PowerNorm(gamma=0.4)
    mesh = ax.contourf(lon, lat, rain_data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))
    
    # Add colorbar with reduced size
    cbar = plt.colorbar(mesh, label='Rain Rate (mm/h)', shrink=0.5)

    plt.title(f'Rain Rate Plot '+str(time)+' - '+title)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')

    #set axis thick labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.75, color='white', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Save the plot
    if save:
        date_string = time.strftime('%Y-%m-%d %H:%M')
        plot_filename = save+'rain_rate_'+date_string.replace(' ','_')+'_'+title+'.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        plt.close()