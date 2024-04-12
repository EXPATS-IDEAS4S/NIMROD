import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm
from matplotlib.patches import Patch


def plot_rain_data(rain_data, lat, lon, time, title,extent, save=None):
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
    
    # Plot the data with the custom colormap
    norm = mcolors.PowerNorm(gamma=0.4)
    ax.contourf(lon, lat, rain_data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=0.4, levels=np.linspace(0, 100, 500))

    set_map_plot(ax,norm,cmap,extent,'Rain Rate Plot '+str(time)+' - '+title,'Rain Rate (mm/h)')

    # Save the plot
    if save:
        date_string = time.strftime('%Y-%m-%d %H:%M')
        plot_filename = save+'rain_rate_'+date_string.replace(' ','_')+'_'+title+'.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        plt.close()



def set_map_plot(ax, norm, cmap, extent, plot_title, label, colorbar=True):
    # Create a ScalarMappable with the normalization and colormap
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Add color bar
    if colorbar:
        plt.colorbar(sm, ax=ax, orientation='vertical', label=label, shrink=0.8)

    #set axis thick labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.75, color='gray', alpha=0.6, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Adds coastlines and borders to the current axes
    ax.coastlines(resolution='50m', linewidths=0.5) 
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.set_extent(extent) #[left, right, bottom ,top]

    ax.set_title(plot_title, fontsize=12, fontweight='bold')



def plot_cma(data, lat, lon, time, title, extent, save=None):
    """
    Plots cloud mask data (0 for clear, 1 for cloudy) on a map using a binary color map with Matplotlib and Cartopy.
    Adds geographical features for context, and optionally saves the plot to a file.

    Parameters:
    - data (numpy.ndarray): 2D array of cloud mask data (0 for clear, 1 for cloudy).
    - lat (numpy.ndarray): 2D array of latitude values.
    - lon (numpy.ndarray): 2D array of longitude values.
    - time (datetime or str): Time corresponding to the data snapshot.
    - title (str): Title for the plot.
    - extent (list): Geographical extent [west, east, south, north] for the plot.
    - save (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    # Set up the binary colormap: 0 (clear) as blsck, 1 (cloudy) as white
    cmap = mcolors.ListedColormap(['black', 'white'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    #ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot the cloud mask data
    mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    set_map_plot(ax,norm,cmap,extent,title+': '+str(time),'',False)

    # Add a legend for cloud mask
    legend_labels = [Patch(facecolor='white', edgecolor='black', label='Cloudy'),
                     Patch(facecolor='black', edgecolor='black', label='Clear')]
    ax.legend(handles=legend_labels, loc='lower left', title="")

    # Save the plot
    if save:
        date_string = time.strftime('%Y-%m-%d %H:%M')
        plot_filename = save+title+'_'+date_string.replace(' ','_')+'_'+title+'.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        plt.close()


def plot_cot(data, lat, lon, time, title, extent, cmap, norm, save=None):
    """
    Plots COT.
    Adds geographical features for context, and optionally saves the plot to a file.

    Parameters:
    - data (numpy.ndarray): 2D array of cloud mask data (0 for clear, 1 for cloudy).
    - lat (numpy.ndarray): 2D array of latitude values.
    - lon (numpy.ndarray): 2D array of longitude values.
    - time (datetime or str): Time corresponding to the data snapshot.
    - title (str): Title for the plot.
    - extent (list): Geographical extent [west, east, south, north] for the plot.
    - save (str, optional): Path to save the plot image. If None, the plot is not saved.
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    #ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot the cloud mask data
    mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    set_map_plot(ax,norm,cmap,extent,title+': '+str(time),'',True)

    # Save the plot
    if save:
        date_string = str(time)#.strftime('%Y-%m-%d %H:%M')
        plot_filename = save+title+'_'+date_string.replace(' ','_')+'.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
        plt.close()


def savefile(fig,plot_filename):
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")
    plt.close()





