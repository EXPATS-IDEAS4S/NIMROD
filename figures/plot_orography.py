import sys
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import cartopy.crs as ccrs

sys.path.append('/home/dcorradi/Documents/Codes/MSG-SEVIRI/')
from figures.plotting_functions import plot_single_map, calc_channels_max_min, set_map_plot

#orography_path = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/orography_regridded_flood2021.nc"
orography_path = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/orography_flood2021_high_res.nc"
path_output = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/Fig/"

ds = xr.open_dataset(orography_path)
print(ds)

orography = ds['orography'].values
lats = ds['lats'].values
lons = ds['lons'].values
print(orography)

#lats, lons = np.meshgrid(lats, lons, indexing='ij')
print(lats)
print(lons)

terrain_cmap = plt.get_cmap('terrain')
newcolors = terrain_cmap(np.linspace(0.25, 1, 256))  # Skipping blue gradients
#blue = np.array([0, 0, 1, 1])  # RGBA for blue
#newcolors[:25, :] = blue  # Assign blue to the lowest values (assuming these represent sea level and below)
cmap = mcolors.ListedColormap(newcolors)

vmin = np.amin(orography[~np.isnan(orography)])
vmax = np.amax(orography[~np.isnan(orography)])
print(vmin,vmax)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
lonmin, latmin, lonmax, latmax= 5, 48, 9, 52

# Plotting setup
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())  # Adjust this based on your projection

set_map_plot(ax,norm,cmap,[lonmin,lonmax,latmin,latmax],'orography','elevation (m)')

# Plot
mesh = ax.contourf(lons, lats, orography, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap)

plt.title('orography', fontsize=12, fontweight='bold')
fig.savefig(path_output+'orography_flood2021.png', bbox_inches='tight')
plt.close()

#plot_single_map(orography,lons,lats,cmap,norm,[lonmin,lonmax,latmin,latmax],'2021 Flood Domain','orography','elevation (m)',path_output) #[left, right, bottom ,top]

filtered_ds = ds.where(
    (ds.lons >= lonmin) & (ds.lons <= lonmax) & 
    (ds.lats >= latmin) & (ds.lats <= latmax), 
    drop=True
)
orography = filtered_ds['orography'].values

plt.hist(orography.flatten(), bins=50)  # Setting 'kde=True' adds a Kernel Density Estimate plot on top
plt.title('Orography Distribution')
plt.xlabel('Elevation (m)')
plt.ylabel('Counts')
plt.yscale('log')
plt.savefig(path_output+'orography_distribution_original-res-FloodDomain.png', dpi=300, bbox_inches='tight')
plt.close()


