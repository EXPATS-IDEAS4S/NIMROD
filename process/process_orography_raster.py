"""
script to produce ncdf file containing orography for the expats domain at highest resolution. Data are
taken from USGS repository and can be accessed from https://topotools.cr.usgs.gov/gmted_viewer/viewer.htm
https://topotools.cr.usgs.gov/GMTED_viewer/gmted2010_fgdc_metadata.html

"""
import rasterio
import numpy as np
import rasterio
from affine import Affine
from pyproj import Proj, transform
import xarray as xr
import sys

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from compare.config_compare import lonmin,lonmax,latmin,latmax
from process_functions import regrid_data

reg_lats = "/net/yube/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/reg_lats.npy"
reg_lons = "/net/yube/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/reg_lons.npy"

raster_filename_1 = '/net/ostro/50N000E_20101117_gmted_max075.tif'
raster_filename_2 = '/net/ostro/30N000E_20101117_gmted_max075.tif'
#raster_filename = '/net/yube/ESSL/NE1_HR_LC_SR_W_DR/NE1_HR_LC_SR_W_DR.tif'

orography_path = "/work/dcorradi/orography_flood2021_high_res.nc" #"/net/ostro/figs_proposal_TEAMX/orography_expats_high_res.nc"

path_outputs = '/work/dcorradi/'

#lonmin,lonmax,latmin,latmax = 5.,   16.,    42.,   51.5

def main():
    
    # read first dataset 
    ds_1 = raster_to_dataset(raster_filename_1)
    #print(ds)
    ds_2 = raster_to_dataset(raster_filename_2)

    # redefining coordinate of dataset 2 to be able to concatenate 
    ds_2['y'] = np.arange(9600,9600+len(ds_2.lats.values))

    # concatenate the two domains
    ds = xr.concat([ds_1, ds_2], dim="y")
    
    # select domain expats
    ds_expats = ds.where((ds.lats > latmin) * (ds.lats < latmax) * (ds.lons > lonmin) * (ds.lons < lonmax), drop=True)

    # store to ncdf
    ds_expats.to_netcdf(path_outputs+'orography_flood2021_high_res.nc')

def raster_to_dataset(raster_file):
    """
    function to convert raster object in xarray dataset

    Args:
        raster_file (string): filename including path of the raster file
    Returns:
        ds_out : xarray dataset containing input data
    """
    # Read raster
    with rasterio.open(raster_file) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    y, x = np.shape(longs)

    
    # store data in xarray dataset
    ds_out = xr.Dataset(
        data_vars = {'orography':(('y', 'x'), A[0, :, :]), 
                    'lons':(('y', 'x'), lats),
                    'lats':(('y', 'x'), longs),
                    }, 
        coords = {'y':(('y'), np.arange(y)), 
                'x':(('x'), np.arange(x)), 
                }
    )
    return ds_out

def regrid(orography_path, reg_lats, reg_lons, path_outputs):
    """
    Regrids orography data from its original latitude and longitude points
    to a new set of latitude and longitude points using linear interpolation.
    
    Parameters:
    - orography_path (str): Path to the input netCDF file containing the original orography data.
    - reg_lats (str): Path to the numpy file containing the target latitude points.
    - reg_lons (str): Path to the numpy file containing the target longitude points.
    - path_outputs (str): Output directory path where the regridded netCDF will be saved.

    The function loads the original orography data, performs regridding to the target
    lat/lon points, and saves the regridded data as a new netCDF file in the specified output path.
    """

    # Load target latitude and longitude points from numpy files
    lat_points = np.load(reg_lats)
    lon_points = np.load(reg_lons)

    # Generate a meshgrid for the target lat and lon points for regridding
    msg_lat_grid, msg_lon_grid = np.meshgrid(lat_points, lon_points, indexing='ij')

    # Open the original orography dataset
    orography_ds = xr.open_dataset(orography_path)

    # Flatten the original lat, lon, and orography values for regridding
    orography = orography_ds['orography'].values.flatten()
    lats = orography_ds['lats'].values.flatten()
    lons = orography_ds['lons'].values.flatten()

    # Perform regridding from original lat/lon to target lat/lon grid
    orography_regrid = regrid_data(lats, lons, orography, msg_lat_grid, msg_lon_grid, method='linear')

    # Create a new xarray Dataset for the regridded orography
    ds_oro = xr.Dataset()

    # Create a DataArray for the regridded orography, specifying its dimensions and coordinates
    oro_da = xr.DataArray(
                orography_regrid,
                dims=("y", "x"),
                coords={"lat": ("y", lat_points), "lon": ("x", lon_points)},
                name='orography')

    # Add the DataArray to the Dataset
    ds_oro["orography"] = oro_da

    # Save the regridded orography data to a netCDF file
    ds_oro.to_netcdf(path_outputs)

if __name__ == "__main__":
    #main()
    regrid(orography_path,reg_lats,reg_lons, path_outputs+'orography_regridded_flood2021.nc')