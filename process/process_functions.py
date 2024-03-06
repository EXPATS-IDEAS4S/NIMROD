import numpy as np
from scipy.interpolate import griddata

def crop_radar_data(radar_data, radar_lat, radar_lon, lat_min, lat_max, lon_min, lon_max):
    """
    Crop radar data to a specified latitudinal and longitudinal range.

    Parameters
    ----------
    radar_data : ndarray
        2D array containing radar data.
    radar_lat : ndarray
        2D array of latitudes corresponding to `radar_data`.
    radar_lon : ndarray
        2D array of longitudes corresponding to `radar_data`.
    lat_min : float
        Minimum latitude boundary for cropping.
    lat_max : float
        Maximum latitude boundary for cropping.
    lon_min : float
        Minimum longitude boundary for cropping.
    lon_max : float
        Maximum longitude boundary for cropping.

    Returns
    -------
    tuple of ndarray
        A tuple containing the cropped radar data, latitude array, and longitude array.

    Notes
    -----
    This function assumes `radar_lat` and `radar_lon` are 2D arrays of the same shape as `radar_data`.
    """

    # Find the rows and columns that are entirely within the lat/lon boundaries
    valid_rows = np.all((radar_lat >= lat_min) & (radar_lat <= lat_max), axis=1)
    valid_cols = np.all((radar_lon >= lon_min) & (radar_lon <= lon_max), axis=0)

    # Use these to crop the radar data
    cropped_radar_data = radar_data[valid_rows, :][:, valid_cols]

    # Crop the latitude and longitude arrays
    cropped_radar_lat = radar_lat[valid_rows, :][:, valid_cols]
    cropped_radar_lon = radar_lon[valid_rows, :][:, valid_cols]

    return cropped_radar_data, cropped_radar_lat, cropped_radar_lon


def regrid_data(radar_lat, radar_lon, radar_data, msg_lat_grid, msg_lon_grid, method='nearest'):
    """
    Regrid radar data to a Meteosat Second Generation (MSG) grid.

    Parameters
    ----------
    radar_lat : ndarray
        1D or 2D array of latitude coordinates for the radar data.
    radar_lon : ndarray
        1D or 2D array of longitude coordinates for the radar data.
    radar_data : ndarray
        1D or 2D array of radar data values.
    msg_lat_grid : ndarray
        2D array of MSG latitude grid points.
    msg_lon_grid : ndarray
        2D array of MSG longitude grid points.

    Returns
    -------
    ndarray
        2D array of radar data regridded to the MSG grid.

    Notes
    -----
    This function uses nearest-neighbor interpolation for regridding as default method.
    """
    # Source points (radar data) - already flattened
    radar_points = np.array([radar_lat, radar_lon]).T

    # Create a 2D mesh grid for the MSG data
    msg_points = np.array([msg_lat_grid.flatten(), msg_lon_grid.flatten()]).T

    # Perform the regridding
    regridded_data = griddata(radar_points, radar_data, msg_points, method=method)

    # Reshape the regridded data back to 2D (if necessary)
    regridded_data_reshaped = regridded_data.reshape(msg_lat_grid.shape)

    return regridded_data_reshaped



def check_grid(lat,lon,data,data_type):
    """
    Print information about a data grid, including shape and value statistics.

    Parameters
    ----------
    lat : ndarray
        Array of latitude coordinates.
    lon : ndarray
        Array of longitude coordinates.
    data : ndarray
        Data array corresponding to the latitude and longitude coordinates.
    data_type : str
        Descriptive name of the data being checked (e.g., "radar", "satellite").

    Returns
    -------
    None

    Notes
    -----
    This function prints the shape of the latitude, longitude, and data arrays,
    along with the count of NaN values and values above zero within the data array.
    """
    print(data_type+' lat',np.shape(lat), lat)
    print(data_type+' lon',np.shape(lon), lon)
    if data:
        print(data_type+' values',np.shape(data),data)
        # Count NaN values
        nan_count = np.count_nonzero(np.isnan(data))
        # Count values above zero
        above_zero_count = np.count_nonzero(data > 0)
        print(f"Number of NaN values: {nan_count}")
        print(f"Number of values above zero: {above_zero_count}")