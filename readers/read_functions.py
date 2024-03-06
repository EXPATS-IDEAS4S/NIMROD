from netCDF4 import Dataset, num2date


def read_msg_lat_lon(msg_file):
    """
    Read latitude and longitude grid data from a Meteosat Second Generation (MSG) NetCDF file.

    Parameters
    ----------
    msg_file : str
        The file path to the MSG NetCDF file containing latitude and longitude grid data.

    Returns
    -------
    lat : ndarray
        2D array of latitude values from the MSG file.
    lon : ndarray
        2D array of longitude values from the MSG file.

    Notes
    -----
    This function assumes that the latitude and longitude data are stored under the variable
    names 'lat grid' and 'lon grid' within the NetCDF file.
    """
    with Dataset(msg_file, 'r') as nc:
        #print(nc.variables)
        lat = nc['lat_grid'][:] 
        lon = nc['lon_grid'][:] 
    return lat, lon


def read_radar_data_with_lat_lon(radar_file):
    """
    Read radar data along with corresponding latitude, longitude, and time from a NetCDF file.

    Parameters
    ----------
    radar_file : str
        The file path to the NetCDF file containing radar data and associated lat/lon grids.

    Returns
    -------
    time : datetime
        The time corresponding to the radar data, extracted from the 'time' variable in the file.
    latitudes : ndarray
        1D or 2D array of latitude values associated with the radar data.
    longitudes : ndarray
        1D or 2D array of longitude values associated with the radar data.
    rain_rate : ndarray
        2D array of radar-derived rain rate values for the specified time step.

    Notes
    -----
    This function reads a single time step of radar data from the provided NetCDF file, which
    is expected to contain 'latitude', 'longitude', and 'rain_rate' variables, along with a 'time'
    variable describing the time of the radar observation. It is assumed there's only one time
    step per file, so the first (and only) time step is returned.
    """
    with Dataset(radar_file, 'r') as nc:
        time_var = nc.variables['time']
        time = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)

        # Read latitude and longitude data
        latitudes = nc.variables['latitude'][:]
        longitudes = nc.variables['longitude'][:]

        rain_rate = nc.variables['rain_rate'][:]
        # Since there's only one time step per file, we take the first (and only) element
        rain_rate = rain_rate[0, :]

    return time[0], latitudes, longitudes, rain_rate