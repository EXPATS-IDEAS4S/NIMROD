"""
plotting function used
to compare MSG and NIMROD data
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

