"""
Compare the channel values from MSG data and the rain rate from NIMROD

@author: Daniele Corradini
"""

import numpy as np
import os
import xarray as xr
from glob import glob
from netCDF4 import Dataset
import sys
import datetime

from config_compare import msg_folder, msg_filepattern, rain_path, rain_filepattern, path_outputs, cma_path, cma_filepattern
from config_compare import channels, vis_channels, ir_channels, lonmin,lonmax,latmin,latmax

from comparison_function import get_max_min, select_daytime_files_from_hour, mask_nighttime_values, filter_rain_rate, filter_by_cloud_mask

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.quality_check_functions import plot_msg_channels_and_rain_rate, create_gif_from_folder, plot_distributions, plot_distr_rain, plot_channel_daily_trends, plot_single_temp_trend, plot_channel_trends
from figures.plot_comparison_function import plot_rain_rate_vs_msg_channels

# List all rain product files in the folder and sort them alphabetically
fnames_rain = sorted(glob(rain_path+rain_filepattern))

#list of all MSG data
fnames_msg = sorted(glob(msg_folder+msg_filepattern))

# List of all CMA data
fnames_cma = sorted(glob(cma_path+cma_filepattern))

#check length
if len(fnames_rain)==len(fnames_msg):
    print('radar files and msg files have the same number of time stamps')
    n_times = len(fnames_rain)
else:
    print('something is wrong with the lenght of the filenames')
    exit()

# Filenames for msg data only for daytime
fnames_msg_day = select_daytime_files_from_hour(fnames_msg,'04','17')
# Check if the hour is in daytime (TODO check if the daytime range is correct using SZA)
#'04' <= hour <= '17': #Local time 6 am (included) to 20 (excluded) pm?
#print(fnames_msg_day)    

# Open the NetCDF files as a single dataset, concatenating along 'end_time'
ds_msg = xr.open_mfdataset(fnames_msg, combine='nested', concat_dim='end_time', parallel=True)
ds_msg_day = xr.open_mfdataset(fnames_msg_day, combine='nested', concat_dim='end_time', parallel=True)
print('\nDataset MSG\n:',ds_msg)
print('\nDataset MSG dattime\n:',ds_msg_day)
ds_msg = ds_msg.rename({'end_time':'time'})
ds_msg_day = ds_msg_day.rename({'end_time':'time'})

ds_msg_daytime_mask = mask_nighttime_values(ds_msg,'time', vis_channels,'04','17')
print('\nDataset MSG masked\n:',ds_msg_daytime_mask)

# Open the NetCDF files as a single dataset, concatenating along 'end_time'
ds_rain = xr.open_mfdataset(fnames_rain, combine='nested', concat_dim='time', parallel=True)
print('\nDataset rain\n:',ds_rain)

# Open CMA Dataset concateneting along time dimension
ds_cma = xr.open_mfdataset(fnames_cma, combine='nested', concat_dim='time', parallel=True)
print('\nDataset CMA\n:',ds_cma)

# #get max min values in each channel
# min_values = []
# max_values = []
# for ch in channels:
#     min, max = get_max_min(ds_msg,ch)
#     min_values.append(min)
#     max_values.append(max)

#loop trough the files to plot single time maps
# for t in range(n_times):
#     rain_ds = xr.open_dataset(fnames_rain[t])
#     msg_ds = xr.open_dataset(fnames_msg[t])
#     #msg_ds = mask_nighttime_values(msg_ds,'end_time','04','17') #not for single time step dataset!
#     plot_msg_channels_and_rain_rate(rain_ds, msg_ds, path_outputs,[lonmin, lonmax, latmin, latmax], vis_channels, ir_channels, min_values, max_values)
#     #exit()

#create gif
#create_gif_from_folder(path_outputs+'maps/',path_outputs + 'Germany_2021_floods.gif')

#plot distribution
#plotting_functions.plot_distributions(ds_rain,ds_msg,ds_msg_day,path_outputs, vis_channels, ir_channels)
#plotting_functions.plot_distr_rain(ds_rain,path_outputs,min_value = 1e-3, max_value = 1, bin_width = 0.1)

#plot temporal trend with daily average to select the daytime for VIS channels
#TODO do the threshold using solar zenit angle: (SZA<70° daytimes as in Claudia's Thesis)
#plotting_functions.plot_channel_daily_trends(vis_channels, ds_msg, path_outputs,'Reflectances (%)')

#plotting temporal trend 
#plotting_functions.plot_channel_trends(channels,ds_msg,ds_rain,'202107120015','202107160000', path_outputs, vis_channels, ir_channels)
#plotting_functions.plot_single_temp_trend(ds_rain,ds_msg,ds_msg_day,path_outputs,vis_channels,ir_channels,'202107120015','202107160000')

#plot avarage and max spatial maps? it is relevant? only for rain rate?

#plot scatter plot between rain and channels ( do not include dry pixels)

filt_rain_ds = filter_rain_rate(ds_rain)
print('\nDataset Rain filtered\n:',filt_rain_ds)

cloudy_msg_ds = filter_by_cloud_mask(ds_msg_daytime_mask, ds_cma)
print('\nDataset MSG filered\n:',cloudy_msg_ds)

plot_rain_rate_vs_msg_channels(filt_rain_ds,cloudy_msg_ds,5,50,vis_channels,ir_channels, path_outputs)