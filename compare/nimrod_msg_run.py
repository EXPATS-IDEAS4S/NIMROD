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
import pandas as pd

from config_compare import msg_folder, msg_filepattern, rain_path, rain_filepattern, path_outputs, cma_path, cma_filepattern
from config_compare import channels, vis_channels, ir_channels, lonmin,lonmax,latmin,latmax
from config_compare import rain_class_thresholds, rain_class_names, path_orography, elevation_class_thresholds, elevation_class_names

from comparison_function import get_max_min, select_daytime_files_from_hour, mask_nighttime_values, filter_rain_rate, filter_by_cloud_mask, filter_elevation, mask_nighttime_single_step, find_max_ets_threshold
from comparison_function import generate_cloud_classes, calc_distribution_stats_by_rain_classes, save_ets_thresholds_channels_trends, select_max_ets_thresholds, save_metrics_thresholds_channels, save_spatial_metrics_to_dataset

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.quality_check_functions import plot_msg_channels_and_rain_rate, create_gif_from_folder, plot_distributions, plot_distr_rain, plot_channel_daily_trends, plot_single_temp_trend, plot_channel_trends
from figures.plot_comparison_function import plot_rain_rate_vs_msg_channels, plot_distributions_by_rain_classes, plot_ets_trend_rain_threshold, plot_spatial_percentile, plot_spatial_metrics, plot_ets_trend_elev_class

plot_ets_trend_rain_threshold(path_outputs,vis_channels,'whole_distribution')

"""

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
#fnames_msg_day = select_daytime_files_from_hour(fnames_msg,'04','17')
# Check if the hour is in daytime (TODO check if the daytime range is correct using SZA)
#'04' <= hour <= '17': #Local time 6 am (included) to 20 (excluded) pm?
#print(fnames_msg_day)    

# Open the NetCDF files as a single dataset, concatenating along 'end_time'
ds_msg = xr.open_mfdataset(fnames_msg, combine='nested', concat_dim='end_time', parallel=True)
#ds_msg_day = xr.open_mfdataset(fnames_msg_day, combine='nested', concat_dim='end_time', parallel=True)
print('\nDataset MSG\n',ds_msg)
#print('\nDataset MSG dattime\n:',ds_msg_day)
ds_msg = ds_msg.rename({'end_time':'time'})
#ds_msg_day = ds_msg_day.rename({'end_time':'time'})

#TODO apply day mask using SZA
ds_msg_daytime_mask = mask_nighttime_values(ds_msg,'time', vis_channels,'04','17')
print('\nDataset MSG masked\n',ds_msg_daytime_mask)

# Open the NetCDF files as a single dataset, concatenating along 'end_time'
ds_rain = xr.open_mfdataset(fnames_rain, combine='nested', concat_dim='time', parallel=True)
print('\nDataset rain\n',ds_rain)

# Open CMA Dataset concateneting along time dimension
ds_cma = xr.open_mfdataset(fnames_cma, combine='nested', concat_dim='time', parallel=True)
print('\nDataset CMA\n',ds_cma)

#Open Orography Dataset
ds_oro = xr.open_dataset(path_orography)
print('\n Dataset Orography\n', ds_oro)

# Define elevation classes
elev_min, elev_max = get_max_min(ds_oro,'orography')
print(elev_min,elev_max)
print(elevation_class_thresholds) 
# Correctly constructing the list of boundaries
boundaries = [elev_min] + elevation_class_thresholds + [elev_max]
el_classes = generate_cloud_classes(boundaries)
print(el_classes, elevation_class_names)

# Define Rain classes
rain_min, rain_max = get_max_min(ds_rain,'rain_rate')
print(rain_min,rain_max)
print(rain_class_thresholds) #[0.1,2.5,10]
# Correctly constructing the list of boundaries
boundaries = [rain_min] + rain_class_thresholds + [rain_max]
classes = generate_cloud_classes(boundaries)

#get max min values in each channel
min_values = []
max_values = []
for ch in channels:
    min, max = get_max_min(ds_msg,ch)
    min_values.append(min)
    max_values.append(max)

#loop trough the files to plot single time maps
for t in range(n_times):
    rain_ds = xr.open_dataset(fnames_rain[t])
    msg_ds = xr.open_dataset(fnames_msg[t])
    msg_ds = mask_nighttime_single_step(msg_ds,'end_time',vis_channels,'04','17') 
    for rain_th in rain_class_thresholds:
        print('\nrain th\n', rain_th)
        plot_msg_channels_and_rain_rate(rain_ds, msg_ds, ds_oro, path_outputs+"ets_results_whole_distribution.csv", path_outputs+'maps_thresholds/rain_'+str(rain_th)+'/',[lonmin, lonmax, latmin, latmax], vis_channels, ir_channels, min_values, max_values, rain_th, True)
    #exit()

#create gif
#create_gif_from_folder(path_outputs+'maps/',path_outputs + 'Germany_2021_floods.gif')


#for rain_th in rain_class_thresholds: #[0.1,2.5,10]
#    plot_ets_trend_elev_class(path_outputs,channels,vis_channels,rain_th,elevation_class_names)

#plot distribution
#plotting_functions.plot_distributions(ds_rain,ds_msg,ds_msg_day,path_outputs, vis_channels, ir_channels)
#plotting_functions.plot_distr_rain(ds_rain,path_outputs,min_value = 1e-3, max_value = 1, bin_width = 0.1)

#plot temporal trend with daily average to select the daytime for VIS channels
#TODO do the threshold using solar zenit angle: (SZA<70° daytimes as in Claudia's Thesis)
#plotting_functions.plot_channel_daily_trends(vis_channels, ds_msg, path_outputs,'Reflectances (%)')

#plotting temporal trend 
#plotting_functions.plot_channel_trends(channels,ds_msg,ds_rain,'202107120015','202107160000', path_outputs, vis_channels, ir_channels)
#plotting_functions.plot_single_temp_trend(ds_rain,ds_msg,ds_msg_day,path_outputs,vis_channels,ir_channels,'202107120015','202107160000')


#plot spatial percentiles
plot_spatial_percentile(ds_msg_daytime_mask,ds_rain,ds_oro,[10,90],vis_channels,ir_channels,[lonmin, lonmax, latmin, latmax],path_outputs)
plot_spatial_percentile(ds_msg_daytime_mask,ds_rain,ds_oro,[5,95],vis_channels,ir_channels,[lonmin, lonmax, latmin, latmax],path_outputs)
plot_spatial_percentile(ds_msg_daytime_mask,ds_rain,ds_oro,[1,99],vis_channels,ir_channels,[lonmin, lonmax, latmin, latmax],path_outputs)

cloudy_msg_ds = filter_by_cloud_mask(ds_msg_daytime_mask, ds_cma)
print('\nDataset MSG filered\n:',cloudy_msg_ds)

#Align time coordinates of MSG and Rain dataset --> both should indicate the start time of the 15 minutes interval
times = pd.to_datetime(cloudy_msg_ds['time'].values)

# Round down to the nearest 15 minutes. 
# This can be done by subtracting the minute modulo 15 and resetting seconds and microseconds to 0.
rounded_times = times - pd.to_timedelta(times.minute % 15, unit='m') - pd.to_timedelta(times.second, unit='s') - pd.to_timedelta(times.microsecond, unit='us')

# Now, assign these rounded times back to your dataset's coordinate.
cloudy_msg_ds['time'] = ('time', rounded_times)

#check time series
#print('time in MSG. ', cloudy_msg_ds['time'].values)
#print('time in Rain: ', ds_rain['time'].values)


#plot scatter plot between rain and channels ( do not include dry pixels)



# bin_widths_rain = [0.1,0.5,2]
# bin_widths_channels = [5,5,5,5,2,2,5,2,5,5,2]
# for i,rain_cl in enumerate(classes):
#     if i>0:
#         print('rain class', rain_cl, rain_class_names[i])
#         filt_rain_ds = filter_rain_rate(ds_rain,rain_cl[0],rain_cl[1])
#         print('\nDataset Rain filtered\n',filt_rain_ds)
#         plot_rain_rate_vs_msg_channels(filt_rain_ds,cloudy_msg_ds,bin_widths_channels,bin_widths_rain[i-1],vis_channels,ir_channels, path_outputs, rain_class_names[i])


#plot the distributon of the channels for each rain class

bin_widths = [1,2,2,2,2,1,1,2,1,2,2,2]
#plot_distributions_by_rain_classes(ds_rain,cloudy_msg_ds,path_outputs,vis_channels,ir_channels,classes,rain_class_names,bin_widths)
#plot_distributions_by_elevation_classes(ds_rain,cloudy_msg_ds,ds_oro,path_outputs,vis_channels,ir_channels,el_classes,elevation_class_names,bin_widths)

classes = classes + [(0.1,rain_max)] + [(2.5,rain_max)]
rain_class_names = rain_class_names + ['rain'] + ['moderate+heavy']
print('Rain classes:','\n', classes,'\n', rain_class_names)

#calc_distribution_stats_by_rain_classes(ds_rain,cloudy_msg_ds,path_outputs,channels,classes,rain_class_names)

# for i, elev_class_name in enumerate(elevation_class_names):
#     if elev_class_name!='Plains':
#         print(elev_class_name,el_classes[i])
#         filt_rain_ds, filt_cloudy_msg_ds = filter_elevation(cloudy_msg_ds,ds_rain,ds_oro,el_classes[i])

#         save_ets_thresholds_channels_trends(filt_cloudy_msg_ds,filt_rain_ds,vis_channels,ir_channels,rain_class_thresholds,elev_class_name,20,path_outputs)

#         plot_ets_trend_rain_threshold(path_outputs,vis_channels,elev_class_name)

#         #find threshold that maximaze ets
#         #select_max_ets_thresholds(path_outputs+"ets_results_whole_distribution.csv", path_outputs+'channel_thresholds_max_ets.csv')

#         #find the rest of the metrics 
#         save_metrics_thresholds_channels(filt_cloudy_msg_ds,filt_rain_ds,vis_channels,ir_channels,rain_class_thresholds,elev_class_name,path_outputs)



metrics = ['pod', 'far', 'bias', 'csi', 'ets']
for rain_threshold in rain_class_thresholds:
    if rain_threshold==10:
        print('rain th ', rain_threshold)
        save_spatial_metrics_to_dataset(cloudy_msg_ds,ds_rain,rain_threshold,vis_channels,ir_channels, path_outputs)

        ds_metrics = xr.open_dataset(path_outputs+'metrics_results_pixelwise_rain-th-'+str(rain_threshold)+'.nc')
        print(ds_metrics)

        for metric in metrics:
            print('metric: ',metric)
            plot_spatial_metrics(ds_metrics,ds_rain,metric,rain_threshold,vis_channels, ir_channels,[lonmin, lonmax, latmin, latmax],path_outputs )
"""