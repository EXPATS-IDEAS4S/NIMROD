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
from config_compare import channels, vis_channels, ir_channels, lonmin,lonmax,latmin,latmax, channel_comb
from config_compare import rain_class_thresholds, rain_class_names, path_orography, elevation_class_thresholds, elevation_class_names

from comparison_function import get_max_min, select_daytime_files_from_hour, mask_nighttime_values, filter_rain_rate, filter_by_cloud_mask, filter_elevation, mask_nighttime_single_step, find_max_ets_threshold
from comparison_function import generate_cloud_classes, calc_distribution_stats_by_rain_classes, save_ets_thresholds_channels_trends, select_max_ets_thresholds, save_metrics_thresholds_channels, save_spatial_metrics_to_dataset

sys.path.append('/home/dcorradi/Documents/Codes/NIMROD/')
from figures.quality_check_functions import plot_msg_channels_and_rain_rate, create_gif_from_folder, plot_distributions, plot_distr_rain, plot_channel_daily_trends, plot_single_temp_trend, plot_channel_trends
from figures.plot_comparison_function import plot_rain_rate_vs_msg_channels, plot_distributions_by_rain_classes, plot_ets_trend_rain_threshold, plot_spatial_percentile, plot_spatial_metrics, plot_ets_trend_elev_class
from figures.msg_comb_plot_func import plot_msg_channels_comb, plot_distributions_comb, plot_channel_daily_trends_comb, plot_spatial_percentile_comb, plot_distributions_by_rain_classes_comb, filter_dataset_by_variable, save_ets_thresholds_channels_trends_comb, plot_ets_trend_rain_threshold_comb, save_metrics_thresholds_channels_comb, plot_conf_matrix_maps, save_conf_matrix_values, merge_csv_files, calculate_metrics, calculate_and_plot_metrics_over_time

#Domain
case_domain = [lonmin, lonmax, latmin, latmax]

# List all rain product files in the folder and sort them alphabetically
fnames_rain = sorted(glob(rain_path+rain_filepattern))

fnames_msg_comb= sorted(glob(msg_folder+'combination/'+msg_filepattern))

fnames_msg= sorted(glob(msg_folder+msg_filepattern))

#time_dim_name = 'time'
#channels = channel_comb

# List of all CMA data
fnames_cma = sorted(glob(cma_path+cma_filepattern))

#check length
if len(fnames_rain)==len(fnames_msg):
    print('radar files and msg files have the same number of time stamps')
    n_times = len(fnames_rain)
else:
    print('something is wrong with the lenght of the filenames')
    exit()
 

# # Open the NetCDF files as a single dataset, concatenating along 'end_time'
# ds_msg_comb = xr.open_mfdataset(fnames_msg_comb, combine='nested', concat_dim='time', parallel=True)
# ds_msg = xr.open_mfdataset(fnames_msg, combine='nested', concat_dim='end_time', parallel=True)
# ds_msg = ds_msg.rename({'end_time':'time'})
# ds_msg_merge_all = xr.merge([ds_msg,ds_msg_comb])
# #ds_msg_day = xr.open_mfdataset(fnames_msg_day, combine='nested', concat_dim='end_time', parallel=True)
# print('\nDataset MSG\n',ds_msg_merge_all.var)
# #print('\nDataset MSG dattime\n:',ds_msg_day)

#TODO apply day mask using SZA
#ds_msg_daytime_mask = mask_nighttime_values(ds_msg,'time', ['VIS006:IR_016'] ,'04','17') #COT should be already be masked

#print('\nDataset MSG masked\n',ds_msg_daytime_mask)

# # Open the NetCDF files as a single dataset, concatenating along 'end_time'
# ds_rain = xr.open_mfdataset(fnames_rain, combine='nested', concat_dim='time', parallel=True)
# print('\nDataset rain\n',ds_rain)

# # Open CMA Dataset concateneting along time dimension
# ds_cma = xr.open_mfdataset(fnames_cma, combine='nested', concat_dim='time', parallel=True)
# print('\nDataset CMA\n',ds_cma)

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

# # Define Rain classes
# rain_min, rain_max = get_max_min(ds_rain,'rain_rate')
# print(rain_min,rain_max)
# print(rain_class_thresholds) #[0.1,2.5,10]
# # Correctly constructing the list of boundaries
# boundaries = [rain_min] + rain_class_thresholds + [rain_max]
# classes = generate_cloud_classes(boundaries)


# #units and cmaps for comb msg
# cmaps = ['cool','cool','cool','cool','cool','cool','cool','cool']
# units = ['','','K','K','K','K','K','K']
# binwidths = [1,0.2,1,0.2,0.2,1,1,1]

channels_to_plot = ['COT','WV_062','IR_108','WV_062-IR_108']
cmaps_to_plot = ['viridis','cool','cool','cool']
units_to_plot = ['','K','K','K']
trends_to_plot = ['positive','negative','negative','positive']
min_values =[0,200,200,-60]
max_values=[150,300,300,0]
msg_thresholds = [15.7894736842105, 230.952929045025 , 252.283447265625 ,-20.7063831530119 ]

# #get max min values in each channel
# min_values = []
# max_values = []
# for ch in channels_to_plot:
#     min, max = get_max_min(ds_msg_merge_all,ch)
#     min_values.append(min)
#     max_values.append(max)
# print(min_values,max_values)


#loop trough the files to plot single time maps with delineated rainy areas
#for i, ch in enumerate(channels_to_plot):
# for t in range(n_times):
#     #print(ch)
#     rain_ds = xr.open_dataset(fnames_rain[t])
#     msg_ds = xr.open_dataset(fnames_msg[t])
#     msg_ds = msg_ds.rename({'end_time':'time'})
#     msg_comb_ds = xr.open_dataset(fnames_msg_comb[t])
#     msg_ds = mask_nighttime_single_step(msg_ds,'time',['VIS006', 'VIS008','IR_016'],'04','17') 
#     msg_comb_ds = mask_nighttime_single_step(msg_comb_ds,'time',['VIS006:IR_016'],'04','17')
#     combined_msg_ds = xr.merge([msg_ds, msg_comb_ds])
#     #plot_conf_matrix_spatial_maps(combined_msg_ds, rain_ds, ds_oro, 0.1, msg_thresholds[i], trends_to_plot[i], path_outputs+'nimrod_channel_comb/', case_domain, ch, cmaps_to_plot[i], units_to_plot[i], min_values[i], max_values[i])
#     #plot_conf_matrix_maps(combined_msg_ds, rain_ds, ds_oro, 0.1, msg_thresholds, trends_to_plot, path_outputs+'nimrod_channel_comb/', case_domain, channels_to_plot)
#     save_conf_matrix_values(combined_msg_ds, rain_ds,  ds_oro, 0.1, msg_thresholds, channels_to_plot, trends_to_plot, el_classes, elevation_class_names, path_outputs+'nimrod_channel_comb/csv_files/')
#     #exit()

#merge_csv_files(path_outputs+'nimrod_channel_comb/csv_files/', path_outputs+'nimrod_channel_comb/merged_file.csv')
#calculate_metrics(path_outputs+'nimrod_channel_comb/merged_file.csv', path_outputs+'nimrod_channel_comb/output_metrics.csv')
calculate_and_plot_metrics_over_time(path_outputs+'nimrod_channel_comb/merged_file.csv', path_outputs+'nimrod_channel_comb/metrics_temp_trend_plot.png')
#create gif
#create_gif_from_folder( path_outputs+'nimrod_channel_comb/conf_matrix_maps/',path_outputs+'nimrod_channel_comb/'+ 'conf_matrix_Germany_2021_floods.gif')

#for rain_th in rain_class_thresholds: #[0.1,2.5,10]
#    plot_ets_trend_elev_class(path_outputs,channels,vis_channels,rain_th,elevation_class_names)

#plot distribution
#plot_distributions_comb(ds_msg_daytime_mask, path_outputs+'nimrod_channel_comb/', channels, units, min_values, max_values, binwidths)

#plot temporal trend with daily average to select the daytime for VIS channels
#plot_channel_daily_trends_comb(['COT','VIS006:IR_016'], ds_msg, path_outputs+'nimrod_channel_comb/', '','vis-related')
#plot_channel_daily_trends_comb(['WV_062-IR_108','IR_087-IR_108','IR_108-IR_120','IR_039-IR_108','IR_039-WV_073','WV_073-IR_120'], ds_msg, path_outputs+'nimrod_channel_comb/', 'K','ir-related')



# #plot spatial percentiles
#percentiles = [90, 90, 90, 90, 10, 10, 10, 90]
# percentiles = [95, 95, 95, 95, 5, 5, 5, 95]
# plot_spatial_percentile_comb(ds_msg_daytime_mask, ds_oro, percentiles, channels, case_domain, path_outputs+'nimrod_channel_comb/', cmaps, units)

"""  
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
print('time in MSG. ', cloudy_msg_ds['time'].values)
print('time in Rain: ', ds_rain['time'].values)


#plot the distributon of the channels for each rain class
#plot_distributions_by_rain_classes_comb(ds_rain, cloudy_msg_ds,path_outputs+'nimrod_channel_comb/', channels, classes, rain_class_names, binwidths, units)


classes = classes + [(0.1,rain_max)] + [(2.5,rain_max)]
rain_class_names = rain_class_names + ['rain'] + ['moderate+heavy']
print('Rain classes:','\n', classes,'\n', rain_class_names)

#calc_distribution_stats_by_rain_classes(ds_rain,cloudy_msg_ds,path_outputs,channels,classes,rain_class_names)

class_names = rain_class_names 
classes_values = classes
channel_trends = ['positive','positive','positive','positive','negative','negative','negative','positive'] 

# for i, class_name in enumerate(class_names):
#     #if elev_class_name!='Plains':
#     print(class_name, classes_values[i])

#     save_ets_thresholds_channels_trends_comb(cloudy_msg_ds, ds_rain, channels, channel_trends, rain_class_thresholds, 'whole_distribution', 20, path_outputs+'nimrod_channel_comb/')
#     plot_ets_trend_rain_threshold_comb(path_outputs+'nimrod_channel_comb/',units, 'whole_distribution')

#     #find threshold that maximaze ets
#     select_max_ets_thresholds(path_outputs+'nimrod_channel_comb/'+"ets_results_whole_distribution.csv", path_outputs+'nimrod_channel_comb/'+'channel_thresholds_max_ets.csv')

#     #find the rest of the metrics 
#     save_metrics_thresholds_channels_comb(cloudy_msg_ds, ds_rain, channels,channel_trends, rain_class_thresholds, 'whole_distribution', path_outputs+'nimrod_channel_comb/')


# metrics = ['pod', 'far', 'bias', 'csi', 'ets']
# for rain_threshold in rain_class_thresholds:
#     #if rain_threshold==10:
#     print('rain th ', rain_threshold)
#     save_spatial_metrics_to_dataset(cloudy_msg_ds,ds_rain,rain_threshold,vis_channels,ir_channels, path_outputs)

#     ds_metrics = xr.open_dataset(path_outputs+'metrics_results_pixelwise_rain-th-'+str(rain_threshold)+'.nc')
#     print(ds_metrics)

#     for metric in metrics:
#         print('metric: ',metric)
#         plot_spatial_metrics(ds_metrics,ds_rain,metric,rain_threshold,vis_channels, ir_channels,[lonmin, lonmax, latmin, latmax],path_outputs )

"""
