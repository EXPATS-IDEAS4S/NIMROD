"""
Compare the channel values from MSG data and the rain rate from NIMROD

author: Daniele Corradini
last edit: 09.01.24
"""

import numpy as np
import os
import xarray as xr
from glob import glob
from netCDF4 import Dataset

import quality_check_functions

#path to data
msg_path = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/MSG/HRSEVIRI_20220712_20210715_Flood_domain_DataTailor_nat/HRSEVIRI_20210712_20210715_Processed/'
rain_data_path = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/rain_products/nimrod/nc_files_RegridToMSG/'
path_outputs = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/Fig/'

#channel names
channels = ['VIS006', 'VIS008','IR_016', 'IR_039','WV_062', 'WV_073', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134']
vis_channels = ['VIS006', 'VIS008', 'IR_016'] #channels given in reflectances
ir_channels = ['IR_039', 'WV_062', 'WV_073', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134'] #channles fiven in BT

#definte the domain of interest
lonmin, latmin, lonmax, latmax= 5, 48, 9, 52

#plot maps
#3 rows and 4 columns: 
#1st row (rain rate, VIS (0.6, 0.8) and NIR (1.6) channels)
#2nd and 3rd rows (the remaining IR channels: 3.9, 6.2, 7.3, 8.7, \\ 9.7, 10.8, 12.0, 13.4

# List all rain product files in the folder and sort them alphabetically
radar_files = sorted([f for f in os.listdir(rain_data_path) if f.endswith('.nc')])

#list of all MSG data
msg_files = sorted([f for f in os.listdir(msg_path) if f.endswith('.nc')])

#check length
#print(len(radar_files),len(msg_files))

#merge the nc files in time to create a sigle file

#filenames for msg data
msg_f = "MSG4-SEVI-MSG15-0100-NA-*.nc" #"MSG4-SEVI-MSG15-0100-NA-20210712001243.nc"
fnames_msg = glob(msg_path+msg_f)
fnames_msg_day = []

for file in fnames_msg:
    # Extracting the timestamp part from the filename
    time = file.split('/')[-1].split('-')[-1].split('.')[0]

    # Extracting the hour part from the timestamp
    hour = time[8:10]

    # Check if the hour is between '05' and '18' (TODO check if the daytime range is correct)
    if '04' <= hour <= '17': #Local time 6 am (included) to 20 (excluded) pm?
        fnames_msg_day.append(file)
#print(fnames_msg_day)    

# Open the NetCDF files as a single dataset, concatenating along 'end_time'
ds_msg = xr.open_mfdataset(fnames_msg, combine='nested', concat_dim='end_time', parallel=True)
ds_msg_day = xr.open_mfdataset(fnames_msg_day, combine='nested', concat_dim='end_time', parallel=True)
print(ds_msg)
print(ds_msg_day)

#filenames for rain rate data
rain_files = "regridded_nimrod_rain_data_eu_*.nc"
fnames_rain = glob(rain_data_path+rain_files)

# Open the NetCDF files as a single dataset, concatenating along 'end_time'
ds_rain = xr.open_mfdataset(fnames_rain, combine='nested', concat_dim='time', parallel=True)
print(ds_rain)

#get max min values in each vis channel
min_vis = []
max_vis = []
for ch in vis_channels:
    min, max = quality_check_functions.get_max_min(ds_msg_day,ch)
    min_vis.append(min)
    max_vis.append(max)

#get max min values in each ir channel
min_ir = []
max_ir = []
for ch in ir_channels:
    min, max = quality_check_functions.get_max_min(ds_msg,ch)
    min_ir.append(min)
    max_ir.append(max)

min_values = min_vis+min_ir
max_values = max_vis+max_ir

#loop trough the files
for rain_file, msg_file in zip(radar_files,msg_files):
    #plot map
    quality_check_functions.plot_msg_channels_and_rain_rate(rain_data_path+rain_file,msg_path+msg_file,path_outputs,lonmin, lonmax, latmin, latmax, vis_channels, ir_channels, min_values, max_values)
    exit()
#create gif
#plotting_functions.create_gif_from_folder(path_outputs+'maps/',path_outputs + 'Germany_2021_floods_gif.gif')

#plot distribution
#plotting_functions.plot_distributions(ds_rain,ds_msg,ds_msg_day,path_outputs, vis_channels, ir_channels)
#plotting_functions.plot_distr_rain(ds_rain,path_outputs,min_value = 1e-3, max_value = 1, bin_width = 0.1)

#plot temporal trend with daily average to select the daytime for VIS channels
#TODO do the threshold using solar zenit angle: (SZA<70° daytimes as in Claudia's Thesis)
#plotting_functions.plot_channel_daily_trends(vis_channels, ds_msg, path_outputs,'Reflectances (%)')

#plotting temporal trend TODO change to plot in separate subplots
#plotting_functions.plot_channel_trends(channels,ds_msg,ds_rain,'202107120015','202107160000', path_outputs, vis_channels, ir_channels)
#plotting_functions.plot_single_temp_trend(ds_rain,ds_msg,ds_msg_day,path_outputs,vis_channels,ir_channels,'202107120015','202107160000')

#plot avarage and max spatial maps? it is relevant? only for rain rate?

#plot scatter plot between rain and channels (include dry pixels?)
#TODO perform parallax correction before comparison 
#TODO download cloud mask
