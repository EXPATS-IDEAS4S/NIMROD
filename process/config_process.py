"""
List with all the parameters, path, flag etc for the processing part
"""

# File paths for lat lon grids
#msg_file = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/MSG/HRSEVIRI_20220712_20210715_Flood_domain_DataTailor_nat/HRSEVIRI_20210712_20210715_Processed/MSG4-SEVI-MSG15-0100-NA-20210714121243.nc'
msg_folder = "/data/sat/msg/netcdf/noparallax/2023/04/*/"
msg_filepattern = "MSG*-SEVI-MSG*-0100-NA-*.nc"
reg_lats = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/reg_lats.npy"
reg_lons = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/reg_lons.npy"

#Path to radar files
#radar_folder = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/rain_products/nimrod/nc_files/'  
radar_folder = '/data/sat/msg/radar/nimrod/netcdf/2023/04/'  
radar_filepattern = 'nimrod_rain_data_eu_*.nc'

#Path to save the regridded radar file
#output_folder = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/rain_products/nimrod/nc_files_reg_grid/'
output_folder = "/data/sat/msg/radar/nimrod/netcdf/2023/04/"
fig_folder = "/work/dcorradi/case_studies_expats/Germany_Flood_2021/Fig/nimrod_maps/"

#number of columns and rows for radar data
ncols_radar = 620
nrows_radar = 700

#Define Domain
lon_min, lon_max, lat_min, lat_max = 5. , 16. , 42. , 51.5 

#Flag to save in netcdf
save = True

#Flag to choose wich grid (True: regular grid, False: MSG native grid)
regular_grid = True

