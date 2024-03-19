"""
List with all the parameters, path, flag etc for the comperison part
"""

# File paths for msg
#msg_file = '/home/daniele/Documenti/PhD_Cologne/Case_Studies/Germany_Flood_2021/MSG/HRSEVIRI_20220712_20210715_Flood_domain_DataTailor_nat/HRSEVIRI_20210712_20210715_Processed/MSG4-SEVI-MSG15-0100-NA-20210714121243.nc'
msg_folder = "/work/case_studies_expats/Germany_Flood_2021/data/MSG/MSGNATIVE/Parallax_Corrected/regrid/"
msg_filepattern = "MSG4-SEVI-MSG15-0100-NA-*.nc"

#path to data
rain_path = '/work/case_studies_expats/Germany_Flood_2021/data/rain_products/nimrod/nc_files_reg_grid/'
rain_filepattern = "regridded_nimrod_rain_data_eu_*.nc"

#path to cloud mask
cma_path = "/work/case_studies_expats/Germany_Flood_2021/data/cloud_products/CMA_NWCSAF/Processed/"
cma_filepattern = "S_NWC_CMA_MSG4_FLOOD-GER-2021-VISIR_*.nc"

#path to fig
path_outputs = '/work/case_studies_expats/Germany_Flood_2021/Fig/nimrod_msg_analysis/'

#channel names
channels = ['VIS006', 'VIS008','IR_016', 'IR_039','WV_062', 'WV_073', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134']
vis_channels = ['VIS006', 'VIS008', 'IR_016'] #channels given in reflectances
ir_channels = ['IR_039', 'WV_062', 'WV_073', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134'] #channles fiven in BT

#definte the domain of interest
lonmin, latmin, lonmax, latmax= 5, 48, 9, 52

#Flag to choose wich grid (True: regular grid, False: MSG native grid)
regular_grid = True