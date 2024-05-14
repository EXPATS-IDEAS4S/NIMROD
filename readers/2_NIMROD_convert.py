"""
Created on 18.12.23

This tool creates ascii rasters of Met Office 'NIMROD' rainfall radar
data. It can be used to convert an entire directory of NIMROD files. 


@author: Daniele Corradini 
@author: Adapted from  Declan Valters' (2014) and Charles Kilburn's script (2008)
"""
import struct
import array
import numpy as np
import matplotlib as mpl
from glob import glob
import os
    

# check to see if gride cell resolution is same in both dimensions
def check_horiz_vert_resolution(row_interval_res, col_interval_res):
    if row_interval_res != col_interval_res:
        print("Warning: the row resolution ", row_interval_res, "is not the \
        same resolution as the column resolution",  col_interval_res, " in this \
        dataset. Row resolution has been used to set the ASCII 'cellsize'. Check your data.")
        
#write the array and header to an ascii file
def write_NIMROD_toASCII(asciiname, radararray, header):
    np.savetxt(asciiname, radararray, header=header, comments='', fmt='%1.1f')

#Horizontal grid types
grid_types = {'0':"British National Grid", \
'1':"Lat/Long", \
'2':"Space View", \
'3':"Polar Stereo", \
'4':"UTM32", \
'5':"Rotated Lat/Lon", \
'6':"Other"}

def ingest_NIMROD_file(full_filepath):
    file_id = open(full_filepath,"rb")
    record_length, = struct.unpack(">l", file_id.read(4))
    if record_length != 512: 
        raise( "Unexpected record length", record_length)
    
    # Set up the arrays, with the correct type
    gen_ints = array.array("h")
    gen_reals = array.array("f")
    spec_reals = array.array("f")
    characters = array.array("b")
    spec_ints = array.array("h")
    
    # read in the data from the open file
    # Note array.read deprecated since Py3, use fromfile() instead
    gen_ints.fromfile(file_id, 31) 
    gen_reals.fromfile(file_id, 28)
    spec_reals.fromfile(file_id, 45)
    characters.fromfile(file_id, 56)
    spec_ints.fromfile(file_id, 51)
    
    gen_ints.byteswap()
    gen_reals.byteswap()
    spec_reals.byteswap()
    spec_ints.byteswap()
    
   
    origin = gen_ints[23]
    print(origin) #0=top LH corner, 1=bottom LH corner, 2=top RH corner, 3=bottom RH corner
   
    #print(proj)
    #missing_val = gen_ints[24]
    #print(missing_val)
    
    grid_typeID = gen_ints[14]
    #print(grid_typeID) #3: Polar Stero
    grid_type = grid_types[str(grid_typeID)] #(Dictionaries need a string to look up)
    print(grid_type)
    
    record_length, = struct.unpack(">l", file_id.read(4))
    if record_length != 512: 
        raise( "Unexpected record length", record_length)
        
    chars = characters.tobytes().decode('utf-8') #oldcode: chars = characters.tostring()
        
    #Read the Data
    array_size = gen_ints[15] * gen_ints[16]
    nrows = gen_ints[15]
    ncols = gen_ints[16]
    #xllcornerNG = spec_reals[7]
    #yllcornerNG = spec_reals[6]  # 'NG' = British National Grid, or 'OSGB36' to be precise
    cellsize_y = gen_reals[3]
    cellsize_x = gen_reals[5]
    check_horiz_vert_resolution(cellsize_x,cellsize_y)
    print('cellsize (x,y):',cellsize_x,cellsize_y)
    nodata_value = gen_reals[6]

    #data offset value
    data_offset = gen_reals[8]
    print('data offset or downward longitude',data_offset) #it should be zero if says nodata value

    #standard latitude
    ref_lat = gen_reals[11]
    print('std latitude',ref_lat)

    
    #get origin of the image   
    ytlcorner = gen_reals[2] # get the top left corner y co-ord
    xtlcorner = gen_reals[4] # get the top left corner x co-ord
    print(ytlcorner, xtlcorner) #in degrees
    
    #Note if you use the data in spec_reals, the co-ordnates are 500m apart...probably not big enough to worry about    
    record_length, = struct.unpack(">l", file_id.read(4))
    if record_length != array_size * 2: 
        raise( "Unexpected record length", record_length)
    
    data = array.array("h")
    try:
        data.fromfile(file_id, array_size) # read() is deprecated. And it only reads linearly through a file
        record_length, = struct.unpack(">l", file_id.read(4))
        if record_length != array_size * 2: raise( "Unexpected record length", record_length)
        data.byteswap()
        #print "First 100 values are", data[:100]
        #print "Last 100 values are", data[-100:]
    except:
        print( "Read failed")
        
    radararray = np.reshape(data, (nrows,ncols)) 
    radararray = radararray / 32.0 # This is due to a strange NIMROD convention where everything is *32
    
    # Make the header
    header = 'NCols ' + str(ncols) + '\n' + 'NRows ' + str(nrows) + '\n' + 'xllcorner ' + str(xtlcorner) + '\n' + 'yllcorner ' + str(ytlcorner) + '\n' + 'cellsize ' + str(cellsize_x) + '\n' + 'NODATA_value ' + str(nodata_value) + '\n' + 'std latitude ' + str(ref_lat)
    
    file_id.close()
    return radararray, header
    
def convert_multiple_files(path_to_dir, basename, save_dir):
    # The base name will should be appended to a wildcard to narrow down the search.
    for filename in glob(path_to_dir + basename):
        print( filename)
        thisradararray, thisheader = ingest_NIMROD_file(filename)
        asciiname = save_dir + filename.split('/')[-1].split('.')[0] + '.asc'
        write_NIMROD_toASCII(asciiname, thisradararray, thisheader)
    
def convert_single_file(path, fname, ext, asciiname):
    full_fname = path + fname + ext
    radararray, header = ingest_NIMROD_file(full_fname)
    write_NIMROD_toASCII(asciiname, radararray, header)
    print("NIMROD file converted to ASCII: ", asciiname)
    
def show_radar_image(full_filepath):
    #full_fname = path + fname + '.' + ext
    radararray, header = ingest_NIMROD_file(full_filepath)
    mpl.pyplot.imshow(radararray)
    
########
# MAIN #
########  

#directories paths
fpath = '/data/sat/msg/radar/nimrod/dat/2023/04/'
save_dir = '/data/sat/msg/radar/nimrod/asc/'

# Ensure the output folder exists, create if not
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# for multiple files. You do not need to specify a basename if you want to conver ALL files in dir.
basename = "*composite.dat"
convert_multiple_files(fpath, basename, save_dir)
















