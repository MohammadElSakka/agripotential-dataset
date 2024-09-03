from scripts.data import get_mask, get_meta , serialize_meta, get_altitude_data,get_data,get_categorical_mask
from scripts.hdf5 import create_hdf5
from scripts.WeatherData import temperature_array,rainfall_array
# Retrieve data 
sentinel_data = get_data()

#retrieve meta and serialized it 
meta = get_meta()
meta_serialized = serialize_meta(meta)

# Retrieve altitude
altitude_data = get_altitude_data(meta)

# Retrieve mask
mask = get_mask(meta)

categorical_mask = get_categorical_mask(meta)

# Create HDF5 file using the retrieved data
create_hdf5(sentinel_data,altitude_data,meta_serialized,mask,categorical_mask,temperature_array,rainfall_array)