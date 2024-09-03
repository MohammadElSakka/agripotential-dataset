# from scripts.data import get_mask, get_meta , serialize_meta, get_altitude_data,get_data,get_categorical_mask
# Retrieve data 
# sentinel_data = get_data()

from scripts.dataset import Dataset

Dataset().__download__()


#retrieve meta and serialized it 
# meta = get_meta()

# # Retrieve altitude
# altitude_data = get_altitude_data(meta)
# print(altitude_data.shape)

# mask = get_mask(meta)
# print(altitude_data.shape)

# categorical_mask = get_categorical_mask(meta)
# print(categorical_mask.shape)