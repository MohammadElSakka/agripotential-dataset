import h5py

def create_hdf5(sentinel_data,altitude_data,meta_serialized,mask,categorical_mask,temperature,rainfall,hdf5_filename='hypercube.h5'):
    with h5py.File(hdf5_filename, 'w') as hdf_file:
        
        # Save altitude data
        hdf_file.create_dataset('altitude', data=altitude_data, compression='gzip', compression_opts=9)

        # Save combined mask data
        hdf_file.create_dataset('mask', data=mask.astype(bool), compression='gzip', compression_opts=9)

        # Save data for each month
        for year in sentinel_data.keys():
            for month in sentinel_data[year].keys():
                dataset_name = f"{month}"               
                hdf_file.create_dataset(dataset_name, data=sentinel_data[year][month], compression='gzip', compression_opts=9)

        # Save metadata
        hdf_file.attrs['meta'] = meta_serialized

        # Save Categorical Mask
        hdf_file.create_dataset('categorical_mask', data=categorical_mask, compression='gzip', compression_opts=9)
        
        # Save temeperature and rainfall dataset
        hdf_file.create_dataset('temperature', data=temperature, compression='gzip', compression_opts=9)
        hdf_file.create_dataset('rainfall', data=temperature, compression='gzip', compression_opts=9)