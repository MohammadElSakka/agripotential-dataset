import numpy as np
import rasterio
import h5py
import os

def split_and_filter(binary_mask, block_size = 256):
    h, w = binary_mask.shape
    
    # Découpage en blocs
    blocks = []
    positions = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = binary_mask[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):  # Vérifier la taille
                if np.any(block == 1):  # Garder seulement les blocs contenant des 1
                    blocks.append(block)
                    positions.append((i, j))
    
    # Mélanger les blocs
    indices = np.arange(len(blocks))
    np.random.shuffle(indices)
    blocks = [blocks[i] for i in indices]
    positions = [positions[i] for i in indices]
    
    # Séparer en train (80%) et val (20%)
    split_idx = int(len(blocks) * 0.8)
    train_blocks = blocks[:split_idx]
    val_blocks = blocks[split_idx:]
    train_positions = positions[:split_idx]
    val_positions = positions[split_idx:]
    
    # Créer les masques binaires pour train et val
    train_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    val_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    
    for i, j in train_positions:
        train_mask[i:i+block_size, j:j+block_size] = 1
    
    for i, j in val_positions:
        val_mask[i:i+block_size, j:j+block_size] = 1
    
    return train_mask, val_mask
    
def register_h5_file(config, binary_mask):
    path_h5 = config["path"]

    train_mask, val_mask = split_and_filter(binary_mask, block_size = config["block_size"])

    down, up, left, right = compute_boundaries(binary_mask)

    binary_mask = apply_boundaries(binary_mask, down, up, left, right)
    train_mask = apply_boundaries(train_mask, down, up, left, right)
    val_mask = apply_boundaries(val_mask, down, up, left, right)

    potentials = rasterio.open(config["potentials"]["path"]).read()
    potentials = apply_boundaries(potentials, down, up, left, right)
    potentials = np.argmax(np.transpose(potentials, (1,2,0)), axis=-1)
            
    pixels_to_stations = np.load(config["weather"]["pixels_to_stations"])[0]
    pixels_to_stations = apply_boundaries(pixels_to_stations, down, up, left, right)

    # LOAD ELEVATION DATA
    elevation = rasterio.open(config["elevation"]["path"]).read()[0]
    elevation = apply_boundaries(elevation, down, up, left, right)
    if config["elevation"]["normalization"]:
        elevation = (elevation - np.min(elevation)) / (np.max(elevation) - np.min(elevation))

    # LOAD SENTINEL2 DATA
    sentinel_path = config["sentinel"]["path"]
    sentinel2_files = [f for f in os.listdir(sentinel_path) if f.startswith(config["sentinel"]["prefix"])]
    sentinel2 = {file: rasterio.open(os.path.join(sentinel_path, file)).read() for file in sort_by_index(sentinel2_files)}
    sentinel2 = {k : apply_boundaries(v, down, up, left, right) for k,v in sentinel2.items()}
    sorted_keys = sort_by_index(list(sentinel2))
    sentinel2 = [sentinel2[k] for k in sorted_keys]  
    sentinel2 = np.stack(sentinel2, axis=0)
    if config["sentinel"]["normalization"]:
        min_vals = sentinel2.min(axis=(0, 2, 3), keepdims=True)  # Min par channel
        max_vals = sentinel2.max(axis=(0, 2, 3), keepdims=True)  # Max par channel
        sentinel2 = (sentinel2 - min_vals) / (max_vals - min_vals)

   # Create HDF5 file to store all data
    with h5py.File(path_h5, 'w') as hf:
        # Store the processed data in HDF5 format
        hf.create_dataset('potentials', data=potentials)
        hf.create_dataset('pixels_to_stations', data=pixels_to_stations)
        hf.create_dataset('elevation', data=elevation)
        hf.create_dataset('binary_mask', data=binary_mask)
        hf.create_dataset('train_mask', data=train_mask)
        hf.create_dataset('val_mask', data=val_mask)
        hf.create_dataset('sentinel2', data=sentinel2)

    print(f"HDF5 file created and saved at: {path_h5}")
    
    print(f"Potentials shape: {potentials.shape}, min: {np.min(potentials)}, max: {np.max(potentials)}")
    print(f"Pixels to stations shape: {pixels_to_stations.shape}, min: {np.min(pixels_to_stations)}, max: {np.max(pixels_to_stations)}")
    print(f"Elevation shape: {elevation.shape}, min: {np.min(elevation)}, max: {np.max(elevation)}")
    print(f"Binary mask shape: {binary_mask.shape}, min: {np.min(binary_mask)}, max: {np.max(binary_mask)}")
    print(f"Train mask shape: {train_mask.shape}, min: {np.min(train_mask)}, max: {np.max(train_mask)}")
    print(f"Val mask shape: {val_mask.shape}, min: {np.min(val_mask)}, max: {np.max(val_mask)}")
    print(f"Sentinel shape: {sentinel2.shape}, min: {np.min(sentinel2)}, max: {np.max(sentinel2)}")    


def ordinal_encode(label, num_classes=5):
    thresholds = torch.arange(1, num_classes)  # [1, 2, 3, 4]
    ordinal_target = (label >= thresholds.view(-1, 1, 1)).float()
    return ordinal_target
    
def compute_boundaries(binary_mask) -> tuple[int, int, int, int]:
    # Get indices of non-zero elements (where the mask is 1)
    indices = np.argwhere(binary_mask == 1)
    
    # If no elements are found, return a default or invalid boundary
    if indices.size == 0:
        return -1, -1, -1, -1
    
    # Compute boundaries using min and max along each dimension
    down = indices[:, 0].min()  # Smallest row index
    up = indices[:, 0].max()  # Largest row index
    left = indices[:, 1].min()  # Smallest column index
    right = indices[:, 1].max()  # Largest column index
    
    return down, up, left, right

def apply_boundaries(data, down, up, left, right):
    return data[..., down:up+1, left:right+1]


def get_valid_indexes(binary_mask):
    valid_indexes = np.argwhere(binary_mask == 1)
    return valid_indexes

def sort_by_index(l):
    return sorted(l, key=lambda x: int(x.split('_')[-1].split('.')[0]))





def normalize_channel(data: np.ndarray) -> np.ndarray:
    data_min, data_max = (data.min(), data.max())
    return ((data-data_min)/((data_max - data_min)))

def normalize(data: np.ndarray) -> np.ndarray:
    # Apply normalize_channel to each channel along the last axis
    return np.stack([normalize_channel(data[i]) for i in range(data.shape[0])], axis=-1)


def calculate_group_weatherdata(group, operation):
    if operation == "mean":
        val = group.mean()
    elif operation == "min":
        val = group.min()
    elif operation == "max":
        val = group.max()
    elif operation == "sum":
        val = group.sum()
    else:
        val = group

    return val  


