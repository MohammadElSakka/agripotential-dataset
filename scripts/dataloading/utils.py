import numpy as np


def compute_boundaries(binary_mask) -> tuple[int, int, int, int]:
    # Get indices of non-zero elements (where the mask is 1)
    indices = np.argwhere(binary_mask == 1)
    
    # If no elements are found, return a default or invalid boundary
    if indices.size == 0:
        return -1, -1, -1, -1
    
    # Compute boundaries using min and max along each dimension
    up = indices[:, 0].min()  # Smallest row index
    down = indices[:, 0].max()  # Largest row index
    left = indices[:, 1].min()  # Smallest column index
    right = indices[:, 1].max()  # Largest column index
    
    return up, down, left, right

def apply_boundaries(data, up, down, left, right):
    return data[..., up:down+1, left:right+1]

def get_valid_indexes(binary_mask):
    valid_indexes = np.argwhere(binary_mask == 1)
    return valid_indexes

def normalize_channel(data: np.ndarray) -> np.ndarray:
    data_min, data_max = (data.min(), data.max())
    return ((data-data_min)/((data_max - data_min)))

def normalize(data: np.ndarray) -> np.ndarray:
    # Apply normalize_channel to each channel along the last axis
    return np.stack([normalize_channel(data[i]) for i in range(data.shape[0])], axis=-1)

