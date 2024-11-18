import numpy as np
import rasterio
import rasterio.transform
from pyproj import Proj, Transformer

def compute_boundaries(dataset, saved=True) -> tuple[int, int, int, int]:
    if saved:
        return 3321, 10979, 0, 9401 
    
    binary_mask = dataset.get_binary_mask()
    up, down, left, right = -1, -1, 1e8, -1
    last_down = 0
    
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            
            if binary_mask[i][j] == 1 and up == -1:
                up = i
            
            if binary_mask[i][j] == 1 and last_down == 0 and i != down:
                down = i
            
            if binary_mask[i][j] == 1 and j<left:
                left = j
    
            if binary_mask[i][j] == 1 and j>right:
                right = j
    
            last_down = binary_mask[i][j]
    # Will return 3321 10979 0 9401
    return up, down, left, right

def pixel_to_gps(data: np.ndarray, i, j: int, meta: dict) -> dict:

    return

def gps_to_pixel(gps: dict, meta: dict) -> tuple[int, int]:
    transformer = Transformer.from_crs(crs_from='WGS84', crs_to=meta['crs'])
    x, y = transformer.transform([gps['LAT']], [gps['LON']])
    x, y = ~meta['transform'] * (x[0], y[0])
    return int(x), int(y)