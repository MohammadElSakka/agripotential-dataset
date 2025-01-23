import yaml
import os 
import numpy as np
from PIL import Image
from rasterio import rasterio
import pandas as pd

import torch
from torch.utils.data import Dataset

from scripts.dataloading.utils import *

class AgriPixelDataset(Dataset):
    def __init__(self, config_path):
        # Charger la configuration depuis le fichier YAML
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # LOAD BINARY MASK, CROP AND SEARCH VALID POSITIONS IN CROPPED MASK
        binary_mask_path = self.config["mask"]["path"]
        binary_mask = np.array(Image.open(binary_mask_path))
        up, down, left, right = compute_boundaries(binary_mask)
        binary_mask = apply_boundaries(binary_mask, up, down, left, right)
        self.valid_pixels = get_valid_indexes(binary_mask)


        # LOAD POTENTIAL DATA
        self.potentials = {k : rasterio.open(f).read() for k,f in self.config["potentials"]["path"].items()}
        self.potentials = {k : apply_boundaries(v, up, down, left, right) for k,v in self.potentials.items()}
        self.potentials = {k : np.argmax(np.transpose(v, (1,2,0)), axis=-1) for k,v in self.potentials.items()}
        #self.potentials = {k : np.where(binary_mask, v, np.nan) for k,v in self.potentials.items()}

      
        # LOAD WEATHER DATA
        self.weather_data = pd.read_csv(self.config["weather"]["path"])
        self.weather_data.sort_values(by=["ID", 'Date'])
      
        self.pixels_to_stations = np.load(self.config["weather"]["pixels_to_stations"])[0]
        self.pixels_to_stations = apply_boundaries(self.pixels_to_stations, up, down, left, right)
      
        self.selected_columns = self.config["weather"]["selected_columns"]

      
        # LOAD ELEVATION DATA
        self.elevation = rasterio.open(self.config["elevation"]["path"]).read()[0]
        self.elevation = apply_boundaries(self.elevation, up, down, left, right)

      
        # LOAD SENTINEL2 DATA
        sentinel_path = self.config["sentinel"]["path"]
        sentinel2_files = [f for f in os.listdir(sentinel_path) if f.startswith(self.config["sentinel"]["prefix"])]
        self.sentinel2 = {file: rasterio.open(os.path.join(sentinel_path, file)).read() for file in sort_by_index(sentinel2_files)}
        self.sentinel2 = {k : apply_boundaries(v, up, down, left, right) for k,v in self.sentinel2.items()}
        if self.config["sentinel"]["normalization"]:
            self.sentinel2 = {k : normalize(v) for k,v in self.sentinel2.items()}

          
    def __len__(self):
        return len(self.valid_pixels)
    
    def __getitem__(self, idx):
        valid_pixel = self.valid_pixels[idx]

        station_id = self.pixels_to_stations[valid_pixel[0],valid_pixel[1]]
        data = self.weather_data[self.weather_data["ID"] == station_id]
        data['group'] = (data['Date'].str.endswith('_1')).cumsum()
        acc_weather_data = []
        for c,o in self.selected_columns.items():
            result = data[['group',c]].groupby('group').apply(calculate_group_weatherdata, column=c, operation=o).reset_index(drop=True)
            if o is "" :
                # Concatenate
                r = result.sum()
            else :
                r = result.tolist()
            acc_weather_data.append(r)
        acc_weather_data = torch.stack([torch.tensor(arr) for arr in acc_weather_data])

        elevation = self.elevation[valid_pixel[0],valid_pixel[1]]

        sentinel2_data = [self.sentinel2[k][valid_pixel[0],valid_pixel[1]] for k in self.sentinel2.keys()] 
        sentinel2_data = torch.stack([torch.tensor(arr) for arr in sentinel2_data])

        potential = [self.potentials[k][valid_pixel[0],valid_pixel[1]] for k in self.potentials.keys()]
        if len(potential) == 1 : # Single pred
            potential = potential[0]

        return acc_weather_data,  torch.tensor(elevation.astype(np.int16)), sentinel2_data,  torch.tensor(potential)