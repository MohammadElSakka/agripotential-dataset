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
        self.valid_indexes = get_valid_indexes(binary_mask)

        # LOAD WEATHER DATA
        self.weather_data = pd.read_csv(self.config["weather"]["path"])
        self.pixels_to_stations = np.load(self.config["weather"]["pixels_to_stations"])[0]
        self.selected_columns = self.config["weather"]["selected_columns"]

        # LOAD ELEVATION DATA
        self.elevation = rasterio.open(self.config["elevation"]["path"]).read()[0]
        print(self.elevation.shape)
        self.elevation = apply_boundaries(self.elevation, up, down, left, right)

        # LOAD SENTINEL2 DATA
        sentinel_path = self.config["sentinel"]["path"]
        sentinel2_files = [f for f in os.listdir(sentinel_path) if f.startswith(self.config["sentinel"]["prefix"])]
        self.sentinel2 = {i: rasterio.open(os.path.join(sentinel_path, sentinel2_files[i])).read() for i in range(len(sentinel2_files))}
        for key in self.sentinel2:
            print(self.sentinel2[key].shape)
        self.sentinel2 = {k : apply_boundaries(v, up, down, left, right) for k,v in self.sentinel2.items()}
        if self.config["sentinel"]["normalization"]:
            self.sentinel2 = {k : normalize(v) for k,v in self.sentinel2.items()}
        for key in self.sentinel2:
            print(self.sentinel2[key].shape)
            
        # LOAD POTENTIAL DATA
        self.potentials = {k : rasterio.open(f).read() for k,f in self.config["potentials"]["path"].items()}
        for key in self.potentials:
            print(self.potentials[key].shape)
        self.potentials = {k : apply_boundaries(v, up, down, left, right) for k,v in self.potentials.items()}


    def __len__(self):
        return len(self.valid_indexes)
    
    def __getitem__(self, idx):
        valid_index = self.valid_indexes[idx]

        station_id = self.pixels_to_stations[valid_index]
        data = self.weather_data[self.weather_data["ID"] == station_id]
        acc_weather_data = []
        for c,o in self.selected_columns.items() :
            weather_data_c = data[c]
            if len(o)==0 :
                for d in weather_data_c :
                    acc_weather_data.append(d)
            elif "mean" in o :
                acc_weather_data.append(weather_data_c.mean())
            elif "min" in o :
                acc_weather_data.append(weather_data_c.min())
            elif "max" in o :
                acc_weather_data.append(weather_data_c.max())
            elif "sum" in o :
                acc_weather_data.append(weather_data_c.sum())

        elevation = self.elevation[valid_index]

        sentinel2_data = [self.sentinel2[k][valid_index] for k in sorted(self.sentinel2.keys())] 

        potential = [self.potentials[k][valid_index] for k in sorted(self.potentials.keys())]
        if len(potential) == 1 : # SIngle pred
            potential = potential[0]

        return acc_weather_data, elevation, sentinel2_data, potential