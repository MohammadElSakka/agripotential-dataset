import zipfile
import requests
import io
import os
import shutil
import numpy as np
from PIL import Image
import rasterio 

from rasterio.warp import reproject, Resampling
from rasterio.mask import geometry_mask

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import skimage.draw
from skimage.draw import polygon

from scripts.utils import gps_to_pixel 

class Dataset:
    def __init__(self, download=False, root=".", ind_conf=2, iddiz=2, icucs=2, block_size=256) -> None:
        np.random.seed(42)
        
        self.__data_path = {}
        self.__meta = {} 
        self.ind_conf = ind_conf
        self.iddiz = iddiz
        self.icucs = icucs
        self.root = root

        self.block_size = block_size
        if download:
            self.__download()
        self.__save_paths()
        self.__save_meta()

    def __download(self) -> bool:
        response = requests.get("https://cloud.irit.fr/s/PZgfCYiV3F33Sjv/download")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                zip.extractall(f"{self.root}/data")
            return True
        else:
            print(f"Failed to download zip file [status code {response.status_code}].")
            return False
    
    def __save_meta(self) -> None:
        self.__meta = rasterio.open(self.__data_path["sentinel2"][1]).meta     
    
    # private methods to get paths
    def __save_paths(self) -> None:
        self.__data_path["sentinel2"] = {}
        sentinel2_paths = [os.path.abspath(f"{self.root}/data/raw_data/{f}") for f in os.listdir("data/raw_data") if f.endswith("tif")]
        for path in sentinel2_paths:
            self.__data_path["sentinel2"][int(path.split("_")[-2])] = path
        self.__data_path["labels"] = os.path.abspath(f"{self.root}/data/raw_data/labels.geojson")
    
    def __get_labels_path(self) -> str:
        return self.__data_path["labels"]

    def __get_sentinel2_path(self, year: int, month: int) -> str:
        return self.__data_path["sentinel2"][year][month]

    # public methods to export data to files ready to be used
    def get_meta(self):
        return self.__meta
    
    def get_binary_mask(self) -> np.ndarray:
        ind_conf = self.ind_conf
        icucs = self.icucs
        iddiz = self.iddiz
        
        gdf = gpd.read_file(self.__get_labels_path())
        gdf = gdf.to_crs(self.__meta['crs'])
        shape = self.__meta['height'], self.__meta['width']
        transform = self.__meta['transform']

        labels = ["Limite", "Assez_limite", "Moyen", "Assez_fort", "Fort_a_tres_fort"]
        mask = np.zeros((shape), dtype=bool)
        for label in labels:
            label_mask = geometry_mask(gdf[gdf["pot_global"] == label].loc[(gdf["ind_conf"] >= ind_conf)| (gdf["icucs"] >= icucs )| (gdf["iddiz"] >= iddiz)].geometry, 
                                out_shape=shape, 
                                transform=transform, 
                                invert=True)
            mask |= label_mask
        return mask
    
    # potentials = {pot_global, potent_gc, potent_ma, potent_vit}  
    def get_categorical_potential_data(self, potential: str) -> np.ndarray: 
        ind_conf = self.ind_conf
        icucs = self.icucs
        iddiz = self.iddiz
        gdf = gpd.read_file(self.__data_path["labels"])
        gdf = gdf.to_crs(self.__meta["crs"])
        labels = ["Limite", "Assez_limite", "Moyen", "Assez_fort", "Fort_a_tres_fort"]
        categorical_mask = np.zeros((self.__meta['height'], self.__meta['width'], 5))
        for i, label in enumerate(labels):
            mask = geometry_mask(gdf[gdf[potential]==label].loc[(gdf["ind_conf"] >= ind_conf) | (gdf["icucs"] >= icucs) | (gdf["iddiz"] >= iddiz)].geometry, 
                                    out_shape=(self.__meta['height'], self.__meta['width']), 
                                    transform=self.__meta['transform'], 
                                    invert=True)
            categorical_mask[:,:,i][mask] = 1
        return categorical_mask
     
    def get_sentinel2_data(self, year: int, month: int) -> np.ndarray:
        sentinel2_data = np.zeros((self.__meta["height"], self.__meta["width"], 4), np.float32)
        band_list = sorted(list(self.__get_sentinel2_path(year, month).keys()))
        # change band_list
        assert band_list == ["B02", "B03", "B04", "B08"], f"Something is wrong with the bands of {month}/{year}"
        for i in range(len(band_list)):
            band = band_list[i]
            band_path = self.__get_sentinel2_path(year, month)[band]
            # super resolution when needed
            sentinel2_data[:,:,i] = rasterio.open(band_path).read(1)
        # Compute VI
        return sentinel2_data

    def split_and_filter(self):
        block_size = self.block_size
        binary_mask = self.get_binary_mask()
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
        return train_mask, train_positions, val_mask, val_positions
    
    def process(self) -> None:
        binary_mask = self.get_binary_mask()
        train_mask, train_positions, val_mask, val_positions = self.split_and_filter()
        vit_potential = self.get_categorical_potential_data(potential="potent_vit")
        return train_mask, train_positions, val_mask, val_positions, vit_potential, binary_mask
        
    
    def export_dataset(self) -> None:
        dataset_path = f"{self.root}/data/dataset.zip"
        if os.path.exists(dataset_path):
            user_input = input("An existing dataset.zip already exists. Override? (y/n) ")
            while user_input != "y" and user_input != "n":
                user_input = input("An existing dataset.zip  directory already exists. Override? (y/n) ")
            if user_input == "y":
                os.remove(dataset_path)
            else:
                print(f"Failed to export dataset: dataset.zip cannot be overrided.")
                return False 
