import zipfile
import requests
import io
import os
import shutil
import numpy as np
from PIL import Image
import h5py

import rasterio 
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.mask import geometry_mask

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import skimage.draw
from skimage.draw import polygon

from scripts.utils import gps_to_pixel 

class Dataset:
    def __init__(self, download=False, root=".", ind_conf=2, iddiz=2, icucs=2) -> None:
        np.random.seed(42)
        
        self.__data_path = {}
        self.__meta = {} 
        self.ind_conf = ind_conf
        self.iddiz = iddiz
        self.icucs = icucs
        self.root = root

        self.block_size = 256
        self.patch_size = 128
        
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
    
    def get_meta(self):
        return self.__meta
    
    def get_binary_mask(self) -> np.ndarray:
        ind_conf = self.ind_conf
        icucs = self.icucs
        iddiz = self.iddiz
        
        gdf = gpd.read_file(self.__data_path["labels"])
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
        v = split_idx+(len(blocks)-split_idx)//2
        
        train_blocks = blocks[:split_idx]
        sample = train_blocks[199]
        val_blocks = blocks[split_idx:v]
        test_blocks = blocks[v:]
        
        train_positions = positions[:split_idx]
        val_positions = positions[split_idx:v]
        test_positions = positions[v:]
        
        # Créer les masques binaires pour train et val
        train_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        val_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        test_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        
        for i, j in train_positions:
            train_mask[i:i+block_size, j:j+block_size] = 1
        
        for i, j in val_positions:
            val_mask[i:i+block_size, j:j+block_size] = 1
        
        for i, j in test_positions:
            test_mask[i:i+block_size, j:j+block_size] = 1  
        
        return train_mask, train_positions, val_mask, val_positions, test_mask, test_positions

    def normalize_sentinel2(self, arr, cut):
        data = np.clip(arr/10000, 0.00001, cut)
        data = data/cut
        return data

    def generate_label_grid(self, positions, label):
        patch_size = self.patch_size
        labels = []
        for i, j in positions:
            patch = label[i:i+patch_size, j:j+patch_size]
            labels.append(patch)
        return labels
        
    def generate_sentinel2_grid(self, positions):
        patch_size = self.patch_size
        data = [ [] for _ in range(12)]
        for month in range(1, 13):
            for i, j in positions:
                patch = rasterio.open(self.__data_path["sentinel2"][month]).read(window=Window(j, i, patch_size, patch_size))
                patch = self.normalize_sentinel2(patch, 0.25)
                data[month-1].append(patch)
        return data
    
    def blocks_to_patches(self, positions, binary_mask):
        patch_size = self.patch_size
        overlay = patch_size//2
        pos = []
        for i, j in positions:
            for k in range(3):
                for l in range(3):
                    ii = i+k*overlay
                    jj = j+l*overlay
                    if  np.any(binary_mask[ii:ii+patch_size, jj:jj+patch_size]) == 1:
                        pos += [[ii, jj]]

        return pos
    
    def export_dataset(self) -> None:
        binary_mask = self.get_binary_mask()
        _, train_positions, _, val_positions, _, test_positions = self.split_and_filter()

        train_positions = self.blocks_to_patches(train_positions, binary_mask)
        val_positions = self.blocks_to_patches(val_positions, binary_mask)
        test_positions = self.blocks_to_patches(test_positions, binary_mask)

        train_data = self.generate_sentinel2_grid(train_positions)
        val_data = self.generate_sentinel2_grid(val_positions)
        test_data = self.generate_sentinel2_grid(test_positions)
        
        vit_potential = self.get_categorical_potential_data(potential="potent_vit")
        ma_potential = self.get_categorical_potential_data(potential="potent_ma")
        gc_potential = self.get_categorical_potential_data(potential="potent_gc")
        
        with h5py.File("dataset.h5", 'w') as hf:
            
            # TRAIN
            train_hf = hf.create_group("train") 
            train_sentinel_hf = hf.create_group("train/sentinel2")
            
            train_labels_hf = hf.create_group("train/labels") 

            train_sentinel_hf.create_dataset('01_january_2019', data=np.array(train_data[0]))
            train_sentinel_hf.create_dataset('02_february_2019', data=np.array(train_data[1]))
            train_sentinel_hf.create_dataset('03_march_2019', data=np.array(train_data[2]))
            train_sentinel_hf.create_dataset('04_april_2019', data=np.array(train_data[3]))
            train_sentinel_hf.create_dataset('05_may_2019', data=np.array(train_data[4]))
            train_sentinel_hf.create_dataset('06_june_2019', data=np.array(train_data[5]))
            train_sentinel_hf.create_dataset('07_july_2019', data=np.array(train_data[6]))
            train_sentinel_hf.create_dataset('08_august_2019', data=np.array(train_data[7]))
            train_sentinel_hf.create_dataset('09_september_2019', data=np.array(train_data[8]))
            train_sentinel_hf.create_dataset('10_october_2019', data=np.array(train_data[9]))
            train_sentinel_hf.create_dataset('11_november_2019', data=np.array(train_data[10]))
            train_sentinel_hf.create_dataset('12_december_2019', data=np.array(train_data[11]))
        
            train_labels_hf.create_dataset("viticulture", data=np.array(self.generate_label_grid(train_positions, vit_potential))) 
            train_labels_hf.create_dataset("market", data=np.array(self.generate_label_grid(train_positions, ma_potential))) 
            train_labels_hf.create_dataset("field", data=np.array(self.generate_label_grid(train_positions, gc_potential))) 
        
            # TEST
            test_hf = hf.create_group("test") 
            test_sentinel_hf = hf.create_group("test/sentinel2") 
            
            test_labels_hf = hf.create_group("test/labels") 
            
            test_sentinel_hf.create_dataset('01_january_2019', data=np.array(test_data[0]))
            test_sentinel_hf.create_dataset('02_february_2019', data=np.array(test_data[1]))
            test_sentinel_hf.create_dataset('03_march_2019', data=np.array(test_data[2]))
            test_sentinel_hf.create_dataset('04_april_2019', data=np.array(test_data[3]))
            test_sentinel_hf.create_dataset('05_may_2019', data=np.array(test_data[4]))
            test_sentinel_hf.create_dataset('06_june_2019', data=np.array(test_data[5]))
            test_sentinel_hf.create_dataset('07_july_2019', data=np.array(test_data[6]))
            test_sentinel_hf.create_dataset('08_august_2019', data=np.array(test_data[7]))
            test_sentinel_hf.create_dataset('09_september_2019', data=np.array(test_data[8]))
            test_sentinel_hf.create_dataset('10_october_2019', data=np.array(test_data[9]))
            test_sentinel_hf.create_dataset('11_november_2019', data=np.array(test_data[10]))
            test_sentinel_hf.create_dataset('12_december_2019', data=np.array(test_data[11]))
        
            test_labels_hf.create_dataset("viticulture", data=np.array(self.generate_label_grid(test_positions, vit_potential))) 
            test_labels_hf.create_dataset("market", data=np.array(self.generate_label_grid(test_positions, ma_potential))) 
            test_labels_hf.create_dataset("field", data=np.array(self.generate_label_grid(test_positions, gc_potential))) 
        
            # VALIDATION
            val_hf = hf.create_group("val") 
            val_sentinel_hf = hf.create_group("val/sentinel2") 
            
            val_labels_hf = hf.create_group("val/labels") 
            
            val_sentinel_hf.create_dataset('01_january_2019', data=np.array(val_data[0]))
            val_sentinel_hf.create_dataset('02_february_2019', data=np.array(val_data[1]))
            val_sentinel_hf.create_dataset('03_march_2019', data=np.array(val_data[2]))
            val_sentinel_hf.create_dataset('04_april_2019', data=np.array(val_data[3]))
            val_sentinel_hf.create_dataset('05_may_2019', data=np.array(val_data[4]))
            val_sentinel_hf.create_dataset('06_june_2019', data=np.array(val_data[5]))
            val_sentinel_hf.create_dataset('07_july_2019', data=np.array(val_data[6]))
            val_sentinel_hf.create_dataset('08_august_2019', data=np.array(val_data[7]))
            val_sentinel_hf.create_dataset('09_september_2019', data=np.array(val_data[8]))
            val_sentinel_hf.create_dataset('10_october_2019', data=np.array(val_data[9]))
            val_sentinel_hf.create_dataset('11_november_2019', data=np.array(val_data[10]))
            val_sentinel_hf.create_dataset('12_december_2019', data=np.array(val_data[11]))
        
            val_labels_hf.create_dataset("viticulture", data=np.array(self.generate_label_grid(val_positions, vit_potential))) 
            val_labels_hf.create_dataset("market", data=np.array(self.generate_label_grid(val_positions, ma_potential))) 
            val_labels_hf.create_dataset("field", data=np.array(self.generate_label_grid(val_positions, gc_potential)))         
