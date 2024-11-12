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

class Dataset:
    def __init__(self, download=False) -> None:
        self.__data_path = {}
        self.__meta = {} 
        if download:
            self.__download()
        self.__get_paths()
        self.__get_meta()

    def __download(self) -> bool:
        response = requests.get("https://cloud.irit.fr/s/pJ8LZfamduAR88a/download")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                zip.extractall("data")
            return True
        else:
            print(f"Failed to download zip file [status code {response.status_code}].")
            return False
    
    def __get_meta(self) -> None:
        self.__meta = rasterio.open(self.__data_path["sentinel2"][2019][1]['B02']).meta     
    
    # private methods to get paths
    def __get_paths(self) -> None:
        self.__data_path["sentinel2"] = {}
        years = os.listdir("data/sentinel2_bands")
        for year in years:
            y = int(year)
            self.__data_path["sentinel2"][y] = {}
            months = os.listdir(f"data/sentinel2_bands/{year}")
            for month in months:
                m = int(month.split("_")[0])
                self.__data_path["sentinel2"][y][m] = {}
                bands = os.listdir(f"data/sentinel2_bands/{year}/{month}/") 
                for band in bands:
                    band_name = band.split("_")[2]
                    self.__data_path["sentinel2"][y][m][band_name] = os.path.abspath(f"./data/sentinel2_bands/{year}/{month}/{band}")

        self.__data_path["labels"] = os.path.abspath("./data/labels.geojson")
        self.__data_path["elevation"] = os.path.abspath("./data/elevation/raw_elevation_data_10m.tif")
        self.__data_path["weather"] = os.path.abspath("./data/weather/weather.csv")
    
    def __get_weather_path(self) -> str:
        return self.__data_path["weather"]
    
    def __get_sentinel2_path(self, year: int, month: int) -> str:
        return self.__data_path["sentinel2"][year][month]

    def __get_elevation_path(self) -> str:
        return self.__data_path["elevation"]

    def __get_labels_path(self) -> str:
        return self.__data_path["labels"]
    
    # public methods to export data to files ready to be used

    def get_binary_mask(self, ind_conf: int = 2) -> np.ndarray:
        gdf = gpd.read_file(self.__get_labels_path())
        gdf = gdf.to_crs(self.__meta['crs'])
        shape = self.__meta['height'], self.__meta['width']
        transform = self.__meta['transform']

        labels = ["Limite", "Assez_limite", "Moyen", "Assez_fort", "Fort_a_tres_fort"]
        mask = np.zeros((shape), dtype=bool)
        for label in labels:
            label_mask = geometry_mask(gdf[gdf["pot_global"] == label].loc[gdf["ind_conf"] > ind_conf].geometry, 
                                out_shape=shape, 
                                transform=transform, 
                                invert=True)
            mask |= label_mask
        return mask
    
    # potentials = {pot_global, potent_gc, potent_ma, potent_vit}  
    def get_categorical_potential_data(self, potential: str, ind_conf: int =2) -> np.ndarray: 
        gdf = gpd.read_file(self.__data_path["labels"])
        gdf = gdf.to_crs(self.__meta["crs"])
        labels = ["Limite", "Assez_limite", "Moyen", "Assez_fort", "Fort_a_tres_fort"]
        categorical_mask = np.zeros((self.__meta['height'], self.__meta['width'], 6))
        for i, label in enumerate(labels):
            mask = geometry_mask(gdf[gdf[potential]==label].loc[gdf["ind_conf"] > ind_conf].geometry, 
                                    out_shape=(self.__meta['height'], self.__meta['width']), 
                                    transform=self.__meta['transform'], 
                                    invert=True)
            categorical_mask[:,:,i][mask] = 1
        return categorical_mask
     
    def get_elevation_data(self) -> np.ndarray:
        elevation_path = self.__get_elevation_path()
        meta = self.__meta
        with rasterio.open(elevation_path) as src:
            elevation = np.zeros((meta['height'], meta['width']), np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=elevation,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=meta['transform'],
                dst_crs=meta['crs'],
                resampling=Resampling.nearest)
        mask = self.get_binary_mask()
        elevation[elevation <-500] = 0
        elevation[~mask] = 0
        return elevation    
    
    # INST: duree totale d'insolation sur la decade (en mn)
    # RR: cumul decadaire des hauteurs de precipitation (en mm et 1/10)
    # TN: moyenne decadaire de la temperature minimale (en °C et 1/10)
    # TX: moyenne decadaire de la temperature maximale (en °C et 1/10)
    # ETP: evapotranspiration Penman decadaire (en mm et 1/10)

    def get_weather_data(self) -> pd.DataFrame:
        weather_path = self.__get_weather_path()
        df = pd.read_csv(weather_path, sep=";")
        station_filter = df["NOM_USUEL"] == "MONTPELLIER-AEROPORT"
        date_filter = df["AAAAMM"].astype(str).str.startswith("2019")
        df = df[station_filter & date_filter]
        df['Temperature'] = df[['TX', 'TN']].mean(axis=1).round(1)
        df['Precipitation'] = df["RR"].round(1)
        df['Evapotranspiration'] = df["ETP"].round(1)
        df['Insolation'] = df["INST"]

        df['Date'] = df["AAAAMM"]
        df['Suffix'] = df.groupby('Date').cumcount() + 1
        df['Date'] = df['Date'].astype(str) + '_' + df['Suffix'].astype(str).str.zfill(2)
        df = df.drop(columns=['Suffix'])

        df = df[["Date", "Temperature", "Precipitation", "Evapotranspiration", "Insolation"]]
        return df 
    
        # import matplotlib.pyplot as plt
        # ax = df["RR"].plot(kind="bar")
        # ax.set_xticklabels(df["AAAAMM"].astype(str).str[-2:], rotation=90)
        # plt.savefig("rr.png")

    def get_sentinel2_data(self, year: int, month: int) -> np.ndarray:
        sentinel2_data = np.zeros((self.__meta["height"], self.__meta["width"], 4), np.float32)
        band_list = sorted(list(self.__get_sentinel2_path(year, month).keys()))
        assert band_list == ["B02", "B03", "B04", "B08"], f"Something is wrong with the bands of {month}/{year}"
        mask = self.get_binary_mask()
        for i in range(len(band_list)):
            band = band_list[i]
            band_path = self.__get_sentinel2_path(year, month)[band]
            sentinel2_data[:,:,i] = rasterio.open(band_path).read(1)
            
        return sentinel2_data


    def export_dataset(self) -> None:
        def __save_np_as_tiff(file_path: str, arr: np.ndarray) -> None:
            meta = dict(self.__meta)
            meta["count"] = arr.shape[0]
            with rasterio.open(file_path,"w",**meta) as f:
                f.write(arr)
        
        dataset_path = "data/dataset.zip"
        if os.path.exists(dataset_path):
            user_input = input("An existing dataset.zip already exists. Override? (y/n) ")
            while user_input != "y" and user_input != "n":
                user_input = input("An existing dataset.zip  directory already exists. Override? (y/n) ")
            if user_input == "y":
                os.remove(dataset_path)
            else:
                print(f"Failed to export dataset: dataset.zip cannot be overrided.")
                return False 
        
        tmp_dir = "data/.tmp/"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        # features (inputs)
        # elevation : save as tif = 1 tiff
        elevation_data = np.expand_dims(self.get_elevation_data(), axis=0)
        __save_np_as_tiff(tmp_dir+"elevation.tif", elevation_data)

        # save as csv = 1 csv
        weather_data = self.get_weather_data()
        weather_data.to_csv(tmp_dir+'weather_data.csv', index=False)
        
        for year in self.__data_path["sentinel2"]:
            for month in self.__data_path["sentinel2"][year]:
                sentinel2_data = np.transpose(self.get_sentinel2_data(year, month), (2, 0, 1))    
                # save as tiff/month = 12 tiff
                __save_np_as_tiff(tmp_dir+f"sentinel2_{year}_{month}.tif",sentinel2_data)

        # labels (outputs)
        # save as tiff per each category = 4 tiffs
        global_potential = np.transpose(self.get_categorical_potential_data(potential="pot_global"), (2, 0, 1))
        gc_potential = np.transpose(self.get_categorical_potential_data(potential="potent_gc"), (2, 0, 1))
        ma_potential = np.transpose(self.get_categorical_potential_data(potential="potent_ma"), (2, 0, 1))
        vit_potential = np.transpose(self.get_categorical_potential_data(potential="potent_vit"), (2, 0, 1))
        __save_np_as_tiff(tmp_dir+"global_potential.tif", global_potential)
        __save_np_as_tiff(tmp_dir+"gc_potential.tif", gc_potential)
        __save_np_as_tiff(tmp_dir+"ma_potential.tif", ma_potential)
        __save_np_as_tiff(tmp_dir+"vit_potential.tif", vit_potential)

        binary_mask = self.get_binary_mask()
        Image.fromarray(binary_mask).save(tmp_dir+"binary_mask.png")

        with zipfile.ZipFile(dataset_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if file.split(".")[-1] != "xml":
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmp_dir)
                        zipf.write(file_path, arcname)

        shutil.rmtree(tmp_dir)