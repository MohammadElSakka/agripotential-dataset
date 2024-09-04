import zipfile
import requests
import io
import os
import shutil
import rasterio 
import numpy as np
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from rasterio.mask import geometry_mask

class Dataset:
    def __init__(self, download=False) -> None:
        self.__data_path = {}
        self.__sentinel2_meta = {} 
        if download:
            self.__download()
        self.__get_paths()
        self.__get_sentinel2_meta()

    def __download(self) -> bool:
        if os.path.exists("data"):
            user_input = input("An existing data directory already exists. Override? (y/n) ")
            while user_input != "y" and user_input != "n":
                user_input = input("An existing data directory already exists. Override? (y/n) ")
            if user_input == "y":
                shutil.rmtree("data")
            else:
                print(f"Failed to download zip file: data directory cannot be overrided.")
                return False 
        
        response = requests.get("https://cloud.irit.fr/s/IVLWTCPV5AmNSZp/download")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                zip.extractall("data")
            return True
        else:
            print(f"Failed to download zip file [status code {response.status_code}].")
            return False
    
    def __get_sentinel2_meta(self) -> None:
        self.__sentinel2_meta = rasterio.open(self.__data_path["sentinel2"][2019][1]['B02']).meta     
    
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

        self.__data_path["labels"] = os.path.abspath("./data/dataset.geojson")
        self.__data_path["elevation"] = os.path.abspath("./data/elevation_data/elevation_10m.tiff")
        self.__data_path["weather"] = os.path.abspath("./data/weather.csv")
    
    def __get_weather_path(self) -> str:
        return self.__data_path["weather"]
    
    def __get_sentinel2_path(self, month: int, year: int = 2019) -> str:
        return self.__data_path["sentinel2"][year][month],

    def __get_elevation_path(self) -> str:
        return self.__data_path["elevation"]

    def __get_labels_path(self) -> str:
        return self.__data_path["elevation"]

