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
        self.data_path = {}
        self.sentinel2_meta = {} 
        if download:
            self.__download__()
        self.__get_paths__()
        self.__get_sentinel2_meta__()

    def __download__(self) -> bool:
        if os.path.exists("data"):
            user_input = input("An existing data directory already exists. Override? (y/n)")
            while user_input != "y" and user_input != "n":
                user_input = input("An existing data directory already exists. Override? (y/n)")
            if user_input == "y":
                shutil.rmtree("data")
            else:
                print(f"Failed to download zip file: data directory cannot be overrided.")
                return False 
        
        response = requests.get("https://cloud.irit.fr/s/cyt2sL9sASd4EJl/download")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                os.mkdir("data")
                zip.extractall("data")
            return True
        else:
            print(f"Failed to download zip file [status code {response.status_code}].")
            return False
        
    def __get_paths__(self) -> None:
        self.data_path["sentinel2"] = {}
        years = os.listdir("data/sentinel2_bands")
        for year in years:
            y = int(year)
            self.data_path["sentinel2"][y] = {}
            months = os.listdir(f"data/sentinel2_bands/{year}")
            for month in months:
                m = int(month.split("_")[0])
                self.data_path["sentinel2"][y][m] = {}
                bands = os.listdir(f"data/sentinel2_bands/{year}/{month}/") 
                for band in bands:
                    band_name = band.split("_")[2]
                    self.data_path["sentinel2"][y][m][band_name] = os.path.abspath(f"./data/sentinel2_bands/{year}/{month}/{band}")

        self.data_path["labels"] = os.path.abspath("./data/dataset.geojson")
        self.data_path["elevation"] = os.path.abspath("./data/elevation_data/raw_elevation_data_10m.tif")
        self.data_path["weather"] = os.path.abspath("./data/weather_data/")

    def __get_sentinel2_meta__(self) -> None:
        self.sentinel2_meta = rasterio.open(self.data_path["sentinel2"][2019][1]['B02']).meta

    # save to raw file
    def get_altitude_data(self):

        def get_mask(meta):
            gdf = gpd.read_file( self.data_path["labels"])
            gdf = gdf.to_crs(meta['crs'])
            shape = meta['height'], meta['width']
            transform = meta['transform']

            labels = ["Limite", "Assez_limite", "Moyen", "Assez_fort", "Fort_a_tres_fort"]
            mask = np.zeros((shape), dtype=bool)
            for label in labels:
                label_mask = geometry_mask(gdf[gdf["pot_global"] == label].geometry, 
                                    out_shape=shape, 
                                    transform=transform, 
                                    invert=True)
                mask |= label_mask
            return mask 
        # Resample altitude data to match the other bands
        elevation_path = self.data_path["elevation"]
        with rasterio.open(elevation_path) as src:
            altitude_resampled = np.zeros((self.sentinel2_meta['height'], self.sentinel2_meta['width']), np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=altitude_resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.sentinel2_meta['transform'],
                dst_crs=self.sentinel2_meta['crs'],
                resampling=Resampling.nearest)
        
        mask = get_mask(self.sentinel2_meta)
        altitude_resampled[altitude_resampled <-500] = 0
        altitude_resampled[~mask] = np.nan 

        with rasterio.open('elevation_10m.tiff', 'w', **self.sentinel2_meta) as dst:
            dst.write(np.expand_dims(altitude_resampled, axis=0))  # Write the array, it automatically writes all bands
        

        return altitude_resampled

