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
    def __init__(self, download=False, root=".", ind_conf=2, iddiz=2, icucs=2) -> None:
        self.__data_path = {}
        self.__meta = {} 
        self.ind_conf = ind_conf
        self.iddiz = iddiz
        self.icucs = icucs
        self.root = root
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
        self.__meta = rasterio.open(self.__data_path["sentinel2"][2019][1]['B02']).meta     
    
    # private methods to get paths
    def __save_paths(self) -> None:
        self.__data_path["sentinel2"] = {}
        years = os.listdir(f"{self.root}/data/raw_data/sentinel2_bands")
        for year in years:
            y = int(year)
            self.__data_path["sentinel2"][y] = {}
            months = os.listdir(f"{self.root}/data/raw_data/sentinel2_bands/{year}")
            for month in months:
                m = int(month.split("_")[0])
                self.__data_path["sentinel2"][y][m] = {}
                bands = os.listdir(f"{self.root}/data/raw_data/sentinel2_bands/{year}/{month}/") 
                for band in bands:
                    band_name = band.split(".")[0]
                    self.__data_path["sentinel2"][y][m][band_name] = os.path.abspath(f"{self.root}/data/raw_data/sentinel2_bands/{year}/{month}/{band}")

        self.__data_path["labels"] = os.path.abspath(f"{self.root}/data/raw_data/labels.geojson")
        self.__data_path["elevation"] = os.path.abspath(f"{self.root}/data/raw_data/elevation/raw_elevation_data_10m.tif")
        self.__data_path["weather"] = os.path.abspath(f"{self.root}/data/raw_data/weather/weather.csv")
    
    def __get_weather_path(self) -> str:
        return self.__data_path["weather"]
    
    def __get_sentinel2_path(self, year: int, month: int) -> str:
        return self.__data_path["sentinel2"][year][month]

    def __get_elevation_path(self) -> str:
        return self.__data_path["elevation"]

    def __get_labels_path(self) -> str:
        return self.__data_path["labels"]
    

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
        elevation = elevation
        return elevation    
    
    # INST: duree totale d'insolation sur la decade (en mn)
    # RR: cumul decadaire des hauteurs de precipitation (en mm et 1/10)
    # TN: moyenne decadaire de la temperature minimale (en °C et 1/10)
    # TX: moyenne decadaire de la temperature maximale (en °C et 1/10)
    # ETP: evapotranspiration Penman decadaire (en mm et 1/10)

    def get_weather_data(self) -> pd.DataFrame:
        weather_path = self.__get_weather_path()
        df = pd.read_csv(weather_path, sep=";")
        valid_stations = [
            "LES AIRES",
            # "BEZIERS-COURTADE", # missing data
            "MARSEILLAN-INRAE",
            "MONTARNAUD",
            "MONTPELLIER-ENSAM",
            # "MOULES-ET-BAUCELS", # missing data
            "MURVIEL LES BEZIERS",
            "LES PLANS",
            "BEZIERS-VIAS",
            "PRADES LE LEZ",
            "ROUJAN-INRAE",
            "ST ANDRE DE SANGONIS",
            "ST MARTIN DE LONDRES",
            "SETE",
            "SOUMONT",
            "PEZENAS-TOURBES",
            "LA VACQUERIE_SAPC",
            "VAILHAN",
            "VILLENEUVE-LES-MAG-INRAE",
        ]

        df = df[df["NOM_USUEL"].isin(valid_stations)]
        date_filter = df["AAAAMM"].astype(str).str.startswith("2019")
        df = df[date_filter]
        df['Temperature'] = df[['TX', 'TN']].mean(axis=1).round(1)
        df['Max Temperature'] = df["TX"].round(1)
        df['Min Temperature'] = df['TN'].round(1)
        df['Precipitation'] = df["RR"].round(1)
        # df['Evapotranspiration'] = df["ETP"].round(1)
        # df['Insolation'] = df["INST"]

        df['ID'] = df.groupby('NOM_USUEL').ngroup() + 10

        df['Date'] = df["AAAAMM"]
        df['Suffix'] = df.groupby(['ID', 'Date']).cumcount() + 1
        df['Date'] = df['Date'].astype(str) + '_' + df['Suffix'].astype(str)
        df = df.drop(columns=['Suffix'])
        

        # df = df[["NUM_POSTE","NOM_USUEL","LAT","LON", "Date", "Temperature", "Precipitation", "Evapotranspiration", "Insolation"]]
        df = df[["ID", "NOM_USUEL","LAT","LON", "Date", "Temperature", 'Max Temperature', 'Min Temperature', "Precipitation"]]
        return df 
    
    def map_pixels_to_stations(self, df: pd.DataFrame):
        def calculate_distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
        df_copy = pd.DataFrame(df)[["ID", "LAT", "LON"]].drop_duplicates()
        binary_mask = self.get_binary_mask()
        img = np.zeros(binary_mask.shape)
        img[binary_mask] = 1
        meta = self.get_meta()

        locations = {}
        for station_id, lat, lon in zip(df_copy["ID"], df_copy["LAT"], df_copy["LON"]):
            gps = {
                "LAT": lat,
                "LON": lon
                }
            x, y = gps_to_pixel(gps, meta)
            locations[station_id] = [x, y]
            rr, cc = skimage.draw.disk((y, x), radius=2500)
            rr = np.clip(rr, 0, img.shape[0] - 1)
            cc = np.clip(cc, 0, img.shape[1] - 1)
            for r, c in zip(rr, cc):
                if img[r, c]:
                    if img[r, c]==1:
                        img[r, c] = station_id
                    else:
                        d1 = calculate_distance(c,r,x,y)
                        d2 = calculate_distance(c, r, locations[img[r, c]][0],locations[img[r, c]][1])
                        if d1 < d2:
                            img[r, c] = station_id
            if 0<img[r, c] < 10:
                print(img[r,c], r, c, station_id)
        return img

    def get_sentinel2_data(self, year: int, month: int) -> np.ndarray:
        sentinel2_data = np.zeros((self.__meta["height"], self.__meta["width"], 4), np.float32)
        band_list = sorted(list(self.__get_sentinel2_path(year, month).keys()))
        assert band_list == ["B02", "B03", "B04", "B08"], f"Something is wrong with the bands of {month}/{year}"
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
        
        pixels_to_stations =  np.expand_dims(self.map_pixels_to_stations(weather_data), axis=0)
        np.save(tmp_dir+"pixels_to_stations.npy", pixels_to_stations)
        for year in self.__data_path["sentinel2"]:
            for month in self.__data_path["sentinel2"][year]:
                sentinel2_data = np.transpose(self.get_sentinel2_data(year, month), (2, 0, 1))    
                # save as tiff/month = 12 tiff
                __save_np_as_tiff(tmp_dir+f"sentinel2_{year}_{month}.tif",sentinel2_data)


        # labels (outputs)
        # save as tiff per each category = 4 tiffs
        # global_potential = np.transpose(self.get_categorical_potential_data(potential="pot_global"), (2, 0, 1))
        gc_potential = np.transpose(self.get_categorical_potential_data(potential="potent_gc"), (2, 0, 1))
        ma_potential = np.transpose(self.get_categorical_potential_data(potential="potent_ma"), (2, 0, 1))
        vit_potential = np.transpose(self.get_categorical_potential_data(potential="potent_vit"), (2, 0, 1))
        # __save_np_as_tiff(tmp_dir+"global_potential.tif", global_potential)
        __save_np_as_tiff(tmp_dir+"gc_potential.tif", gc_potential)
        __save_np_as_tiff(tmp_dir+"ma_potential.tif", ma_potential)
        __save_np_as_tiff(tmp_dir+"vit_potential.tif", vit_potential)

        binary_mask = self.get_binary_mask()
        Image.fromarray(binary_mask).save(tmp_dir+"binary_mask.png")

        # separate train and test masks
        # poly_coords = [(10980, 3133), (7153, 3133), (5578, 5777), (5073, 6074), (5133, 7024), (5043, 7886), (5281, 8153), (5192, 8599), (5430, 8688), (5608, 9252), (10980, 9252)]
        poly_coords = [(0, 854), (3226, 854), (3039, 2077), (2086, 3382), (2127, 4190), (1982, 4874), (3329, 7651), (0, 7651)]
        poly_rows = [pt[1] for pt in poly_coords]
        poly_cols = [pt[0] for pt in poly_coords]
        poly_mask = np.zeros(binary_mask.shape, dtype=bool)
        rr, cc = polygon(poly_rows, poly_cols, shape=binary_mask.shape)
        poly_mask[rr, cc] = True
        
        selection = (binary_mask == True) & poly_mask
        Image.fromarray(selection).save(tmp_dir+"test_mask.png")
       
        selection = (binary_mask == True) & (poly_mask == False)
        Image.fromarray(selection).save(tmp_dir+"train_mask.png")

        with zipfile.ZipFile(dataset_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if file.split(".")[-1] != "xml":
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmp_dir)
                        zipf.write(file_path, arcname)

        shutil.rmtree(tmp_dir)