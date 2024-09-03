import zipfile
import requests
import io
import os
import shutil

class Dataset:
    def __init__(self) -> None:
        self.data_path = {}

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
        
        response = requests.get("https://cloud.irit.fr/s/p3EHRsTyfP9hZzU/download")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                os.mkdir("data")
                zip.extractall("data")
            return True
        else:
            print(f"Failed to download zip file [status code {response.status_code}].")
            return False
        
    def __get_paths__(self) -> None:
        self.data_path["Sentinel-2"] = {}
        years = os.listdir("data/sentinel2_bands")
        for year in years:
            y = int(year)
            self.data_path["Sentinel-2"][y] = {}
            months = os.listdir(f"data/sentinel2_bands/{year}")
            for month in months:
                m = int(month.split("_")[0])
                self.data_path["Sentinel-2"][y][m] = {}
                bands = os.listdir(f"data/sentinel2_bands/{year}/{month}/") 
                for band in bands:
                    band_name = band.split("_")[2]
                    self.data_path["Sentinel-2"][y][m][band_name] = os.path.abspath(f"./data/sentinel2_bands/{year}/{month}/{band}")

        self.data_path["labels"] = os.path.abspath("./data/dataset.geojson")
        self.data_path["elevation"] = os.path.abspath("./data/elevation_data/raw_elevation_data_10m.tif")
        self.data_path["weather"] = os.path.abspath("./data/weather_data/")