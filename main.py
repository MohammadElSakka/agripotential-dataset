from rasterio import rasterio
import warnings
import sys
import requests
import zipfile
import os 

from scripts.dataset import Dataset
from scripts.visualize import visualize_weather, visualize_sentinel2
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Calling float on a single element Series.*")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


if len(sys.argv) > 1:
    if sys.argv[1] == "dataset":
        response = requests.get("https://cloud.irit.fr/s/t0fnKon6YRq57UZ/download")
        zip_file_path = "data/dataset.zip"
        extract_path = "data/dataset/"
        if response.status_code == 200:
            with open(zip_file_path, "wb") as f:
                f.write(response.content)
            try:    
                with zipfile.ZipFile(zip_file_path, 'r') as zip:
                    os.mkdir(extract_path)
                    zip.extractall(extract_path)
            except Exception as e:
                print(f"Failed to extract the zip file: {e}")

            try:
                os.remove(zip_file_path) 
            except Exception as e:
                print(f"Error deleting the zip file: {e}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

dataset = Dataset(download=False)
# dataset.export_dataset()
