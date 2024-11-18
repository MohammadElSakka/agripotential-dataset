from scripts.dataset import Dataset
from scripts.visualize import visualize, visualize_sentinel2, normalize, brighten, visualize_weather
from scripts.utils import gps_to_pixel

from PIL import Image
import numpy as np
import pandas as pd
from rasterio import rasterio

import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Calling float on a single element Series.*")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

dataset = Dataset(download=False)
# dataset.export_dataset()
# # 
data = np.load("data/.tmp/pixels_to_stations.npy")[0]
weather = pd.read_csv("data/.tmp/weather_data.csv")
print("Stations")
visualize_weather(data, dataset, weather, None)
print("Temperature")
visualize_weather(data, dataset, weather, "Temperature")
print("Max Temperature")
visualize_weather(data, dataset, weather, "Max Temperature")
print("Min Temperature")
visualize_weather(data, dataset, weather, "Min Temperature")
print("Precipitation")
visualize_weather(data, dataset, weather, "Precipitation")
# visualize()
# visualize_sentinel2(1)