from rasterio import rasterio
import warnings
import sys
import requests
import zipfile
import os 
import argparse

from scripts.dataset import Dataset
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Calling float on a single element Series.*")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


if len(sys.argv) > 1:
    data_path = sys.argv[1] 
    dataset = Dataset(data_path=data_path, ind_conf=3, iddiz=2.5 , icucs=3)
    dataset.export_dataset()