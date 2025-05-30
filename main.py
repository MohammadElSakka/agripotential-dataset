from rasterio import rasterio
import warnings
import sys
import requests
import zipfile
import os 
import argparse
import io

from scripts.dataset import Dataset
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Calling float on a single element Series.*")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser("python3 main.py [OPTIONS]", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--download_raw_data", 
                            action="store_true", 
                            help="Download the raw data."
                        ) 
    
    parser.add_argument("--download_dataset", 
                            action="store_true", 
                            help="Download the final dataset."
                        ) 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    if args.download_dataset:
        print("Downloading final dataset. There is nothing else to do.", flush=True)
        with open("download_link.txt", "r") as file:
            link = file.readline().strip()
        response = requests.get(link)
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
    
    if args.download_raw_data:
        print("Downloading raw data.")
        response = requests.get("https://zenodo.org/records/15551802/files/raw_data.zip?download=1")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                zip.extractall(f"./data")
    
    if args.generate_dataset:
        dataset = Dataset(ind_conf=3, iddiz=2.5 , icucs=3)
        dataset.export_dataset()