import zipfile
import requests
import io
import os
import shutil

class Dataset:
    def __init__(self) -> None:
        pass

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