import zipfile
import requests
import io

class Dataset:
    def __init__(self) -> None:
        pass

    def __download__(self) -> bool:
        response = requests.get("https://cloud.irit.fr/s/p3EHRsTyfP9hZzU/download")
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
                zip.extractall(".")
            return True
        else:
            print(f"Failed to download zip file. [status code {response.status_code}]")
            return False