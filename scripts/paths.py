import os

def get_elevation_data_path() -> str:
    return os.path.abspath("./data/elevation_data/raw_elevation_data_10m.tif")

def get_geojson_path() -> str:
    return os.path.abspath("./data/dataset.geojson")
    
def get_sentinel_data_path() -> dict:
    paths = {}
    years = os.listdir("data/sentinel2_bands")
    for year in years:
        y = int(year)
        paths[y] = {}
        months = os.listdir(f"data/sentinel2_bands/{year}")
        for month in months:
            m = int(month.split("_")[0])
            paths[y][m] = {}
            bands = os.listdir(f"data/sentinel2_bands/{year}/{month}/") 
            for band in bands:
                band_name = band.split("_")[2]
                paths[y][m][band_name] = os.path.abspath(f"./data/sentinel2_bands/{year}/{month}/{band}")
    return paths

def get_weather_path() -> str:  
    return os.path.abspath("./data/Weather_data/")

sentinel_data_path = get_sentinel_data_path()
elevation_path = get_elevation_data_path()
geojson_path = get_geojson_path()
