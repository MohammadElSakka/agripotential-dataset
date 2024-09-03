#/bin/bash

if ls download* > /dev/null 2>&1; then
    rm download*
fi

# Sentinel-2
wget https://cloud.irit.fr/s/E22CahuMVTBhsbI/download
unzip download
rm download
mv sentinel2_bands data

# Altitude
wget https://cloud.irit.fr/s/HBeMWkfLvhkysrf/download
unzip download
rm download
mv 'Elevation data' data/elevation_data

# GeoJSON Dataset
wget https://cloud.irit.fr/s/3hEr67GQ2C4bWG9/download
mv download data/dataset.geojson

# Weather
wget https://cloud.irit.fr/s/TxHqTO5aDXpRMyz/download
unzip download
rm download
mv Weather_data data
