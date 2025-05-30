# Agricultural Potential Dataset

Agricultural potential refers to the capacity of a specific area to produce crops. This potential is influenced by environmental factors such as soil quality and climate. This repository proposes a remote sensing multimodal multisource and multitemporal dataset built for supervised AI. This repository contains the code needed to construct the dataset from raw files.  

The raw files are available for download [here](https://zenodo.org/records/15551802):  
```
https://zenodo.org/records/15551802
```

After downloading the raw files archive, extract it into your preferred directory (e.g. raw_data/) and pass it to the program like this if you want to construct the dataset from scratch.  
**Be careful, generating the dataset requires a lot of memory resources**
```
python3 main.py <path/to/the/raw/files/directory/>  
```
This will produce a "dataset.h5" file, which is the AgriPotential dataset.  

If you want to skip this part, you are free to download the ready-to-use dataset from [this link](https://zenodo.org/records/15551830):  
```
https://zenodo.org/records/15551830
```
There you can find a dataset.zip that you contains the dataset.h5 file that you would construct using this code.  

Tutorials are provided to guide you at using the dataset structure in ```tutorials/```
Specifically, you can find: 

- a tutorial on how to access and manipulate the data: [Tutorial 0](tutorials/Tutorial%200/)
- a tutorial on making a PyTorch dataloader for multiple purposes: [Dataloader](tutorials/Dataloader/)

# Data specs:
| Property           | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| Dataset name       | AgriPotential                                                         |
| File format        | HDF5 (.h5)                                                            |
| File size          | 28.4 GB                                                               |
| Data volume        | 145.7 GB                                                              |
| Number of images   | 8890                                                                  |
| Timestamps         | 11                                                                    |
| Spectral channels  | 10                                                                    |
| Spatial resolution | 5 m/px                                                                |
| Dimensions         | 128x128 (0.41 km²)                                                    |
| Annotation level   | Pixel level                                                           |
| Crop types         | viticulture, market gardening, field crops                            |
| Potential classes  | Very low, low, average, high, very high                               |
| License            | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)      |
| Data link          | [Zenodo Record](https://zenodo.org/records/15551830)   


# Creating Python virtual environment

## On Linux 

```
python3 -m venv my_env
source my_env/bin/activate
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=my_env --display-name "Python (my_env)"
cat requirements.txt | xargs -n 1 pip install 
```


# Downloading data

## Raw data (Linux) 

```
wget -O raw_data.zip https://zenodo.org/records/15551802/files/raw_data.zip?download=1
unzip raw_data.zip -d raw_data
```

## The dataset (Linux) 
```
wget -O dataset.zip https://zenodo.org/records/15551830/files/dataset.zip?download=1
unzip dataset.zip
```