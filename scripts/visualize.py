from PIL import Image
import numpy as np
import pandas as pd
from rasterio import rasterio

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

def save_categorical_plot(output_file_path: str, plot_title: str, data: any) -> None:
    colors = ["red", "orange", "yellow", "lightgreen", "green"]
    cmap = ListedColormap(colors)
    bounds = np.linspace(0, 5, 6)
    norm = BoundaryNorm(bounds, cmap.N)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.axis("off")
    plt.title(plot_title)
    plt.savefig(output_file_path, dpi=1600, bbox_inches='tight')
    plt.close()
    return 

def plot_color_bar(output_file_path: str) -> None:
    colors = ["red", "orange", "yellow", "lightgreen", "green"]
    labels = ["Very limited", "Limited", "Moderate", "High", "Very high"]
    cmap = ListedColormap(colors)
    bounds = np.linspace(0, 5, 6)
    norm = BoundaryNorm(bounds, cmap.N)
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    colorbar = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                                cmap=cmap, norm=norm)
    colorbar.set_ticks(np.arange(0.5, 5.5))
    colorbar.set_ticklabels(labels)
    colorbar.set_label("Agri Potentials")
    plt.savefig(output_file_path,  bbox_inches='tight')

def visualise_potential(potential_path: str, 
                        output_file: str,
                        potential_name: str, 
                        binary_mask: np.ndarray)-> None:
    
    potential = rasterio.open(potential_path).read()
    potential = np.transpose(potential, (1, 2, 0))    
    potential = np.argmax(potential, axis=-1)
    potential = np.where(binary_mask, potential, np.nan)
    save_categorical_plot(output_file, potential_name, potential)
    return

def save_plot(output_file_path: str, plot_title: str, display_colorbar: bool, data: any) -> None:
    plt.imshow(data)
    if display_colorbar:
        plt.colorbar()
    plt.axis("off")
    plt.title(plot_title)
    plt.savefig(output_file_path, dpi=1600, bbox_inches='tight')
    plt.close()
    return 

def save_graph(x: any, y: any, ylabel: str, plot_title: str, output_file_path: str)->None:
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.xlabel('Time (Ten-day)')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()

def brighten(data: np.ndarray) -> np.ndarray:
    alpha=0.08
    beta=0
    return np.clip(alpha*data+beta, 0,255)

def normalize(data: np.ndarray) -> np.ndarray:
    data_min, data_max = (data.min(), data.max())
    return ((data-data_min)/((data_max - data_min)))

def visualize():
    output_path = "media/"
    dataset_path = "data/dataset/"

    binary_mask_path = dataset_path+"binary_mask.png"
    elevation_data_path = dataset_path+"elevation.tif"
    sentinel2_paths = [dataset_path+f"sentinel2_2019_{i}.tif" for i in range(1, 13)]
    
    global_potential_path = dataset_path+"global_potential.tif"
    gc_potential_path = dataset_path+"gc_potential.tif"
    ma_potential_path = dataset_path+"ma_potential.tif"
    vit_potential_path = dataset_path+"vit_potential.tif"
    
    weather_data_path = dataset_path+"weather_data.csv"
    
    binary_mask = np.array(Image.open(binary_mask_path))
    save_plot(output_path+"binary_mask.png", "Binary Mask", False, binary_mask)

    elevation = rasterio.open(elevation_data_path).read(1)
    elevation = np.where(binary_mask, elevation, np.nan)
    save_plot(output_path+"elevation.jpg", "Elevation (m)", True, elevation)
    
    for i in range(len(sentinel2_paths)):
        sentinel2_image_path = sentinel2_paths[i]
        sentinel2_image = rasterio.open(sentinel2_image_path)

        blue = normalize(brighten(sentinel2_image.read(1)))
        green = normalize(brighten(sentinel2_image.read(2)))
        red = normalize(brighten(sentinel2_image.read(3)))
        nir = normalize(brighten(sentinel2_image.read(4)))

        color_image = np.dstack([red, green, blue])
        false_color_image = np.dstack([nir, red, green])

        save_plot(output_path+f"sentinel2_2019_{i+1}.jpg", f"Color image {i+1}/2019", False, color_image)
        save_plot(output_path+f"false_sentinel2_2019_{i+1}.jpg", f"False color image {i+1}/2019", False, false_color_image)
    
    visualise_potential(global_potential_path, output_path+f"global_potential.jpg", "Global potential", binary_mask)
    visualise_potential(gc_potential_path, output_path+f"gc_potential.jpg", "Grandes cultures", binary_mask)
    visualise_potential(ma_potential_path, output_path+f"ma_potential.jpg", "Maraîchage", binary_mask)
    visualise_potential(vit_potential_path, output_path+f"vit_potential.jpg", "Viticulture", binary_mask)
    plot_color_bar(output_path+"categorical_colorbar.jpg")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    weather_df = pd.read_csv(weather_data_path)
    weather_df['months'] = [months[int(date[4:].split('_')[0])-1]+date[-2:] for date in weather_df['Date']]
    save_graph(weather_df['months'], weather_df['Temperature'], 'Temperature (°C)', 'Temperature over time (2019)', output_path+"temperature.jpg")
    save_graph(weather_df['months'], weather_df['Evapotranspiration'], 'Evapotranspiration (mm)', 'Evapotranspiration over time (2019)', output_path+"evapotranspiration.jpg")
    save_graph(weather_df['months'], weather_df['Precipitation'], 'Precipitation (mm)', 'Precipitation over time (2019)', output_path+"precipitation.jpg")
    save_graph(weather_df['months'], weather_df['Insolation'], 'Insolation (min)', 'Insolation over time (2019)', output_path+"insolation.jpg")
