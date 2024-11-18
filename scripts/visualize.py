from PIL import Image
import numpy as np
import pandas as pd
from rasterio import rasterio

from tqdm import tqdm

import matplotlib as mpl
from matplotlib import patheffects, cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.patches as patches

import skimage.draw
from scripts.utils import compute_boundaries, gps_to_pixel

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
    plt.close()
    return

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
    plt.savefig(output_file_path, dpi=800, bbox_inches='tight')
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
    return 

def brighten(data: np.ndarray) -> np.ndarray:
    alpha=0.08
    beta=0
    return np.clip(alpha*data+beta, 0,255)

def normalize(data: np.ndarray) -> np.ndarray:
    data_min, data_max = (data.min(), data.max())
    return ((data-data_min)/((data_max - data_min)))

def visualize_sentinel2(idx: int):
    output_path = "media/"
    dataset_path = "data/dataset/"
    sentinel2_paths = [dataset_path+f"sentinel2_2019_{idx}.tif" for i in range(1, 13)]
    
    sentinel2_image_path = sentinel2_paths[idx]
    sentinel2_image = rasterio.open(sentinel2_image_path)

    blue = normalize(brighten(sentinel2_image.read(1)))
    green = normalize(brighten(sentinel2_image.read(2)))
    red = normalize(brighten(sentinel2_image.read(3)))
    nir = normalize(brighten(sentinel2_image.read(4)))

    color_image = np.dstack([red, green, blue])
    false_color_image = np.dstack([nir, red, green])

    save_plot(output_path+f"sentinel2_2019_{idx+1}.jpg", f"Color image {idx+1}/2019", False, color_image)
    save_plot(output_path+f"false_sentinel2_2019_{idx+1}.jpg", f"False color image {idx+1}/2019", False, false_color_image)

def visualize():
    output_path = "media/"
    dataset_path = "data/dataset/"

    binary_mask_path = dataset_path+"binary_mask.png"
    elevation_data_path = dataset_path+"elevation.tif"
    
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
    
    visualise_potential(global_potential_path, output_path+f"global_potential.jpg", "Global potential", binary_mask)
    visualise_potential(gc_potential_path, output_path+f"gc_potential.jpg", "Grandes cultures", binary_mask)
    visualise_potential(ma_potential_path, output_path+f"ma_potential.jpg", "Maraîchage", binary_mask)
    visualise_potential(vit_potential_path, output_path+f"vit_potential.jpg", "Viticulture", binary_mask)
    plot_color_bar(output_path+"categorical_colorbar.jpg")

    # months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # weather_df = pd.read_csv(weather_data_path)
    # weather_df['months'] = [months[int(date[4:].split('_')[0])-1]+date[-2:] for date in weather_df['Date']]
    # save_graph(weather_df['months'], weather_df['Temperature'], 'Temperature (°C)', 'Temperature over time (2019)', output_path+"temperature.jpg")
    # save_graph(weather_df['months'], weather_df['Evapotranspiration'], 'Evapotranspiration (mm)', 'Evapotranspiration over time (2019)', output_path+"evapotranspiration.jpg")
    # save_graph(weather_df['months'], weather_df['Precipitation'], 'Precipitation (mm)', 'Precipitation over time (2019)', output_path+"precipitation.jpg")
    # save_graph(weather_df['months'], weather_df['Insolation'], 'Insolation (min)', 'Insolation over time (2019)', output_path+"insolation.jpg")


def calculate_distance(x1, y1, x2, y2: int) -> float:
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def visualize_weather(data: np.ndarray, dataset: any, df:pd.DataFrame, criteria: str = None) -> None:
    assert criteria in ["Temperature", "Max Temperature", "Min Temperature", "Precipitation", None], "invalid criteria"
    meta = dataset.get_meta()
    img = np.transpose(rasterio.open("data/dataset/sentinel2_2019_6.tif").read()[:3], (1, 2, 0))
    img = img[..., ::-1]

    up, down, left, right = 3321, 10979, 0, 9401 # compute_boundaries
    for i in range(3):
        img[:,:,i]= normalize(brighten(img[:,:,i]))*255
    img = img[up:down+1, left:right+1, :]
    data = np.array(data)
    data = data[up:down+1, left:right+1]

    postes = pd.unique(df["ID"]).tolist()
    if criteria == None:
        colormap = cm.tab20
        norm = Normalize(vmin=0, vmax=len(postes)-1)
        colors = [colormap(norm(i)) for i in range(len(postes))]
    else:
        norm = Normalize(vmin=df[criteria].min(), vmax=df[criteria].max())
        values = {}
        for station_id in postes:
            values[station_id] = df[df["ID"] == station_id][criteria].mean()
        if "Temperature" in criteria:
            colormap = cm.jet
            suffix = "°C"
            colors = [colormap(norm(i)) for i in range(-4, 37)]
        else:
            suffix = "mm"
            colormap = cm.Blues
            colors = [colormap(norm(i)) for i in range(150, 360)]

    for i in range(3):
        img[:, :, i] = np.where(data == 0, img[:, :, i] * 0.5, img[:, :, i])

    displayed_stations = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            station_id = int(data[i, j])

            if station_id > 1:
                if np.all(img[i,j] != [255, 0, 0]):
                    if criteria != None:
                        value = values[station_id]
                        color = colors[int(value)]
                        img[i, j] = 0.5*img[i,j] + (125*color[0], 125*color[1], 125*color[2])  
                    else:
                        color = colors[postes.index(station_id)]
                        img[i, j] = 0.5*img[i,j] + (125*color[0], 125*color[1], 125*color[2])

                if station_id not in displayed_stations:
                    poste =  df[df["ID"]==station_id][["ID", "LAT", "LON", "NOM_USUEL"]].drop_duplicates()
                    displayed_stations += [station_id]
                    gps = {
                            "LAT": poste["LAT"],
                            "LON": poste["LON"]
                        }
                    x, y = gps_to_pixel(gps, meta)
                    if left<=x<=right and up<=y<=down:
                        x = x - left
                        y = y - up

                    if criteria != None:
                        text_obj = plt.text(x+75, y+100, f"{poste['NOM_USUEL'].iloc[0]} ({int(value)} {suffix})", fontsize=5, color="white", fontweight='bold')
                    else:
                        text_obj = plt.text(x+75, y+100, f"{poste['NOM_USUEL'].iloc[0]}", fontsize=5, color="white", fontweight='bold')
                    text_obj.set_path_effects([
                        patheffects.withStroke(linewidth=1, foreground="black"), 
                        patheffects.Normal()
                    ])
                    rr, cc = skimage.draw.disk((y, x), radius=50)
                    rr = np.clip(rr, 0, img.shape[0] - 1)
                    cc = np.clip(cc, 0, img.shape[1] - 1)
                    img[rr,cc] = (255, 0, 0)

    
    plt.imshow(img)
    plt.axis('off') 
    plt.tight_layout()
    if criteria == None:
        plt.savefig("media/stations.png", bbox_inches='tight', pad_inches=0, dpi=800)
    else:
        plt.savefig(f"media/{criteria.lower()}.png", bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

# for i in tqdm(range(img.shape[0])):
#     for j in range(img.shape[1]):
#         if binary_mask[i,j]:
#             min_distance = np.inf
#             min_location = None
#             for location in locations.keys():
#                 x = locations[location][0]
#                 y = locations[location][1]
#                 if abs(j-x) < 2000 and abs(i-y)<2000:
#                     distance = calculate_distance(j, i, x, y)
#                     if distance < min_distance:
#                         min_distance = distance
#                         min_location = location
#             color = colormap(norm(coordinates["NOM_USUEL"].tolist().index(min_location)))
#             img[i,j] = 0.5*img[i,j] + (0.5*color[0]*255, 0.5*color[1]*255, 0.5*color[2]*255)

# meters_per_pixel = 10 
# real_world_distance_in_meters = 100
# scale_bar_length_in_pixels = real_world_distance_in_meters / meters_per_pixel
# x_start = 50
# y_start = img.shape[0] - 50
# scale_bar = patches.Rectangle((x_start, y_start), scale_bar_length_in_pixels, 10, linewidth=2, edgecolor='white', facecolor='white', alpha=0.7)
# plt.gca().add_patch(scale_bar)
# plt.text(x_start + scale_bar_length_in_pixels + 10, y_start + 5, f'{real_world_distance_in_meters} m', fontsize=12, color='white', verticalalignment='center')

