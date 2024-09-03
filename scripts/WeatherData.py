import os
import glob
import pandas as pd
from scripts.paths import get_weather_path

# Define the path to the directory containing the CSV files
data_directory = get_weather_path()

# Use glob to get all CSV files in the directory
csv_files = glob.glob(os.path.join(data_directory, '*.csv'))

# Check if any files were found
if not csv_files:
    print(f"No CSV files found in the directory '{data_directory}'. Please check the path.")
    exit()

print(f"Found {len(csv_files)} CSV files in '{data_directory}'.")

# Dictionaries to hold average temperatures and average rainfall per month
average_temperatures = {}
average_rainfall_per_month = {}

# Process each CSV file to calculate average temperature and rainfall
for file in csv_files:
    try:
        month_num = os.path.splitext(os.path.basename(file))[0]
        
        # Read temperature data (columns B=1, I=8)
        df_temp = pd.read_csv(file, usecols=[1, 8])
        df_temp.columns = ['DATE', 'TM']  # Rename columns for clarity
        df_temp = df_temp.dropna(subset=['DATE', 'TM'])  # Drop rows where the DATE or TM is missing
        df_temp['DATE'] = pd.to_datetime(df_temp['DATE'], format='%Y%m%d')  # Parse dates
        average_temp = round(df_temp['TM'].mean(), 2)  # Round to 2 decimal places
        average_temperatures[month_num] = average_temp
        
        # Read rainfall data (columns B=1, J=2)
        df_rain = pd.read_csv(file, usecols=[1, 2])
        df_rain.columns = ['DATE', 'RR']  # Rename columns for clarity
        df_rain = df_rain.dropna(subset=['DATE', 'RR'])  # Drop rows where the DATE or RR is missing
        df_rain['DATE'] = pd.to_datetime(df_rain['DATE'], format='%Y%m%d')  # Parse dates
        average_rainfall = round(df_rain['RR'].mean(), 2)  # Round to 2 decimal places
        average_rainfall_per_month[month_num] = average_rainfall
        
    except Exception as e:
        print(f"Error processing data from {file}: {e}")

# Check if any data was processed
if not average_temperatures or not average_rainfall_per_month:
    print("No data was processed. Please check the CSV files.")
    exit()

# Sort the keys to ensure data is in order from 01 to 12
sorted_months = sorted(average_temperatures.keys())

# Create arrays for temperatures and rainfall with values rounded to 2 decimal places
temperature_array = [average_temperatures[month] for month in sorted_months]
rainfall_array = [average_rainfall_per_month[month] for month in sorted_months]


