import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define the path to the directory containing the CSV files
data_directory = '/home/olga/Desktop/TER/Weather/Montpellier' # Replace with your actual path

# Use glob to get all CSV files in the directory
csv_files = glob.glob(os.path.join(data_directory, '*.csv'))

# Check if any files were found
if not csv_files:
    print(f"No CSV files found in the directory '{data_directory}'. Please check the path.")
    exit()

print(f"Found {len(csv_files)} CSV files in '{data_directory}'.")

# Create an empty dictionary to hold DataFrames
months_data = {}

# Load each CSV file into a DataFrame and store it in the dictionary
for file in csv_files:
    try:
        Month_num = os.path.splitext(os.path.basename(file))[0]
        # Specify columns by their indices (B=1, J=2)
        df = pd.read_csv(file, usecols=[1, 2])
        df.columns = ['DATE', 'RR']  # Rename columns for clarity
        df = df.dropna(subset=['DATE', 'RR'])  # Drop rows where the DATE or RR is missing
        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')  # Parse dates
        months_data[Month_num] = df
        print(f"Loaded data for {Month_num}:")
        print(df.head())  # Print first few rows to verify data
    except Exception as e:
        print(f"Error loading data from {file}: {e}")

# Check if any data was loaded
if not months_data:
    print("No data was loaded. Please check the CSV files.")
    exit()

# Calculate the average rainfall for each month
average_rainfall_per_month = {}

for Month_num, df in months_data.items():
    average_rainfall = df['RR'].mean()
    average_rainfall_per_month[Month_num] = average_rainfall
    print(f"The average rainfall for {Month_num} is {average_rainfall:.2f} mm")

# Convert the dictionary to a DataFrame for plotting
average_rainfall_df = pd.DataFrame(list(average_rainfall_per_month.items()), columns=['Month', 'Average Rainfall'])

# Sort by month (assuming filenames are sortable in chronological order)
average_rainfall_df = average_rainfall_df.sort_values(by='Month')

# Plotting the average rainfall for each month
plt.figure(figsize=(12, 8))
plt.plot(average_rainfall_df['Month'], average_rainfall_df['Average Rainfall'], marker='o')
plt.title('Average Rainfall for Each Month in 2019')
plt.xlabel('Month')
plt.ylabel('Average Rainfall (mm)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print("Plot created for average rainfall per month.")
