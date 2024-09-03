import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the directory containing the CSV files
data_directory = '/home/olga/Desktop/TER/Weather/Montpellier'  # Replace with your actual path

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
        month_num = os.path.splitext(os.path.basename(file))[0]
        # Specify columns by their indices (B=1, I=8)
        df = pd.read_csv(file, usecols=[1, 8])
        df.columns = ['DATE', 'TM']  # Rename columns for clarity
        df = df.dropna(subset=['DATE', 'TM'])  # Drop rows where the DATE or TM is missing
        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')  # Parse dates
        df = df.sort_values(by='DATE')  # Sort by date to ensure linear plot
        months_data[month_num] = df
        print(f"Loaded data for {month_num}:")
        print(df.head())  # Print first few rows to verify data
    except Exception as e:
        print(f"Error loading data from {file}: {e}")

# Check if any data was loaded
if not months_data:
    print("No data was loaded. Please check the CSV files.")
    exit()

# Combine all data into a single DataFrame for box plot and calculate average temperature
combined_df = pd.DataFrame()

for month_num, df in months_data.items():
    average_temp = df['TM'].mean()
    print(f"Average temperature for {month_num}: {average_temp:.2f}")
    df['Month'] = month_num  # Add a column for the Month number
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Plotting box plot for temperature data from all months
plt.figure(figsize=(12, 8))
combined_df.boxplot(column='TM', by='Month', grid=False, sym='')
plt.title('Temperature Data Distribution for Different Months')
plt.suptitle('')  # Suppress the automatic 'Boxplot grouped by Months' title
plt.xlabel('Month')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Box plot created for temperature data from all months.")
