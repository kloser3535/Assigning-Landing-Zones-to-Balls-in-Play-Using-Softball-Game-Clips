import os
import pandas as pd

# Folder containing the CSV files
CSV_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\trajectory_csv_files"

# Loop through all CSV files in the folder
for filename in os.listdir(CSV_FOLDER):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(CSV_FOLDER, filename)

        # Read the CSV
        df = pd.read_csv(file_path)

        # Add a new column with the value "ball" for every record
        df["class"] = "ball"

        # Save the updated CSV (overwrites the original)
        df.to_csv(file_path, index=False)

        print(f"Updated: {filename}")

print("Done! All CSV files have been updated.")
