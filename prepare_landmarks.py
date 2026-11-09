# -----------------------------
# Input and output files
# -----------------------------
import pandas as pd
import os

INPUT_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\combined_csv_files"
OUTPUT_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\combined_csv_files_best_landmarks"

##df = pd.read_csv(INPUT_CSV)
##
##print(df["class"].value_counts())
##
##print("\nMissing confidence values:")
##print(df.groupby("class")["confidence"].apply(lambda x: x.isna().sum()))
##
##print("\nTotal rows:")
##print(df.groupby("class").size())

csv_files = sorted(
    [
        f
        for f in os.listdir(INPUT_CSV)
        if f.endswith(".csv")
    ]
)

print()
print(f"Found {len(csv_files)} files")
print()

for csv in csv_files: # IMPORTANCE OF BUILDING PATHS AND THEN SAVING

    input_path = os.path.join(
        INPUT_CSV,
        csv
    )

    df = pd.read_csv(input_path)

    # Create the new columns
    df["best_detection"] = False
    df["usable_detection"] = False

    # Process each frame and class separately
    for (frame, cls), group in df.groupby(["frame", "class"]):

        # Ignore rows without a confidence value
        valid = group[group["confidence"].notna()]

        # Nothing to do if no valid detections
        if valid.empty:
            continue

        # Find the row with the highest confidence
        best_index = valid["confidence"].idxmax()

        # Mark it
        df.loc[best_index, "best_detection"] = True
        df.loc[best_index, "usable_detection"] = True

    output_path = os.path.join(
        OUTPUT_CSV,
        csv
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Finished {csv}")

print()
print("All files complete")
