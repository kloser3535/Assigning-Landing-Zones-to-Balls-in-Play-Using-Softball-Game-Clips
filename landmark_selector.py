import pandas as pd

CSV_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\training_run_2_video_application_results\version_2_CSVs\DAY_INF_O.csv"

df = pd.read_csv(CSV_FILE)

frames = df.groupby("frame")

frame_number = int(input("Type desired frame: "))

frame_df = frames.get_group(frame_number)

home = frame_df[
    frame_df["class"] == "home_plate"
]

first = frame_df[
    frame_df["class"] == "first_base"
]

second = frame_df[
    frame_df["class"] == "second_base"
]

third = frame_df[
    frame_df["class"] == "third_base"
]

rubber = frame_df[
    frame_df["class"] == "pitching_rubber"
]

print()

print("HOME PLATE")

print(home[["frame","confidence", "x_center", "y_center", "width", "height"]].round(2))

print()

print("FIRST BASE")

print(first[["frame","confidence", "x_center", "y_center", "width", "height"]].round(2))

print()

print("SECOND BASE")

print(second[["frame","confidence", "x_center", "y_center", "width", "height"]].round(2))

print()

print("THIRD BASE")

print(third[["frame","confidence", "x_center", "y_center", "width", "height"]].round(2))

print()

print("PITCHING RUBBER")

print(rubber[["frame","confidence", "x_center", "y_center", "width", "height"]].round(2))
