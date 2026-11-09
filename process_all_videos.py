import os
import pandas as pd

from analyze_batted_ball import analyze_batted_ball

BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

VIDEO_FOLDER = os.path.join(
    BASE_FOLDER,
    "testing_videos"
)

COMBINED_CSV_FOLDER = os.path.join(
    BASE_FOLDER,
    "combined_csv_files_best_landmarks"
)

CALIBRATION_FRAME_CSV = os.path.join(
    BASE_FOLDER,
    "calibration_frame_summary.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_FOLDER,
    "batted_ball_results.csv"
)

calibration_frames = pd.read_csv(
    CALIBRATION_FRAME_CSV
)

all_results = []

for video_file in os.listdir(VIDEO_FOLDER):

    if not video_file.endswith(".mp4"):
        continue

    video_name = os.path.splitext(video_file)[0]

    print()
    print("======================")
    print(video_name)
    print("======================")

    match = calibration_frames[
        calibration_frames["video_name"] == video_name
    ]

    if match.empty:
        print("No calibration frame found")
        continue

    calibration_frame = int(
        match.iloc[0]["best_frame"]
    )

    csv_path = os.path.join(
        COMBINED_CSV_FOLDER,
        video_name + ".csv"
    )

    if not os.path.exists(csv_path):
        print("Missing CSV")
        continue
    
    detections = pd.read_csv(csv_path)

    results = analyze_batted_ball(
        detections,
        calibration_frame
    )

    results["video"] = video_name

    all_results.append(results)

summary_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)
