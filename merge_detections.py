# imports
import os
import pandas as pd
import numpy as np

# files
BALL_DETECTIONS_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\cleaned_ball_detection_csv_files"
LANDMARK_DETECTIONS_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\cleaned_landmark_detection_csv_files"
CAMERA_DETECTIONS_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\camera_motion_state_csv_files"

MASTER_DETECTIONS_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\master_detection_csv_files"

os.makedirs(MASTER_DETECTIONS_FOLDER, exist_ok=True)

def merge_detections_files(ball_file, landmark_file, camera_file, output_file):
    LANDMARKS = [
        "home_plate",
        "first_base",
        "second_base",
        "third_base",
        "pitching_rubber",
    ]

    ball_df = pd.read_csv(ball_file)
    landmark_df = pd.read_csv(landmark_file)
    landmark_df = landmark_df.rename(
        columns={
            "x_center": "center_x",
            "y_center": "center_y"
        }
    )
    camera_df = pd.read_csv(camera_file)

    master_df = ball_df.copy()

    master_df = master_df.rename(columns={
        "detected": "ball_detected",
        "track_id": "ball_track_id",
        "center_x": "ball_center_x",
        "center_y": "ball_center_y",
        "x1": "ball_x1",
        "y1": "ball_y1",
        "x2": "ball_x2",
        "y2": "ball_y2",
        "confidence": "ball_confidence",
        "dx": "ball_dx",
        "dy": "ball_dy",
        "speed": "ball_speed",
        "touch_score": "ball_touch_score",
        "speed_change": "ball_speed_change"
    })

    for landmark in LANDMARKS:
        landmark_rows = landmark_df[
            landmark_df["class"] == landmark
        ].copy()
        if landmark_rows.empty:

            landmark_rows = pd.DataFrame({
                "frame": master_df["frame"],
                f"{landmark}_center_x": np.nan,
                f"{landmark}_center_y": np.nan,
                f"{landmark}_width": np.nan,
                f"{landmark}_height": np.nan,
                f"{landmark}_confidence": np.nan
            })

        else:
            landmark_rows = landmark_rows.rename(columns={
                "center_x": f"{landmark}_center_x",
                "center_y": f"{landmark}_center_y",
                "width": f"{landmark}_width",
                "height": f"{landmark}_height",
                "confidence": f"{landmark}_confidence"
            })

            landmark_rows = landmark_rows[
                [
                    "frame",
                    f"{landmark}_center_x",
                    f"{landmark}_center_y",
                    f"{landmark}_width",
                    f"{landmark}_height",
                    f"{landmark}_confidence"
                ]
            ]
            
        master_df = master_df.merge(
            landmark_rows,
            on="frame",
            how="left"
        )

    master_df = master_df.merge(
        camera_df,
        on="frame",
        how="left"
    )

    master_df.to_csv(output_file, index=False)
                
    print(master_df.head())

    print(
        "Duplicate frames (want 0):",
        master_df["frame"].duplicated().sum()
    )

    print("Frames:", len(master_df))
    print("Unique frames (should be the same as Frames):", master_df["frame"].nunique())

    for landmark in [
        "home_plate",
        "first_base",
        "second_base",
        "third_base",
        "pitching_rubber"
    ]:

        print(
            landmark,
            master_df[f"{landmark}_center_x"].notna().sum()
        )

    print("Saved:", output_file)

if __name__ == "__main__":

    for filename in os.listdir(BALL_DETECTIONS_FOLDER):

        if not filename.endswith(".csv"):
            continue

        ball_file = os.path.join(
            BALL_DETECTIONS_FOLDER,
            filename
        )

        landmark_file = os.path.join(
            LANDMARK_DETECTIONS_FOLDER,
            filename
        )

        camera_file = os.path.join(
            CAMERA_DETECTIONS_FOLDER,
            filename
        )

        output_file = os.path.join(
            MASER_DETECTIONS_FOLDER,
            filename
        )

        merge_detections_files(
            ball_file,
            landmark-file,
            camera_file,
            output_file
        )

