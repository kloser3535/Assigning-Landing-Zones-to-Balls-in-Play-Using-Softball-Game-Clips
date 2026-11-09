import cv2
import pandas as pd
import numpy as np
import os

from camera_motion import (
    track_features,
    estimate_camera_transform,
    summarize_motion
)

VIDEO_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos"
CAMERA_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\camera_motion_csv_files"

def process_video(video_name, video_path):
    
    camera_csv = os.path.join(
        CAMERA_FOLDER,
        video_name + ".csv"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Could not open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []

    previous_arrow_length = None

    for frame_number in range(total_frames - 1):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, previous_frame = cap.read()

        success, current_frame = cap.read()

        if not success:
            break
        
        old_points, new_points = track_features(
            previous_frame,
            current_frame
        )

        dx = np.mean(
            new_points[:, 0] - old_points[:, 0]
        )

        dy = np.mean(
            new_points[:, 1] - old_points[:, 1]
        )

        arrow_length = np.sqrt(
            dx**2 +
            dy**2
        )

        if previous_arrow_length is None:

            arrow_change = 0

        else:

            arrow_change = abs(
                arrow_length -
                previous_arrow_length
            )

        previous_arrow_length = arrow_length

        transform = estimate_camera_transform(
            old_points,
            new_points
        )

        motion = summarize_motion(
            transform,
            old_points
        )

        pan_pixels = np.sqrt(
            motion["pan_x"]**2 +
            motion["pan_y"]**2
        )

        results.append({
            "frame": frame_number,
            "pan_x": motion["pan_x"],
            "pan_y": motion["pan_y"],
            "zoom": motion["scale"],
            "rotation": motion["rotation"],
            "arrow_dx": dx,
            "arrow_dy": dy,
            "arrow_length": arrow_length,
            "arrow_change": arrow_change,
            "pan_pixels": pan_pixels,
            "tracked_features": motion["tracked_features"]
        })
        
    motion_df = pd.DataFrame(results)

    motion_df.to_csv(
        camera_csv,
        index=False
    )

    print(f"Saved camera motion CSV → {camera_csv}")

    cap.release()

if __name__ == "__main__":

    process_video("FOR_INF_H")
