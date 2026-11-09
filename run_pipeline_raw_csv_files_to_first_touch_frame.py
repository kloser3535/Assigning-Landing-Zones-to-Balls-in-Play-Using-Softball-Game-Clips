# this sciprt organize the 3 scripts that help us determine when first touch by a defender occurs
# those three scripts are: timeline csv, camear motion csv, first touch scorer

from generate_timeline_csv_from_trajectory_csv import process_video as build_timeline

from batch_camera_motion import process_video as build_camera_motion

from first_touch_scorer_v5 import process_video as score_first_touch

VIDEO_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos"

import os

video_files = sorted(
    [
        f
        for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(".mp4")
    ]
)

for video in video_files:

    video_name = os.path.splitext(video)[0]
    try:
        print("--------------------------------")
        print(video_name)
        print("--------------------------------")

        build_timeline(video_name)
        print("\nTimeline complete.")

        build_camera_motion(video_name)
        print("\nCamera Motion complete.")

        score_first_touch(video_name)
        print("\nFirst Touch Score complete.")
    except Exception as e:
        print(f"{video_name} failed.")
        print(e)
