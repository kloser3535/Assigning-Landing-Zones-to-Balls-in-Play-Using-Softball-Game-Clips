import pandas as pd

from trajectory_analysis import (
    build_ball_timeline,
    compute_ball_motion,
    score_first_touch,
    detect_first_touch
)

TRAJECTORY_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\trajectory_csv_files"
OUTPUT_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\timeline_csv_files"

import os

def process_video(video_name):

    trajectory_path = os.path.join(
        TRAJECTORY_FOLDER,
        video_name + ".csv"
    )

    timeline_csv = os.path.join(
        OUTPUT_FOLDER,
        video_name + ".csv"
    )


    trajectory_df = pd.read_csv(trajectory_path)

    timeline = build_ball_timeline(trajectory_df)

    timeline = compute_ball_motion(timeline)

    timeline = score_first_touch(timeline)

    print(
        timeline[
            [
                "frame",
                "speed",
                "speed_change",
                "touch_score"
            ]
        ].tail(10)
    )

    timeline.to_csv(
        timeline_csv,
        index=False
    )

    print(f"Saved timeline to {timeline_csv}")

    candidates = detect_first_touch(timeline)

    print()

    print("Top First-Touch Candidates")

    print("--------------------------------------")

    print(
        candidates[
            [
                "frame",
                "touch_score",
                "confidence",
                "speed"
            ]
        ]
    )
    
    return candidates

if __name__ == "__main__":

    process_video(
        "FOR_INF_H"
    )
