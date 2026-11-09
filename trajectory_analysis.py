import pandas as pd
import numpy as np


def build_ball_timeline(df):
    """
    Build one ball observation per frame.

    Returns
    -------
    DataFrame
        One row per frame.
    """

    timeline = []

    for frame, group in df.groupby("frame"):
        group = group[group["detected"] == True]

        if group.empty:

            timeline.append({
                "frame": frame,
                "detected": False,
                "x": np.nan,
                "y": np.nan,
                "confidence": np.nan,
                "track_id": np.nan
            })

            continue

        best = group.loc[
            group["confidence"].idxmax()
        ]

        timeline.append({

            "frame": frame,

            "detected": True,

            "x": best["center_x"],

            "y": best["center_y"],

            "confidence": best["confidence"],

            "track_id": best["track_id"]

        })


    return pd.DataFrame(timeline)

def compute_ball_motion(timeline):
    """
    Compute frame-to-frame ball motion.
    """

    timeline = timeline.copy()

    timeline["dx"] = timeline["center_x"].diff()

    timeline["dy"] = timeline["center_y"].diff()

    timeline["speed"] = np.sqrt(
        timeline["dx"]**2 +
        timeline["dy"]**2
    )

    return timeline

def score_first_touch(timeline):
    """
    Compute a first-touch likelihood score for every frame.
    """

    timeline = timeline.copy()

    timeline["touch_score"] = 0.0

    timeline["touch_score"] += timeline["confidence"].fillna(0) # but do I actually think that highest confidence detections are more likely to be the ball than lower confidence detections

    timeline["speed_change"] = timeline["speed"].diff().abs()

    timeline["touch_score"] += timeline["speed_change"].fillna(0)

    timeline["touch_score"] = (
        timeline["touch_score"]
        .rolling(
            window=3,
            center=True,
            min_periods=1
        )
        .mean()
    ) # this prevent the model from selecting a frame where the ball is noisy (jumping all over the frame...
    
    return timeline

def detect_first_touch(timeline):
    """
    Return the most likely first-touch frame.
    """
    valid = timeline[
        (timeline["detected"] == True) &
        timeline["touch_score"].notna() &
        timeline["x"].notna() &
        timeline["y"].notna()
    ]

    if valid.empty:
        return None

    top5 = valid.sort_values(
        by="touch_score",
        ascending=False
    ).head(5)

    return top5    
