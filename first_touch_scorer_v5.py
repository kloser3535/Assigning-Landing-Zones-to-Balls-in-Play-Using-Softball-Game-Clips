import pandas as pd
import numpy as np

# --------------------------------------------------
# Ball score weights
# --------------------------------------------------

BALL_CONFIDENCE_WEIGHT = 0.35
BALL_SPEED_WEIGHT = 0.65

# --------------------------------------------------
# Camera score weights
# --------------------------------------------------

CAMERA_PAN_WEIGHT = 0.75
CAMERA_STABILIZATION_WEIGHT = 0.25

# --------------------------------------------------
# Frame score weights
# --------------------------------------------------

START_END_WEIGHT = 0.30
STATE_WEIGHT = 0.70

# --------------------------------------------------
# First touch score weights
# --------------------------------------------------

FIRST_TOUCH_BALL_WEIGHT = 0.10
FIRST_TOUCH_CAMERA_WEIGHT = 0.45
FIRST_TOUCH_FRAME_WEIGHT = 0.45

# --------------------------------------------------
# utility functions
# --------------------------------------------------

def normalize(value, minimum, maximum):
    """
    Normalize a value to the range [0, 1].
    """

    if maximum == minimum:
        return 0.0

    score = (value - minimum) / (maximum - minimum)

    return np.clip(score, 0.0, 1.0)

def confidence_score(row):
    
    if row["ball_confidence"] > 0.25:
        confidence_score = 1
    else:
        confidence_score = 0
        
    return confidence_score

def speed_score(row, speed_min, speed_max):

    #
    # Normalize speed
    #

    normalized_speed = normalize(
        row["ball_speed"],
        speed_min,
        speed_max
    )

    #
    # Slower balls should receive higher scores.
    #

    return 1.0 - normalized_speed

def pan_score(row, pan_min, pan_max):

    normalized = normalize(
        row["arrow_length"],
        pan_min,
        pan_max
    )

    return 1.0 - normalized

def stabilization_score(
    row,
    arrow_change_average_min,
    arrow_change_average_max
):

    normalized = normalize(
        row["arrow_change_average"],
        arrow_change_average_min,
        arrow_change_average_max
    )

    return 1-normalized

def time_after_cut_score(row):

    frames = row["frames_after_cut"]

    if frames < 20:
        time_after_cut_score = 0.0
    elif frames > 200:
        time_after_cut_score = 0.0
    else:
        time_after_cut_score = 1.0 - ((frames - 20) / 190)
    return time_after_cut_score


def start_end_score(row, total_frames):
    if row["frame"] < 125:
        start_end_score = 0
    elif row["frame"] > total_frames - 125:
        start_end_score = 0
    else:
        start_end_score = 1
    return start_end_score

def stabilization_state_score(row):
    if row["camera_state"] != "Camera Stabilized":
        return 0
    frames = row["frames_into_stabilization"]
    if frames <= 12:
        return 1.0
    length = row["stabilization_length"]
    remaining = max(
        1,
        length - 12
    )
    progress = (
        frames - 12
    ) / remaining
    score = 1.0 - 0.5 * progress
    return max(
        0.5,
        score
    )

def large_pan_state_score(row):
    if pd.isna(row["frames_to_stabilization"]):
        frames = row["frames_into_large_pan"]
        if frames < 20:
            return 0.0
        elif frames <= 60:
            return 1.0
        elif frames > 260:
            return 0.0
        progress = (frames - 60) / 200
        return max(
            0,
            1 - progress
        )
    else:
        frames = row["frames_into_large_pan"]
        if frames < 20:
            return 0.0
        distance = row["frames_to_stabilization"]
        remaining = row["large_pan_length"] - 20
        remaining = max(
            1,
            row["large_pan_length"] - 20
        )
        progress = (frames - 20) / remaining
        return np.clip(
            progress,
            0,
            1
        )
        
def state_score(row):
    state = row["camera_state"]
    if state == "Pitch View":
        return 0
    if state == "Hard Cut":
        return 0
    if state == "Large Pan":
        return large_pan_state_score(row)
    if state == "Camera Stabilized":
        return stabilization_state_score(row)
    if state == "Throw Tracking":
        distance = row["frames_since_throw"]
        if pd.isna(distance):
            return 0.0
        if distance >= 30:
            return 0.0
        return 0.50 * (1 - distance / 30)
    if state == "Following Ball":
        if pd.isna(row["frames_to_stabilization"]):
            distance = row["frames_after_cut"]
            if pd.isna(distance):
                return 0            
            if distance < 20:
                return 0
            elif distance <= 80: # this is where I assume camera stablization is if the camera_state_detector did not pick up camera stabilization frames
                return 1
            elif distance <= 200:
                return max(
                    0,
                    1 - (distance-80)/120
                )
            else:
                return 0  
        else:
            distance = row["frames_to_stabilization"]
            if pd.isna(distance):
                return 0
            distance = max(0, distance)

            return max(
                0,
                1-distance/80
            )

def compute_reason(row):

    reasons = {

        "Slow Ball":
            FIRST_TOUCH_BALL_WEIGHT *
            BALL_SPEED_WEIGHT *
            row["speed_score"],

        "Ball Confidence":
            FIRST_TOUCH_BALL_WEIGHT *
            BALL_CONFIDENCE_WEIGHT *
            row["confidence_score"],

        "Small Camera Pan":
            FIRST_TOUCH_CAMERA_WEIGHT *
            CAMERA_PAN_WEIGHT *
            row["pan_score"],

        "Camera Stabilizing":
            FIRST_TOUCH_CAMERA_WEIGHT *
            CAMERA_STABILIZATION_WEIGHT *
            row["stabilization_score"],

        "During Proper State":
            FIRST_TOUCH_FRAME_WEIGHT *
            STATE_WEIGHT *
            row["state_score"],

        "Middle of Clip":
            FIRST_TOUCH_FRAME_WEIGHT *
            START_END_WEIGHT *
            row["start_end_score"]
    }

    return max(reasons, key=reasons.get)

    return max(reasons, key=reasons.get)

def process_video(video_name):
# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------

    BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

    MASTER_DETECTION_CSV = fr"{BASE_FOLDER}\master_detection_csv_files\{video_name}.csv"

    MASTER_CSV = fr"{BASE_FOLDER}\master_first_touch_score_csv_files\{video_name}.csv"

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------

    detections = pd.read_csv(MASTER_DETECTION_CSV)

    merged = detections.copy()

    duplicates = merged["frame"].duplicated().sum()
    print(
        "Duplicate frames (should be 0):",
        duplicates
    )
    
    merged["ball_detected"] = merged["ball_detected"].fillna(False)
    merged["ball_confidence"] = merged["ball_confidence"].fillna(0)
    
    # -------------------------------------------------------------------
    # Merge on frame number
    # -------------------------------------------------------------------

    merged["frames_after_cut"] = np.nan
    merged["frames_to_stabilization"] = np.nan
    merged["frames_since_throw"] = np.nan

    merged["confidence_score"] = 0.0
    merged["speed_score"] = 0.0
    merged["pan_score"] = 0.0
    merged["stabilization_score"] = 0.0
    merged["state_score"] = 0.0
    merged["ball_score"] = 0.0
    merged["camera_score"] = 0.0
    merged["frame_score"] = 0.0
    merged["first_touch_score"] = 0.0
    merged["reason"] = ""
    merged["frames_into_stabilization"] = np.nan
    merged["stabilization_length"] = np.nan
    merged["frames_into_large_pan"] = np.nan
    merged["large_pan_length"] = np.nan

    # -------------------------------------------------------------------
    # Compute scores
    # -------------------------------------------------------------------

    total_frames = merged["frame"].nunique()

    # camera state info
    
    stabilized = merged.loc[
        merged["camera_state"] == "Camera Stabilized"
    ]

    if len(stabilized) > 0:
        stabilization_start = stabilized.iloc[0]["frame"]
    else:
        stabilization_start = None

    if len(stabilized) > 0:

        stabilization_end = stabilized.iloc[-1]["frame"]

    else:

        stabilization_end = None

    if stabilization_start is not None:

        merged["frames_to_stabilization"] = (
            stabilization_start - merged["frame"]
        )

    if stabilization_start is not None:

        merged["frames_into_stabilization"] = (

            merged["frame"]

            -

            stabilization_start

        )

        merged["stabilization_length"] = (

            stabilization_end

            -

            stabilization_start

        )

    throw = merged[
        merged["camera_state"] == "Throw Tracking"
    ]

    if len(throw) > 0:
        throw_start = throw.iloc[0]["frame"]
    else:
        throw_start = None

    if throw_start is not None:

        merged["frames_since_throw"] = (
            merged["frame"] -
            throw_start
        )

    large_pan = merged[
        merged["camera_state"] == "Large Pan"
    ]

    if len(large_pan) > 0:

        large_pan_start = large_pan.iloc[0]["frame"]

        large_pan_end = large_pan.iloc[-1]["frame"]

    else:

        large_pan_start = None
        large_pan_end = None

    if large_pan_start is not None:

        merged["frames_into_large_pan"] = (
            merged["frame"] - large_pan_start
        )

        merged["large_pan_length"] = (
            large_pan_end - large_pan_start
        )

    # camera movement and ball info
    
    pan_min = merged["arrow_length"].min(skipna=True)

    pan_max = merged["arrow_length"].max(skipna=True)
      
    speed_min = merged["ball_speed"].min(skipna=True)

    speed_max = merged["ball_speed"].max(skipna=True)

    threshold = (
        merged["arrow_length"].mean()
        + 2 * merged["arrow_length"].std()
    )

    camera_cut_candidates = merged[
        merged["arrow_length"] > threshold
    ]

    if camera_cut_candidates.empty:

        camera_cut_frame = merged["frame"].iloc[0]

    else:

        camera_cut_frame = camera_cut_candidates.iloc[0]["frame"]

    merged["arrow_change_average"] = (
        merged["arrow_change"]
        .rolling(
            window=10,
            min_periods=1
        )
        .mean()
    )

    merged["frames_after_cut"] = (
        merged["frame"] -
        camera_cut_frame
    )

    arrow_change_average_min = merged["arrow_change_average"].min()

    arrow_change_average_max = merged["arrow_change_average"].max()

    merged["confidence_score"] = merged.apply(
        confidence_score,
        axis=1
    )

    merged["speed_score"] = merged.apply(
        lambda row: speed_score(
            row,
            speed_min,
            speed_max
        ),
        axis=1
    )

    merged.loc[
        merged["ball_detected"] == False,
        "speed_score"
    ] = 0

    merged["pan_score"] = merged.apply(
        lambda row: pan_score(
            row,
            pan_min,
            pan_max
        ),
        axis=1
    )

    merged["stabilization_score"] = merged.apply(
        lambda row: stabilization_score(
            row,
            arrow_change_average_min,
            arrow_change_average_max
        ),
        axis=1
    )

    merged["start_end_score"] = merged.apply(
        lambda row: start_end_score(
            row,
            total_frames
        ),
        axis=1
    )

    merged["state_score"] = merged.apply(
        state_score,
        axis=1
    )

    merged["ball_score"] = (
        BALL_CONFIDENCE_WEIGHT * merged["confidence_score"] +
        BALL_SPEED_WEIGHT * merged["speed_score"]
    )

    merged["camera_score"] = (
        CAMERA_PAN_WEIGHT * merged["pan_score"] +
        CAMERA_STABILIZATION_WEIGHT * merged["stabilization_score"]
    )
    merged["frame_score"] = (
        STATE_WEIGHT * merged["state_score"] +
        START_END_WEIGHT * merged["start_end_score"]
    )
    merged["first_touch_score"] = (
        (FIRST_TOUCH_BALL_WEIGHT * merged["ball_score"]) +
        (FIRST_TOUCH_CAMERA_WEIGHT * merged["camera_score"]) +
        (FIRST_TOUCH_FRAME_WEIGHT * merged["frame_score"])
    )
    merged["reason"] = merged.apply(
        compute_reason,
        axis=1
    )
    merged.loc[
        merged["frames_after_cut"] < 5,
        "first_touch_score"
    ] *= 0.05
    
    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------

    merged.to_csv(
        MASTER_CSV,
        index=False
    )

    top5 = merged.sort_values(
        "first_touch_score",
        ascending=False
    ).head(5)

    top5.insert(
        0,
        "Rank",
        range(1, len(top5)+1)
    )

    best = top5.iloc[0]

    summary_row = pd.DataFrame([{
        "video": video_name,
        "predicted_frame": best["frame"],
        "first_touch_score": best["first_touch_score"],
        "ball_score": best["ball_score"],
        "camera_score": best["camera_score"]
    }])

    import os

    SUMMARY_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\first_touch_summary.csv"

    if os.path.exists(SUMMARY_CSV) and os.path.getsize(SUMMARY_CSV) > 0:

        existing = pd.read_csv(SUMMARY_CSV)

        existing = existing[
            existing["video"] != video_name
        ]

        existing = pd.concat(
            [existing, summary_row],
            ignore_index=True
        )

        existing.to_csv(
            SUMMARY_CSV,
            index=False
        )

    else:

        summary_row.to_csv(
            SUMMARY_CSV,
            index=False
        )

    print("===================================")
    print("First Touch Scoring Complete")
    print("===================================")
    print(f"Frames scored: {merged['frame'].nunique()}")
    print(f"Top candidate: Frame {top5.iloc[0]['frame']}")
    print(f"Score: {top5.iloc[0]['first_touch_score']:.3f}")
    print(f"Saved master CSV to: {MASTER_CSV}")
    print(f"Appended summary row to: {SUMMARY_CSV}")

if __name__ == "__main__":
    process_video("URI_INF_H")
