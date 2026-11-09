import pandas as pd
import numpy as np
import os

#Instead...
FEATURE_DROP_THRESHOLD = 280
FEATURE_STABLE_THRESHOLD = 294
FEATURE_PAN_THRESHOLD = 290
##############
PITCH_ARROW_PERCENTILE = 0.20
#Instead ...
STABLE_ARROW_PERCENTILE = 0.40
##############
BIG_ARROW_PERCENTILE = 0.75
THROW_ARROW_PERCENTILE = 0.55
WINDOW = 10
        
CAMERA_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\camera_motion_csv_files"
OUTPUT_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\camera_motion_state_csv_files"

def process_camera_csv(input_csv, output_csv):
    
    motion = pd.read_csv(input_csv)
    
    motion["feature_change"] = (
        motion["tracked_features"].diff()
    )

    motion["previous_features"] = (
        motion["tracked_features"].shift(1)
    )

    motion["next_features"] = (
        motion["tracked_features"].shift(-1)
    )

    small_arrow = motion["arrow_length"].quantile(PITCH_ARROW_PERCENTILE)

    big_arrow = motion["arrow_length"].quantile(BIG_ARROW_PERCENTILE)

    stable_arrow = motion["arrow_length"].quantile(STABLE_ARROW_PERCENTILE)

    throw_arrow = motion["arrow_length"].quantile(THROW_ARROW_PERCENTILE)

    motion["arrow_mean"] = (
        motion["arrow_length"]
        .rolling(WINDOW, min_periods=1)
        .mean()
    )

    motion["feature_change_mean"] = (
        motion["feature_change"]
        .rolling(WINDOW, min_periods=1)
        .mean()
    )

    motion["camera_state"] = ""
    motion["hard_cut_frame"] = np.nan
    motion["cut_type"] = "Unknown"

    large_pan_counter = 0
    stable_counter = 0
    throw_counter = 0

    current_state = "Pitch View"
    hard_cut_frame = None
    cut_type = None
    
    for i in range(len(motion)):

        row = motion.iloc[i]

        if current_state == "Pitch View":
            # should stay in pitch view or transition?
            if (
                row["tracked_features"] <= FEATURE_DROP_THRESHOLD
                and row["previous_features"] > FEATURE_STABLE_THRESHOLD
                and row["next_features"] > FEATURE_STABLE_THRESHOLD
            ):

                hard_cut_frame = row["frame"]
                motion.loc[i, "hard_cut_frame"] = hard_cut_frame
                cut_type = "Hard Cut"
                current_state = "Following Ball"
                print(f"{row['frame']}: Hard Cut")
                print(f"{row['frame']}: Following Ball")
                stable_counter = 0

            elif (
                row["arrow_mean"] > big_arrow
                and row["feature_change_mean"] < 0
            ):

                large_pan_counter += 1

            else:

                large_pan_counter = 0

            if large_pan_counter >=6:

                cut_type = "Large Pan"
                current_state = "Large Pan"
                print(f"{row['frame']}: Large Pan")
                large_pan_counter = 0

        elif current_state == "Large Pan":

            if (
                row["tracked_features"] > FEATURE_PAN_THRESHOLD
                and row["arrow_length"] < stable_arrow
            ):
                stable_counter += 1

            else:

                stable_counter = 0

            if stable_counter >= 6:
                
                current_state = "Camera Stabilized"
                print(f"{row['frame']}: Camera Stabilized")
                stable_counter = 0
                
        elif current_state == "Following Ball":

            if (
                row["tracked_features"] > FEATURE_STABLE_THRESHOLD
                and row["arrow_length"] < stable_arrow
            ):

                stable_counter += 1

            else:

                stable_counter = 0

            if stable_counter >= 6:

                current_state = "Camera Stabilized"
                print(f"{row['frame']}: Camera Stabilized")
                stable_counter = 0
                
        elif current_state == "Camera Stabilized":

            if row["arrow_length"] > throw_arrow:

                current_state = "Throw Tracking"
                print(f"{row['frame']}: Throw Tracking")

        motion.loc[i, "camera_state"] = current_state
        motion.loc[i, "cut_type"] = cut_type
        
    motion.to_csv(
        output_csv,
        index=False
    )

    print(f"Saved {output_csv}")

    return motion

def main():

    csv_files = sorted(

        f

        for f in os.listdir(CAMERA_FOLDER)

        if f.endswith(".csv")

    )

    for csv_name in csv_files:

        input_csv = os.path.join(
            CAMERA_FOLDER,
            csv_name
        )

        output_csv = os.path.join(
            OUTPUT_FOLDER,
            csv_name
        )

        process_camera_csv(
            input_csv,
            output_csv
        )

if __name__ == "__main__":

    main()
