# Imports
import pandas as pd
import numpy as np
import os

# Constants

# Functions
def allowed_position_error(row):
    if row["camera_state"] == "Pitch View":
        return None
    if row["camera_state"] == "Following Ball":

        if row["cut_type"] == "Hard Cut":
            return 30

        if row["cut_type"] == "Large Pan":
            return 50
    if row["camera_state"] == "Camera Stabilized":

        if row["cut_type"] == "Hard Cut":
            return 30

        if row["cut_type"] == "Large Pan":
            return 25
    if row["camera_state"] == "Throw Tracking":

        if row["frames_since_throw"] <= 125:
            return 65

        return None

def motion_error(
    predicted_dx,
    predicted_dy,
    actual_dx,
    actual_dy
):

    return np.sqrt(
        (actual_dx - predicted_dx)**2 +
        (actual_dy - predicted_dy)**2
    )

def position_error(
    predicted_x,
    predicted_y,
    current_x,
    current_y
):

    return np.sqrt(
        (current_x - predicted_x)**2 +
        (current_y - predicted_y)**2
    )

# Process video function
def process_video(video_name):
# File paths
    BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

    TRAJECTORY_CSV = fr"{BASE_FOLDER}\trajectory_csv_files\{video_name}.csv"
    
    OUTPUT_CSV = fr"{BASE_FOLDER}\trajectory_csv_files\{video_name}.csv"

    trajectory = pd.read_csv(TRAJECTORY_CSV)
    
    merged = trajectory.copy()

    # new columns (at first every detection is trusted that is why refined coordinates are equal to center coordinates)
    merged["frame_after_cut"] = np.nan
    merged["frames_since_throw"]= np.nan
    merged["detection_trusted"] = True
    merged["rejection_reason"] = ""
    merged["tracking_state"] = ""
    merged["candidate_count"] = 0
    merged["bad_detection_count"] = 0
    merged["delta_x"] = np.nan
    merged["delta_y"] = np.nan
    merged["position_error"] = np.nan
    merged["motion_error"] = np.nan
    merged["good_position"] = False
    merged["good_motion"] = False
    merged["candidate"] = False
    merged["last_predicted_x"] = np.nan
    merged["last_predicted_y"] = np.nan
    merged["predicted_x"] = np.nan
    merged["predicted_y"] = np.nan
    merged["predicted_dx"] = np.nan
    merged["predicted_dy"] = np.nan
    merged["refined_x"] = np.nan
    merged["refined_y"] = np.nan
    
    last_predicted_x = None
    last_predicted_y = None
    
    predicted_x = None
    predicted_y = None

    predicted_dx = 0
    predicted_dy = 0

    tracking_state = "TRACKING"
    bad_detection_count = 0
    candidate_count = 0
    lost_count = 0

    cut_start_frame = None
    throw_start_frame = None

    for i in range(len(merged)):

        row = merged.loc[i]

        current_x = row["center_x"]
        current_y = row["center_y"]

        camera_state = row["camera_state"]
        cut_type = row["cut_type"]

##        row = merged.loc[i]
##        current_state = merged.loc[i, "camera_state"]
##        
##        if (
##            cut_start_frame is None
##            and current_state in ["Hard Cut", "Large Pan"]
##        ):
##            cut_start_frame = row["frame"]
##
##        if cut_start_frame is None:
##            merged.loc[i, "frames_after_cut"] = np.nan
##        else:
##            merged.loc[i, "frames_after_cut"] = (
##                row["frame"] - cut_start_frame
##            )
##
##        if (
##            throw_start_frame is None
##            and current_state in ["Throw Tracking"]
##        ):
##            throw_start_frame = row["frame"]
##
##        if throw_start_frame is None:
##            merged.loc[i, "frames_since_throw"] = np.nan
##        else:
##            merged.loc[i, "frames_since_throw"] = (
##                row["frame"] - throw_start_frame
##            )
##
##        merged.loc[i, "last_predicted_x"] = last_predicted_x
##        merged.loc[i, "last_predicted_y"] = last_predicted_y
##
##        # 1) get the YOLO detection
##        #current_x = merged["center_x"] is the whole row but only want one field
##        current_x = merged.loc[i, "center_x"]
##        current_y = merged.loc[i, "center_y"]
##
##        if row["camera_state"] == "Pitch View":
##            
##            if pd.isna(current_x) or pd.isna(current_y):
##                continue
##            
##            predicted_x = current_x
##            predicted_y = current_y
##
##            if last_predicted_x is not None:
##
##                predicted_dx = current_x - last_predicted_x
##                predicted_dy = current_y - last_predicted_y
##
##            last_predicted_x = current_x
##            last_predicted_y = current_y
##
##            merged.loc[i, "predicted_x"] = predicted_x
##            merged.loc[i, "predicted_y"] = predicted_y
##            merged.loc[i, "refined_x"] = current_x
##            merged.loc[i, "refined_y"] = current_y
##
##            continue
##
##        # Update predicted coordinates
##        if predicted_x is None:
##
##            if pd.isna(current_x) or pd.isna(current_y):
##                continue
##
##            predicted_x = current_x
##            predicted_y = current_y
##
##            last_predicted_x = current_x
##            last_predicted_y = current_y
##
##            merged.loc[i,"predicted_x"] = predicted_x
##            merged.loc[i,"predicted_y"] = predicted_y
##            merged.loc[i,"refined_x"] = predicted_x
##            merged.loc[i,"refined_y"] = predicted_y
##
##            continue
##                    
##        if last_predicted_x is None:
##
##            predicted_x = current_x
##            predicted_y = current_y
##
##            last_predicted_x = current_x
##            last_predicted_y = current_y
##
##            predicted_dx = 0
##            predicted_dy = 0
##
##            merged.loc[i, "predicted_x"] = predicted_x
##            merged.loc[i, "predicted_y"] = predicted_y
##            merged.loc[i, "refined_x"] = predicted_x
##            merged.loc[i, "refined_y"] = predicted_y
##
##            continue
##        
##        if (
##            (row["cut_type"] == "Hard Cut"
##            and row["frames_after_cut"] <= 10) or
##            (row["cut_type"] == "Large Pan"
##             and row["frames_after_cut"] <= 5)
##        ):
##            tracking_state = "REACQUIRING"
##
##            predicted_x += predicted_dx
##            predicted_y += predicted_dy
##
##            last_predicted_x = predicted_x
##            last_predicted_y = predicted_y
##                        
##            merged.loc[i, "predicted_x"] = predicted_x
##            merged.loc[i, "predicted_y"] = predicted_y
##            merged.loc[i, "predicted_dx"] = predicted_dx
##            merged.loc[i, "predicted_dy"] = predicted_dy                
##            merged.loc[i, "refined_x"] = predicted_x
##            merged.loc[i, "refined_y"] = predicted_y
##
##            continue
##
##        # 2) Predict where the ball should be
##
##        delta_x = current_x - last_predicted_x
##        delta_y = current_y - last_predicted_y
##        merged.loc[i, "delta_x"] = delta_x
##        merged.loc[i, "delta_y"] = delta_y
##        actual_dx = delta_x
##        actual_dy = delta_y
##        
##
##        # 3) compute position error and motion error
##
##        # how far is the YOLO ball detection away from where I thought the ball should be (based on predicted ball coordinates))
##        pos_error = position_error(predicted_x, predicted_y, current_x, current_y)
##        merged.loc[i, "position_error"] = pos_error
##
##        # is the ball moving in the direction I predicted it to move in?
##        mot_error = motion_error(predicted_dx, predicted_dy, actual_dx, actual_dy)
##        merged.loc[i, "motion_error"] = mot_error
##
##        # 4) is the YOLO ball detection a candidate?
##        limit = allowed_position_error(row)
##        good_position = (limit is None or pos_error <= limit)
##        merged.loc[i, "good_position"] = good_position
##        
##        good_motion = (mot_error <= 10)
##        merged.loc[i, "good_motion"] = good_motion
##
##        candidate = (good_position and good_motion)
##        merged.loc[i, "candidate"] = candidate
## 
##        if pd.isna(current_x) or pd.isna(current_y):
##            merged.loc[i, "detection_trusted"] = False
##            merged.loc[i, "rejection_reason"] = "Missing Detection"
##            bad_detection_count +=1
##            lost_count += 1
##            merged.loc[i, "predicted_x"] = predicted_x
##            merged.loc[i, "predicted_y"] = predicted_y
##            merged.loc[i, "predicted_dx"] = predicted_dx
##            merged.loc[i, "predicted_dy"] = predicted_dy
##            merged.loc[i, "refined_x"] = predicted_x
##            merged.loc[i, "refined_y"] = predicted_y
##            if lost_count >= 10:
##                tracking_state = "LOST"
##                candidate_count = 0
##                bad_detection_count = 0
##                lost_count = 0
##            continue
##
##        # 5) Update tracking state
##        if tracking_state == "TRACKING":
##            if candidate:
##                predicted_x = current_x
##                predicted_y = current_y
##                predicted_dx = delta_x
##                predicted_dy = delta_y
##                last_predicted_x = current_x
##                last_predicted_y = current_y
##                bad_detection_count = 0
##                lost_count = 0
##            elif not candidate:
##                bad_detection_count += 1
##                lost_count = 0
##                last_predicted_x = predicted_x
##                last_predicted_y = predicted_y                
##                if bad_detection_count >= 4:
##                    tracking_state = "REACQUIRING"
##                    bad_detection_count = 0
##
##            
##        elif tracking_state == "REACQUIRING":
##            if candidate:
##                candidate_count += 1
##                bad_detection_count = 0
##                lost_count = 0
##                predicted_x += predicted_dx
##                predicted_y += predicted_dy
##                last_predicted_x = predicted_x
##                last_predicted_y = predicted_y
##                if candidate_count >= 4:
##                    tracking_state = "TRACKING"
##                    predicted_x = current_x
##                    predicted_y = current_y
##                    predicted_dx = delta_x
##                    predicted_dy = delta_y
##                    last_predicted_x = current_x
##                    last_predicted_y = current_y
##                    bad_detection_count = 0
##                    candidate_count = 0
##                    lost_count = 0
##            elif not candidate:
##                candidate_count = 0
##                lost_count = 0
##                bad_detection_count += 1
##                predicted_x += predicted_dx
##                predicted_y += predicted_dy
##
##        elif tracking_state == "LOST":
##            if candidate:
##                candidate_count += 1
##                bad_detection_count = 0
##                lost_count = 0
##                if candidate_count >= 4:
##                    tracking_state = "TRACKING"
##                    predicted_x = current_x
##                    predicted_y = current_y
##                    predicted_dx = delta_x
##                    predicted_dy = delta_y
##                    last_good_x = current_x
##                    last_good_y = current_y
##                    bad_detection_count = 0
##                    candidate_count = 0
##                    lost_count = 0
##            elif not candidate:
##                candidate_count = 0
##                bad_detection_count += 1
##                lost_count += 1
##                predicted_x += predicted_dx
##                predicted_y += predicted_dy
##                
##        merged.loc[i, "predicted_x"] = predicted_x
##        merged.loc[i, "predicted_y"] = predicted_y 
##        merged.loc[i, "predicted_dx"] = predicted_dx
##        merged.loc[i, "predicted_dy"] = predicted_dy
##        merged.loc[i, "refined_x"] = predicted_x
##        merged.loc[i, "refined_y"] = predicted_y
##    
##    merged.to_csv(
##        OUTPUT_CSV,
##        index=False
##    )
##    
def main():

    trajectory_folder = r"C:\Users\klose\Downloads\ScreenTrackProject\trajectory_csv_files"

    csv_files = sorted(
        f
        for f in os.listdir(trajectory_folder)
        if f.endswith(".csv")
    )

    for csv_name in csv_files:

        video_name = os.path.splitext(csv_name)[0]

        process_video(video_name)

        print("Video complete!")

if __name__ == "__main__":
    main()
