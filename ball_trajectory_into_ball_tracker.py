# Imports
import pandas as pd
import numpy as np
import os

# Functions
def allowed_position_error(row, tracking_status):
    if row["camera_state"] == "Pitch View":
        return None
    if row["camera_state"] == "Following Ball":

        if row["cut_type"] == "Hard Cut":
            if tracking_status == "TRACKING":
                return 30
            elif tracking_status == "REACQUIRING":
                return 70
            elif tracking_status == "LOST":
                return 120

        if row["cut_type"] == "Large Pan":
            if tracking_status == "TRACKING":
                return 50
            elif tracking_statis == "REACQUIRING":
                return 90
            elif tracking_status == "LOST":
                return 140
    elif row["camera_state"] == "Camera Stabilized":

        if row["cut_type"] == "Hard Cut":
            if tracking_status == "TRACKING":
                return 30
            elif tracking_status == "REACQUIRING":
                return 50
            elif tracking_status == "LOST":
                return 80

        if row["cut_type"] == "Large Pan":
            if tracking_status == "TRACKING":
                return 25
            elif tracking_status == "REACQUIRING":
                return 45
            elif tracking_status == "LOST":
                return 75
    elif row["camera_state"] == "Throw Tracking":

        if row["frames_since_throw"] <= 125:
            if tracking_status == "TRACKING":
                return 65
            elif tracking_status == "REACQUIRING":
                return 100
            elif tracking_status == "LOST":
                return 140

        return None

def allowed_motion_error(row, tracking_status):
    if tracking_status == "TRACKING":
        return 10
    elif tracking_status == "REACQUIRING":
        return 20
    elif tracking_status == "LOST":
        return 40
    return None

def motion_error(
    track_dx,
    track_dy,
    detection_dx,
    detection_dy
):

    return np.sqrt(
        (detection_dx - track_dx)**2 +
        (detection_dy - track_dy)**2
    )

def position_error( # pythagorean theorem
    track_x,
    track_y,
    detection_x,
    detection_y
):

    return np.sqrt(
        (detection_x - track_x)**2 +
        (detection_y - track_y)**2
    )

def process_video(video_name):

    BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

    TRAJECTORY_CSV = fr"{BASE_FOLDER}\trajectory_csv_files\{video_name}.csv"

    OUTPUT_CSV = TRAJECTORY_CSV

    df = pd.read_csv(TRAJECTORY_CSV)

    # basically the jump that we computed during the first stages of writing this script
    df["detection_dx"] = np.nan
    df["detection_dy"] = np.nan
    
    df["track_x"] = np.nan
    df["track_y"] = np.nan

    df["track_dx"] = np.nan
    df["track_dy"] = np.nan

    df["refined_x"] = np.nan
    df["refined_y"] = np.nan

    df["tracking_status"] = ""

    df["position_error"] = np.nan
    df["motion_error"] = np.nan

    df["candidate"] = False

    df["candidate_count"] = 0
    df["bad_detection_count"] = 0

    # where do I think the ball is?
    track_x = None
    track_y = None

    track_initialized = False

    track_dx = 0
    track_dy = 0
    
    tracking_status = "TRACKING"

    candidate_count = 0
    bad_detection_count = 0
    cut_start_frame = None

    previous_detection_x = None
    previous_detection_y = None

    previous_track_x = None
    previous_track_y = None

    for i in range(len(df)):
        row = df.loc[i]
        
        # YOLO
        detection_x = row["center_x"]
        detection_y = row["center_y"]
        
        camera_state = row["camera_state"]
        cut_type = row["cut_type"]

        # create frames_after_cut row
        if (
            cut_start_frame is None
            and camera_state in ["Hard Cut", "Large Pan"]
        ):
            cut_start_frame = row["frame"]

        if cut_start_frame is None:
            df.loc[i, "frames_after_cut"] = np.nan
        else:
            df.loc[i, "frames_after_cut"] = (
                row["frame"] - cut_start_frame
            )
        ###############################
            
        tracking_enabled = (
            row["camera_state"] in [
                "Following Ball",
                "Camera Stabilized",
                "Throw Tracking"
            ]
        )

        if camera_state == "Pitch View":
            if not pd.isna(detection_x):
                if previous_detection_x is None:
                    detection_dx = 0
                    detection_dy = 0
                else:
                    detection_dx = detection_x - previous_detection_x
                    detection_dy = detection_y - previous_detection_y

                previous_detection_x = detection_x
                previous_detection_y = detection_y

                track_x = detection_x
                track_y = detection_y
                previous_track_x = track_x
                previous_track_y = track_y

                track_dx = detection_dx
                track_dy = detection_dy

                refined_x = track_x
                refined_y = track_y

                track_initialized = True
                continue
                # notice that we are not saving anything to the columns (but we are updating the variable for future frames
                # we are just making sure we store where the ball's last known location was    
            else:
                continue

        if ((cut_type == "Hard Cut" and row["frames_after_cut"] <= 10)
            or (cut_type == "Large Pan" and row["frames_after_cut"] <= 5)):
            track_x += track_dx
            track_y += track_dy

            refined_x = track_x
            refined_y = track_y

            previous_detection_x = detection_x
            previous_detection_y = detection_y

            previous_track_x = track_x
            previous_track_y = track_y

            continue # since we do not want to trust YOLO yet...

        # if the code reaches this point, we know that the camera state is one of the following: following ball, camera stabilize, or throw tracking
        
        if track_initialized:
            track_x += track_dx
            track_y += track_dy

            if previous_detection_x is None:
                previous_detection_x = detection_x
                previous_detection_y = detection_y
                detection_dx = 0
                detection_dy = 0
            else:
                detection_dx = detection_x - previous_detection_x
                detection_dy = detection_y - previous_detection_y
                previous_detection_x = detection_x
                previous_detection_y = detection_y

        if not track_initialized:
            if pd.isna(detection_x) or pd.isna(detection_y):
                continue
        
            track_x = detection_x
            track_y = detection_y

            track_dx = 0
            track_dy = 0

            previous_detection_x = detection_x
            previous_detection_y = detection_y

            previous_track_x = track_x
            previous_track_y = track_y

            df.loc[i,"track_x"] = track_x
            df.loc[i,"track_y"] = track_y
            df.loc[i,"refined_x"] = track_x
            df.loc[i,"refined_y"] = track_y

            track_initialized = True

            continue

        pos_error = position_error(
            track_x,
            track_y,
            detection_x,
            detection_y
        )

        mot_error = motion_error(
            track_dx,
            track_dy,
            detection_dx,
            detection_dy
        )

        df.loc[i, "position_error"] = pos_error
        df.loc[i, "motion_error"] = mot_error

        limit = allowed_position_error(row, tracking_status)

        good_position = (
            limit is None or
            pos_error <= limit
        )

        motion_limit = allowed_motion_error(row, tracking_status)

        good_motion = (mot_error <= motion_limit)

        df.loc[i, "good_position"] = good_position
        df.loc[i, "good_motion"] = good_motion
                        
        candidate = (
            good_position and
            good_motion
        )

        df.loc[i, "candidate"] = candidate

        if tracking_status == "TRACKING":

            if candidate:
                track_x = detection_x # this merges the track x/y and detection x/y since we trust YOLO
                track_y = detection_y

                track_dx = detection_dx # this merges the track dx/dy and detection dx/dy since we trust YOLO
                track_dy = detection_dy # dx/dy are kind of like how the camera moved...
                
                previous_detection_x = detection_x
                previous_detection_y = detection_y

                previous_track_x = track_x
                previous_track_y = track_y
                
                bad_detection_count = 0
         
            else:
                bad_detection_count += 1
                
                previous_detection_x = detection_x
                previous_detection_y = detection_y
                                                    # the previous detection and previous track are NOT the same because we did not trust YOLO                                                
                previous_track_x = track_x
                previous_track_y = track_y
                
                if bad_detection_count >= 4:
                    tracking_status = "REACQUIRING"
                    bad_detection_count = 0

        elif tracking_status == "REACQUIRING":

            if candidate:
                candidate_count += 1
                if candidate_count >= 4:

                    tracking_status = "TRACKING"

                    track_x = detection_x
                    track_y = detection_y

                    track_dx = detection_dx
                    track_dy = detection_dy

                    previous_detection_x = detection_x
                    previous_detection_y = detection_y

                    previous_track_x = track_x
                    previous_track_y = track_y

                    candidate_count = 0
                    bad_detection_count = 0

            else:

                candidate_count = 0
                bad_detection_count += 1

                if bad_detection_count >= 15:
                    tracking_status = "LOST"

        elif tracking_status == "LOST":

            if candidate:

                candidate_count += 1

                if candidate_count >= 4:

                    tracking_status = "TRACKING"

                    track_x = detection_x
                    track_y = detection_y

                    track_dx = detection_dx
                    track_dy = detection_dy

                    previous_detection_x = detection_x
                    previous_detection_y = detection_y

                    previous_track_x = track_x
                    previous_track_y = track_y

                    candidate_count = 0
                    bad_detection_count = 0

            else:

                candidate_count = 0

        df.loc[i, "detection_dx"] = detection_dx
        df.loc[i, "detection_dy"] = detection_dy
        df.loc[i, "refined_x"] = track_x
        df.loc[i, "refined_y"] = track_y
        df.loc[i, "track_dx"] = track_dx
        df.loc[i, "track_dy"] = track_dy
        df.loc[i, "tracking_status"] = tracking_status
        df.loc[i, "candidate_count"] = candidate_count
        df.loc[i, "bad_detection_count"] = bad_detection_count
        
    df.to_csv(
        OUTPUT_CSV,
        index=False
    )


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
