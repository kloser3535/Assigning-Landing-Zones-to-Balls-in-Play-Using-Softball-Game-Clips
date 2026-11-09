# imports
import pandas as pd
import numpy as np
import os
import cv2

# step 1.1
# input: raw video (mp4)
# output: camera motion csv for each video
from batch_camera_motion import process_video as camera_motion_detector

# step 1.2
# input: camera_motion_csv_files folder for each video
# output: camera_motion_state_csv_files folder for each video
# now we know the camera state for each frame
from camera_state_detector import process_camera_csv as camera_state_detector

# step 2 - NEED TO FIND
# input: raw ball detections csv folder (exact path in notebook - MAY CHANGE)
# this step also computes speed and touch score which will later be used for first touch
# output: cleaned ball detections csv for each video
from clean_ball_detections import clean_ball_file 

# step 3 - NEED TO FIND
# input: raw landmark detections csv folder (exact path in notebook - MAY CHANGE)
# output: cleaned landmark detections csv for each video
from clean_landmark_detections import clean_landmark_file 

# step 4 - NEED TO WRITE
# input: clean_ball_detection_csv_files, clean_landmark_detection_csv_files, camera_motion_state_csv_files
# output: one master csv for each video with all relevant detections (ball, landmark, camera)
from merge_detections import merge_detections_files 

# step 5
# input: master csv file for each video
# (step 6) output: master csv file of the best first touch frame for each video
from first_touch_scorer_v5 import process_video as first_touch_scorer

# step 7
# input: master csv for each video AND master csv for first touch frames
# (step 8) output: master csv of the best calibration frame for each video
from calibration_frame_scorer_v2 import process_video as calibration_frame_scorer

# step 9
# input: master calibration frame csv
# output: master zone assignment with big dictionary
from analyze_batted_ball import analyze_batted_ball

# helper function for saving results to proper export.csv

def update_export_csv(export_path, export_row, results):

    export_df = pd.read_csv(
        export_path,
        dtype=str
    )


    export_row = int(export_row)


    for key,value in results.items():

        column = "BB_" + key


        if column not in export_df.columns:

            export_df[column] = ""


        export_df.loc[
            export_row,
            column
        ] = str(value)


    export_df.to_csv(
        export_path,
        index=False
    )


    print(
        f"Updated {export_path} row {export_row}"
    )

def update_manifest_status(
    manifest,
    index,
    status,
    reason="",
    zone_assigned=False
):

    manifest.loc[
        index,
        "pipeline_done"
    ] = True


    manifest.loc[
        index,
        "pipeline_status"
    ] = status


    manifest.loc[
        index,
        "failure_reason"
    ] = reason


    manifest.loc[
        index,
        "zone_assigned"
    ] = zone_assigned


    return manifest

BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

VIDEO_FOLDER = r"C:\YOLO_BATCH\videos"

RAW_BALL_FOLDER = fr"{BASE_FOLDER}\final_raw_ball_detections_600\balls"

RAW_LANDMARK_FOLDER = fr"{BASE_FOLDER}\final_raw_landmark_detections_600\landmarks"

CAMERA_MOTION_FOLDER = fr"{BASE_FOLDER}\camera_motion_csv_files"

CAMERA_STATE_FOLDER = fr"{BASE_FOLDER}\camera_motion_state_csv_files"

CLEAN_BALL_FOLDER = fr"{BASE_FOLDER}\cleaned_ball_detection_csv_files"

CLEAN_LANDMARK_FOLDER = fr"{BASE_FOLDER}\cleaned_landmark_detection_csv_files"

MASTER_FOLDER = fr"{BASE_FOLDER}\master_detection_csv_files"

MASTER_FIRST_TOUCH_CSV = fr"{BASE_FOLDER}\first_touch_summary.csv"

CALIBRATION_FRAME_FOLDER = fr"{BASE_FOLDER}\calibration_score_csv_files"

CALIBRATION_CANDIDATE_FOLDER = fr"{BASE_FOLDER}\calibration_candidates_topten_csv_files"

OUTPUT_CSV = fr"{BASE_FOLDER}\batted_ball_results.csv"

MANIFEST_CSV = fr"{BASE_FOLDER}\manifest.csv"

manifest = pd.read_csv(MANIFEST_CSV)

manifest["pipeline_done"] = (
    manifest["pipeline_done"]
    .astype(str)
    .str.upper()
    .eq("TRUE")
)

manifest["landmark_done"] = (
    manifest["landmark_done"]
    .astype(str)
    .str.upper()
    .eq("TRUE")
)

manifest["ball_done"] = (
    manifest["ball_done"]
    .astype(str)
    .str.upper()
    .eq("TRUE")
)

calibration_results = []
print("all_results intitialized")
print()
all_results = []

for index, row in manifest.iterrows():

    if (
        row["pipeline_done"]
        and row["pipeline_status"] != "FAILED"
    ):
        continue

    video_path = row["original_path"]
    video_name = row["video_id"]
    
    try:
        print("--------------------------------")
        print(video_name)
        print("--------------------------------")

        camera_motion_detector(video_name, video_path)
        print("\nCamera Motion complete.")

        camera_motion_csv = os.path.join(
            CAMERA_MOTION_FOLDER,
            video_name + ".csv"
        )
        camera_state_csv = os.path.join(
            CAMERA_STATE_FOLDER,
            video_name + ".csv"
        )
        
        camera_state_detector(camera_motion_csv, camera_state_csv)
        print("\nCamera State complete.")

        ball_input_path = os.path.join(
            RAW_BALL_FOLDER,
            video_name + ".csv"
        )
        ball_output_path = os.path.join(
            CLEAN_BALL_FOLDER,
            video_name + ".csv"
        )
        
        clean_ball_file(ball_input_path, ball_output_path)
        print("\nBall Detections cleaned.")
        
        landmark_input_path = os.path.join(
            RAW_LANDMARK_FOLDER,
            video_name + ".csv"
        )
        landmark_output_path = os.path.join(
            CLEAN_LANDMARK_FOLDER,
            video_name + ".csv"
        )
                
        clean_landmark_file(landmark_input_path, landmark_output_path)
        print("\nLandmark Detections cleaned.")

        master_detection_file = os.path.join(
            MASTER_FOLDER,
            video_name + ".csv"
        )
        
        merge_detections_files(
            ball_output_path,
            landmark_output_path,
            camera_state_csv,
            master_detection_file
        )
        print("\nDetections merged.")
        
        first_touch_scorer(video_name)
        print("\nFirst Touch Frame identified.")

        first_touch_df = pd.read_csv(MASTER_FIRST_TOUCH_CSV)

        calibration_output_file = os.path.join(
            CALIBRATION_FRAME_FOLDER,
            video_name + ".csv"
        )

        calibration_candidate_file = os.path.join(
            CALIBRATION_CANDIDATE_FOLDER,
            video_name + ".csv"
        )   
        
        calibration_result = calibration_frame_scorer(video_name, master_detection_file, calibration_candidate_file, calibration_output_file, first_touch_df)

        if calibration_result is None:
            print("No calibration candidates found.")

            update_manifest_status(
                manifest,
                index,
                status="NO_CALIBRATION",
                reason="No calibration candidates generated",
                zone_assigned=False
            )


            manifest.to_csv(
                MANIFEST_CSV,
                index=False
            )
            continue

        print("\nCalibration candidates generated.")

        df = pd.read_csv(master_detection_file)


        # Load top 10 calibration candidates for THIS video
        calibration_candidates = pd.read_csv(
            calibration_candidate_file
        )

        if calibration_candidates.empty:

            print(
                "No calibration candidates for",
                video_name
            )

            update_manifest_status(
                manifest,
                index,
                status="NO_CALIBRATION",
                reason="Calibration candidate CSV empty",
                zone_assigned=False
            )


            manifest.to_csv(
                MANIFEST_CSV,
                index=False
            )

            continue


        print()
        print("Trying calibration candidates:")
        print(
            calibration_candidates[
                [
                    "frame",
                    "quality_level",
                    "ranking_score",
                    "geometry_score_v2"
                ]
            ]
        )

        try:
            results = analyze_batted_ball(
                df,
                calibration_candidates
            )

        except Exception as e:

            update_manifest_status(
                manifest,
                index,
                status="NO_ZONE",
                reason=f"analyze_batted_ball failed: {str(e)}",
                zone_assigned=False
            )


            manifest.to_csv(
                MANIFEST_CSV,
                index=False
            )


            continue
                
        print("\nZone assigned")

        results["video"] = video_name

        # Save master results
        all_results.append(results)

        # paths and function call to update export CSV
        export_path = row["export_csv"]
        export_row = int(row["export_row"])

        if not os.path.exists(export_path):

            raise FileNotFoundError(
                f"Missing Export CSV: {export_path}"
            )

        update_export_csv(
            export_path,
            export_row,
            results
        )

        update_manifest_status(
            manifest,
            index,
            status="SUCCESS",
            reason="",
            zone_assigned=True
        )


        manifest.to_csv(
            MANIFEST_CSV,
            index=False
        )


        print(f"\n{video_name} success.")
                                                          
    except Exception as e:

        print()
        print(f"{video_name} failed.")
        print(e)


        update_manifest_status(
            manifest,
            index,
            status="FAILED",
            reason=str(e),
            zone_assigned=False
        )


        manifest.to_csv(
            MANIFEST_CSV,
            index=False
        )

results_df = pd.DataFrame(all_results)
        
results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("Pipeline complete.")

