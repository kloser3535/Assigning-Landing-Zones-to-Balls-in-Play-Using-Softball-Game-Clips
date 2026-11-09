import cv2
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from calibration_models import (
    estimate_similarity,
    estimate_affine,
    estimate_homography,
    transform_points,
    project_ball,
    distance,
    bearing_angle,
    assign_zone,
    normalize_distance,
    mean_landmark_error,
    compute_transformation,
    transform_ball,
    compute_transformations,
)

from analyze_batted_ball import score_transformations

from field_geometry import FIELD_POINTS

from field_viewer import (
    build_zone_polygon,
    draw_field,
)

from frame_scoring_v1 import (
    compute_frame_score,
    choose_model,
    compute_transform
)

# ============================================================
# User Choice for Video
# ============================================================
def choose_video(video_folder):
    video_files = sorted([
        f
        for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(".mp4")
    ])

    print()

    print("Available videos")

    print("----------------------")

    for i, video in enumerate(video_files):

        print(f"{i+1}. {video}")

    choice = int(input("\nChoose video number: "))

    return os.path.splitext(
        video_files[choice-1]
    )[0]

# ============================================================
# CHANGE THESE PATHS
# ============================================================

VIDEO_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\wcws_bip\wcws_champ_series_inf_bip\video"

CSV_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\master_detection_csv_files"

SUMMARY_CSV = (
    r"C:\Users\klose\Downloads\ScreenTrackProject"
    r"\first_touch_summary.csv"
)

VIDEO_NAME = choose_video(VIDEO_FOLDER)

#####################################################
# Top 5 candidates

#####################################################

summary = pd.read_csv(SUMMARY_CSV)

VIDEO_FILE = os.path.join(
    VIDEO_FOLDER,
    VIDEO_NAME + ".mp4"
)

CSV_FILE = os.path.join(
    CSV_FOLDER,
    VIDEO_NAME + ".csv"
)

if not os.path.exists(CSV_FILE):

    raise Exception(
        f"No landmark CSV found for {VIDEO_NAME}"
    )

match = summary[
    summary["video"] == VIDEO_NAME
]

##############################
# Calibration model scorer constants

MODEL_PRIORITY = {
    "similarity": 1,
    "affine": 2,
    "homography": 3,
}

MODEL_SCORE_TOLERANCE = 5
###############################

# =========================================================================
# Start video at best first touch detection
# =========================================================================

START_FRAME = 0

# =============================================================
# Other constants below
# =============================================================

CONF_THRESHOLD = 0.00

# ============================================================
# COLORS
# ============================================================

COLORS = {
    "home_plate": (0,255,255),
    "first_base": (0,255,0),
    "second_base": (255,0,0),
    "third_base": (0,0,255),
    "pitching_rubber": (255,255,0),
    "ball": (255,255,255)
}

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception("Could not open video.")

TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = START_FRAME
playing = False
show_triangle = False

print()
print("Controls")
print("---------------------------")
print("Right Arrow : Next Frame")
print("Left Arrow  : Previous Frame")
print("Space       : Play/Pause")
print("S           : Save Frame")
print("J           : Jump to Frame")
print("T           : Draw Triangle")
print("C           : Print Correspondence Table")
print("G           : Print Geometry Summary")
print("ESC         : Quit")



def get_frame_points(df, frame_number):

    frame_df = df[df["frame"] == frame_number]

    if frame_df.empty:
        return {}, None

    row = frame_df.iloc[0]

    image_points = {}

    landmarks = [
        "home_plate",
        "first_base",
        "second_base",
        "third_base",
        "pitching_rubber"
    ]

    for landmark in landmarks:

        x = row[f"{landmark}_center_x"]
        y = row[f"{landmark}_center_y"]

        if pd.notna(x) and pd.notna(y):
            image_points[landmark] = (float(x), float(y))

    if (
        pd.notna(row["ball_center_x"]) and
        pd.notna(row["ball_center_y"])
    ):
        ball_pixel = (
            float(row["ball_center_x"]),
            float(row["ball_center_y"])
        )
    else:
        ball_pixel = None

    return image_points, ball_pixel

def build_correspondences(image_points):
    """
    Build matching image and field point arrays
    for every visible landmark.
    """

    image_pts = []
    field_pts = []
    landmarks = []

    for landmark in FIELD_POINTS:

        # Skip landmarks that aren't visible
        if landmark not in image_points:
            continue

        image_pts.append(image_points[landmark])
        field_pts.append(FIELD_POINTS[landmark])
        landmarks.append(landmark)

    image_pts = np.array(image_pts, dtype=np.float32)
    field_pts = np.array(field_pts, dtype=np.float32)

    return image_pts, field_pts, landmarks

def draw_triangle(frame, image_pts, names):

    output = frame.copy()

    # Draw landmark points
    for name, pt in zip(names, image_pts):

        x = int(pt[0])
        y = int(pt[1])

        cv2.circle(output, (x, y), 8, (0, 0, 255), -1)

        cv2.putText(
            output,
            name,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            2
        )

    if len(image_pts) >= 3:

        cv2.polylines(
            output,
            [image_pts.astype(np.int32)],
            True,
            (0,255,0),
            2
        )

    return output

def run_geometry(frame, frame_number):

    print()
    print("=" * 50)
    print(f"Geometry for Frame {frame_number}")
    print("=" * 50)

    image_points, ball_pixel = get_frame_points(df, frame_number)

    image_pts, field_pts, names = build_correspondences(image_points)

    print("Correspondence Table")
    print("-------------------------------")

    for name, img, fld in zip(names, image_pts, field_pts):

        print(f"{name:18s}")
        print(f"    Image : ({img[0]:7.1f}, {img[1]:7.1f})")
        print(f"    Field : ({fld[0]:6.2f}, {fld[1]:6f})")
        print()

    debug_frame = draw_geometry(frame, image_pts, names)

def print_correspondence_table(frame_number, image_pts, field_pts, names, ball_pixel):

    print()
    print("="*60)
    print(f"Frame {frame_number}")
    print("="*60)

    for name, img, fld in zip(names, image_pts, field_pts):

        print(f"{name:18s}")

        print(f"    Image : ({img[0]:7.1f}, {img[1]:7.1f})")

        print(f"    Field : ({fld[0]:6.2f}, {fld[1]:6.2f})")

        print()

    if ball_pixel is not None:

        print(f"Ball pixel : ({ball_pixel[0]:.1f}, {ball_pixel[1]:.1f})")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        break

    frame_row = df[df["frame"] == frame_number]

    if frame_row.empty:
        continue

    row = frame_row.iloc[0]

    image_points, ball_pixel = get_frame_points(df, frame_number)

    # draw landmarks
    for landmark, (x, y) in image_points.items():

        x = int(x)
        y = int(y)

        color = COLORS.get(landmark, (255,255,255))

        cv2.circle(frame, (x, y), 5, color, -1)

        cv2.putText(
            frame,
            landmark,
            (x+8, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # draw ball
    if ball_pixel is not None:

        x = int(ball_pixel[0])
        y = int(ball_pixel[1])

        cv2.circle(frame, (x, y), 8, COLORS["ball"], 2)

        cv2.putText(
            frame,
            "ball",
            (x+10, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["ball"],
            2
        )

    info = f"Frame {frame_number}/{TOTAL_FRAMES-1}"

    cv2.putText(
        frame,
        info,
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    if show_triangle:

        image_points, ball_pixel = get_frame_points(df, frame_number)

        image_pts, field_pts, names = build_correspondences(image_points)

        frame = draw_triangle(frame, image_pts, names)

    cv2.imshow("Landmark Viewer", frame)

    if playing:

        key = cv2.waitKey(30)

        if key == 32:
            playing = False

        elif key == 27:
            break

        elif key == ord("s"):

            filename = f"frame_{frame_number}.png"

            cv2.imwrite(filename, frame)

            print(f"Saved {filename}")

        else:

            if frame_number < TOTAL_FRAMES-1:
                frame_number += 1

    else:

        key = cv2.waitKeyEx(0)

        if key == 27:
            break

        elif key == 32:
            playing = True

        elif key == 2555904:

            if frame_number < TOTAL_FRAMES-1:
                frame_number += 1

        elif key == 2424832:

            if frame_number > 0:
                frame_number -= 1

        elif key == ord("s"):

            filename = f"frame_{frame_number}.png"

            cv2.imwrite(filename, frame)

            print(f"Saved {filename}")

        elif key == ord("g") or key == ord("G"):

            print("\n" + "="*60)
            print(f"3-MODEL COMPARISON - Frame {frame_number}")
            print("="*60)

            # --------------------------------------------------------
            # STEP 1: Extract frame data
            # --------------------------------------------------------
            image_points, ball_pixel = get_frame_points(df, frame_number)

            image_pts, field_pts, names = build_correspondences(image_points)

            print("Visible landmarks:", names)

            transforms = compute_transformations(
                image_pts,
                field_pts
            )

            candidate_landing_points = project_ball(
                transforms,
                ball_pixel
            )

            scores = score_transformations(
                candidate_landing_points,
                transforms,
                image_pts,
                field_pts
            )

            ranked_models = sorted(
                scores.items(),
                key=lambda x: x[1]["overall_score"],
                reverse=True
            )

            top_model = ranked_models[0][0]
            top_score = ranked_models[0][1]["overall_score"]

            if len(ranked_models) == 1:

                best_model = top_model

            else:

                second_model = ranked_models[1][0]
                second_score = ranked_models[1][1]["overall_score"]

                score_gap = top_score - second_score

                if score_gap > MODEL_SCORE_TOLERANCE:

                    best_model = top_model

                else:

                    if MODEL_PRIORITY[top_model] > MODEL_PRIORITY[second_model]:

                        best_model = top_model

                    else:

                        best_model = second_model

            print("\nMODEL SCORES")

            for model_name, score in ranked_models:

                print(
                    model_name,
                    score["overall_score"]
                )

            print()
            print("Winner:", best_model)

            fig, ax = plt.subplots()

            ax.scatter(0,0,color="black")
            ax.text(0,0,"Home")

            ax.scatter(60,0,color="black")
            ax.text(60,0,"1B")

            ax.scatter(60,60,color="black")
            ax.text(60,60,"2B")

            ax.scatter(0,60,color="black")
            ax.text(0,60,"3B")
            
            point = candidate_landing_points.get(best_model)

            if point is not None:

                ax.scatter(
                    point[0],
                    point[1],
                    color="blue",
                    s=80,
                    label=best_model
                )

            ax.legend()

            ax.set_xlim(-50,250)
            ax.set_ylim(-50,250)

            ax.set_aspect("equal")

            plt.show()

        elif key == ord('z'):

            image_points, ball_pixel = get_frame_points(df, frame_number)

            image_pts, field_pts, names = build_correspondences(image_points)

            num_landmarks = len(image_pts)

            if num_landmarks < 2:
                print("Not enough landmarks.")
                continue

            transform, model_type, landmark_error = compute_transformation(image_pts, field_pts)      

            transforms = compute_transformations(
                image_pts,
                field_pts
            )

            candidate_landing_points = project_ball(
                transforms,
                ball_pixel
            )

            scores = score_transformations(
                candidate_landing_points,
                transforms,
                image_pts,
                field_pts
            )

            # same winner-selection code as above

            ball_field = candidate_landing_points[best_model]

            if ball_field is None:
                print("Ball could not be projected.")
                continue
            
            distance_home = distance(ball_field, FIELD_POINTS["home_plate"])

            bearing = bearing_angle(ball_field)

            u = normalize_distance(distance_home, bearing)

            zone = assign_zone(u, bearing)

            draw_field(
                ball=ball_field,
                highlight_zone=zone
            )

        elif key == ord("t") or key == ord("T"):

            show_triangle = not show_triangle

            print(f"Triangle overlay: {show_triangle}")

        elif key == ord("c") or key == ord("C"):

            image_points, ball_pixel = get_frame_points(df, frame_number)

            image_pts, field_pts, names = build_correspondences(image_points)

            print_correspondence_table(
                frame_number,
                image_pts,
                field_pts,
                names,
                ball_pixel
            )

        elif key == ord("j") or key == ord("J"):

            jump = input("Jump to frame: ")

            try:

                jump = int(jump)

                if 0 <= jump < TOTAL_FRAMES:

                    frame_number = jump

                    print(f"Jumped to frame {frame_number}")

                else:

                    print("Frame out of range.")

            except ValueError:

                print("Invalid frame number.")

cap.release()
cv2.destroyAllWindows()
