import cv2
import pandas as pd
import numpy as np
import os

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


VIDEO_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos"

MODEL_N_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\raw_ball_detections_csv_files_bt\ball_detection_csv_files_v8s"

MODEL_S_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\raw_ball_detections_csv_files_v8s"

video_name = choose_video(VIDEO_FOLDER)

START_FRAME = 0

MIN_CONFIDENCE = 0.15

print()
print("Controls")
print("---------------------------")
print("Right Arrow : Next Frame")
print("Left Arrow  : Previous Frame")
print("Space       : Play/Pause")
print("S           : Save Frame")
print("J           : Jump to Frame")
print("ESC         : Quit")
print()


def draw_model(
        frame,
        trajectory,
        frame_number,
        name,
        current_color,
        trail_color
):

    # ------------------------------
    # Current detection
    # ------------------------------

    current = trajectory[
        trajectory["frame"] == frame_number
    ]


    if not current.empty:

        row = current.iloc[0]

        if pd.notna(row["center_x"]):

            x = int(row["center_x"])
            y = int(row["center_y"])

            cv2.circle(
                frame,
                (x,y),
                8,
                current_color,
                -1
            )


    # ------------------------------
    # Trail
    # ------------------------------

    trail = trajectory[
        (trajectory["frame"] <= frame_number) &
        (trajectory["frame"] >= frame_number - 25)
    ]

    previous = None

    for _, row in trail.iterrows():

        if pd.isna(row["center_x"]):
            continue


        x = int(row["center_x"])
        y = int(row["center_y"])


        cv2.circle(
            frame,
            (x,y),
            4,
            trail_color,
            -1
        )


        if previous is not None:

            cv2.line(
                frame,
                previous,
                (x,y),
                trail_color,
                2
            )


        previous = (x,y)


    # ------------------------------
    # Display information
    # ------------------------------

    if not current.empty:

        row = current.iloc[0]

        cv2.putText(
            frame,
            f"{name} Conf: {row['confidence']:.2f}",
            (20,90 if name=="S" else 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            current_color,
            2
        )

def view_video(video_name):

    video_path = os.path.join(
        VIDEO_FOLDER,
        video_name + ".mp4"
    )

    model_s_path = os.path.join(
        MODEL_S_FOLDER,
        video_name + ".csv"
    )

    model_n_path = os.path.join(
        MODEL_N_FOLDER,
        video_name + ".csv"
    )

    trajectory_s = pd.read_csv(model_s_path)

    trajectory_n = pd.read_csv(model_n_path)

    trajectory_n = trajectory_n[
        trajectory_n["confidence"] >= MIN_CONFIDENCE
    ].copy()


    trajectory_n = (
        trajectory_n
        .sort_values(
            by=["frame","confidence"],
            ascending=[True, False]
        )
        .drop_duplicates(
            subset="frame",
            keep="first"
        )
    )

    trajectory_s = trajectory_s[
        trajectory_s["confidence"] >= MIN_CONFIDENCE
    ].copy()
    
    trajectory_s = (
        trajectory_s
        .sort_values(
            by=["frame","confidence"],
            ascending=[True, False]
        )
        .drop_duplicates(
            subset="frame",
            keep="first"
        )
    )
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open video.")

    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = START_FRAME

    playing = False

    while True:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if not success:
            break

        draw_model(
            frame,
            trajectory_s,
            frame_number,
            "S",
            (0,0,255),      # red current
            (0,0,150)       # dark red trail
        )


        draw_model(
            frame,
            trajectory_n,
            frame_number,
            "N",
            (255,255,0),    # cyan current
            (150,150,0)     # dark cyan trail
        )


        cv2.imshow(
            "Trajectory Viewer",
            frame
        )

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

if __name__ == "__main__":

    view_video(
        video_name
    )
