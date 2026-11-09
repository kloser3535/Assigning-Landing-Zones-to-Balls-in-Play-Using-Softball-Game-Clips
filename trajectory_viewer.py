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

TIMELINE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\raw_ball_detections_csv_files\ball_detection_csv_files"

video_name = choose_video(VIDEO_FOLDER)

START_FRAME = 0

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



def draw_ball(frame, row):
        
    if pd.isna(row["center_x"]):
        return

    x = int(row["center_x"])
    y = int(row["center_y"])

    cv2.circle(
        frame,
        (x, y),
        8,
        (0,0,255),
        -1
    )

def view_video(video_name):

    video_path = os.path.join(
        VIDEO_FOLDER,
        video_name + ".mp4"
    )

    timeline_path = os.path.join(
        TIMELINE_FOLDER,
        video_name + ".csv"
    )

    trajectory = pd.read_csv(timeline_path)

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

        current = trajectory[
            trajectory["frame"] == frame_number
        ]
        if not current.empty:
            draw_ball(
                frame,
                current.iloc[0]
            )
            
        trail = trajectory[
            (trajectory["frame"] <= frame_number) &
            (trajectory["frame"] >= frame_number - 25)
        ]

        for _, row in trail.iterrows():

            if pd.isna(row["center_x"]):
                continue

            x = int(row["center_x"])
            y = int(row["center_y"])

            if row["frame"] == frame_number:

                color = (0,0,255)      # red

            elif row["detected"]:

                color = (0,255,0)      # green

            else:

                color = (255,255,255)  # white

            cv2.circle(
                frame,
                (x,y),
                4,
                color,
                -1
            )

            previous = None

            if previous is not None:

                cv2.line(
                    frame,
                    previous,
                    (x, y),
                    color,
                    2
                )

            previous = (x, y)

            cv2.putText(
                frame,
                f"Frame: {frame_number}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            row = current.iloc[0]
            confidence = row["confidence"]

            if confidence >= 0.70:

                confidence_color = (0,255,0)

            elif confidence >= 0.30:

                confidence_color = (0,255,255)

            else:

                confidence_color = (0,0,255)

            cv2.putText(
                frame,
                f"Confidence: {confidence: .2f}",
                (20,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                confidence_color,
                2
            )

            cv2.putText(
                frame,
                f"Detected: {row['detected']}",
                (20,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            if not row["detected"]:

                cv2.line(
                    frame,
                    (x - 6, y - 6),
                    (x + 6, y + 6),
                    (0, 0, 255),
                    2
                )

                cv2.line(
                    frame,
                    (x - 6, y + 6),
                    (x + 6, y - 6),
                    (0, 0, 255),
                    2
                )

            else:

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    color,
                    -1
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
