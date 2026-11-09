import cv2
import pandas as pd
import numpy as np
import os

from camera_motion import (
    find_features,
    track_features,
    estimate_camera_transform,
    summarize_motion,
    draw_motion_vectors,
    draw_motion_summary
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

VIDEO_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos"

CAMERA_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\camera_motion_csv_files"

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


def view_video(video_name):

    video_path = os.path.join(
        VIDEO_FOLDER,
        video_name + ".mp4"
    )

    camera_csv = os.path.join(
        CAMERA_FOLDER,
        video_name + ".csv"
    ) 

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Could not open video.")

    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = START_FRAME

    playing = False

    results = []

    while True:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, previous_frame = cap.read()

        success, current_frame = cap.read()

        if not success:
            break

        old_points, new_points = track_features(
            previous_frame,
            current_frame
        )

        transform = estimate_camera_transform(
            old_points,
            new_points
        )

        motion = summarize_motion(
            transform,
            old_points
        )

        pan_pixels = np.sqrt(
            motion["pan_x"]**2 +
            motion["pan_y"]**2
        )

        results.append({

            "frame": frame_number,

            "pan_x": motion["pan_x"],

            "pan_y": motion["pan_y"],

            "zoom": motion["scale"],

            "rotation": motion["rotation"],

            "pan_pixels": pan_pixels,

            "tracked_features": motion["tracked_features"]

        })

        motion["frame"] = frame_number

        frame = draw_motion_vectors(
            current_frame,
            old_points,
            new_points
        )

        frame = draw_motion_summary(
            frame,
            motion
        )

        cv2.putText(
            frame,
            f"Tracked Features: {motion['tracked_features']}",
            (20,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )
        
    # display large vector that summarizes the average motion of all features on the frame (so basically this is how the camera is moving) 
    # -----------------------------------------------------
        dx = np.mean(
            new_points[:,0] - old_points[:,0]
        )

        dy = np.mean(
            new_points[:,1] - old_points[:,1]
        )

        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2

        scale = 5  # makes the arrow easier to see

        cv2.arrowedLine(
            frame,
            (center_x, center_y),
            (
                int(center_x + dx * scale),
                int(center_y + dy * scale)
            ),
            (0,255,255),
            3
        ) 
    # -----------------------------------------------------

        cv2.imshow(
            "Camera Motion Viewer",
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

