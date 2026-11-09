import cv2
import pandas as pd

# ============================================================
# CHANGE THESE PATHS
# ============================================================

VIDEO_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos\FOR_INF_H.mp4"

CSV_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\combined_csv_files_best_landmarks\FOR_INF_H.csv"

START_FRAME = 0

CONF_THRESHOLD = 0.00

SHOW_ONLY_USABLE = True

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

print()
print("Controls")
print("---------------------------")
print("Right Arrow : Next Frame")
print("Left Arrow  : Previous Frame")
print("Space       : Play/Pause")
print("S           : Save Frame")
print("ESC         : Quit")
print()

import numpy as np

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
# ============================================================
# MAIN LOOP
# ============================================================

while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        break

    detections = df[
        (df["frame"] == frame_number) &
        (df["confidence"] >= CONF_THRESHOLD)
    ].copy()

    detections = detections.dropna(
        subset=[
            "center_x",
            "center_y"
        ]
    )

    # Only display usable detections
    if SHOW_ONLY_USABLE and "usable_detection" in detections.columns:
        detections = detections[detections["usable_detection"] == True]

    for _, row in detections.iterrows():

        cls = row["class"]

        # Skip rows with missing center coordinates
        if pd.isna(row["center_x"]) or pd.isna(row["center_y"]):
            continue

        x = int(row["center_x"])
        y = int(row["center_y"])

        color = COLORS.get(cls, (255,255,255))

        # -----------------------------
        # Draw the ball
        # -----------------------------
        if cls == "ball":

            cv2.circle(frame, (x, y), 8, color, 2)

            cv2.putText(
                frame,
                "ball",
                (x+10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # -----------------------------
        # Draw landmarks
        # -----------------------------
        else:

            # Only draw a box if width/height exist
            if (
                pd.notna(row["bbox_width"]) and
                pd.notna(row["bbox_height"])
            ):

                w = row["bbox_width"]
                h = row["bbox_height"]

                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.circle(frame, (x,y), 4, color, -1)

            cv2.putText(
                frame,
                cls,
                (x+8, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
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

import numpy as np

FIELD_POINTS = {
    "home_plate": np.array([0.0, 0.0]),
    "first_base": np.array([60.0, 0.0]),
    "second_base": np.array([60.0, 60.0]),
    "third_base": np.array([0.0, 60.0]),
    "pitching_rubber": np.array([
        43 / np.sqrt(2),
        43 / np.sqrt(2)
    ])
}

def get_frame_points(df, frame_number):

    frame_df = df[df["frame"] == frame_number]

    image_points = {}
    ball_pixel = None

    for _, row in frame_df.iterrows():

        # Skip anything we don't want to use
        if not row["usable_detection"]:
            continue

        point = (row["center_x"], row["center_y"])

        if row["class"] == "ball":
            ball_pixel = point
        else:
            image_points[row["class"]] = point

    return image_points, ball_pixel

FRAME = int(input("Choosen frame: "))
image_points, ball_pixel = get_frame_points(df, FRAME)

print(image_points)
print(ball_pixel)

image_pts, field_pts, names = build_correspondences(image_points)

print()

print("Correspondence Table")

print("-------------------------------")

for name, img, fld in zip(names, image_pts, field_pts):

    print(f"{name:18s}")

    print(f"    Image : ({img[0]:7.1f}, {img[1]:7.1f})")

    print(f"    Field : ({fld[0]:6.2f}, {fld[1]:6.2f})")

    print()

cap.release()
cv2.destroyAllWindows()
