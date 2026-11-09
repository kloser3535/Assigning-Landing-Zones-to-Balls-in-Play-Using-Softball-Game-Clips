import pandas as pd
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

CALIBRATION_FRAME = int(input("Choosen frame: "))

frame_df = df[df["frame"] == CALIBRATION_FRAME]

print(frame_df)

print(
    frame_df[
        [
            "class",
            "center_x",
            "center_y",
            "confidence",
            "usable_detection"
        ]
    ]
)

def get_image_points(frame_df):
    """
    Extract landmark and ball coordinates from one frame.
    """

    image_points = {}
    ball_pixel = None

    for _, row in frame_df.iterrows():

        if not row["usable_detection"]:
            continue

        point = (
            float(row["center_x"]),
            float(row["center_y"])
        )

        if row["class"] == "ball":
            ball_pixel = point
        else:
            image_points[row["class"]] = point

    return image_points, ball_pixel

image_points, ball_pixel = get_image_points(frame_df)

print(image_points)

print(ball_pixel)

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

image_pts, field_pts, names = build_correspondences(image_points)

print()

print("Correspondence Table")

print("-------------------------------")

for name, img, fld in zip(names, image_pts, field_pts):

    print(f"{name:18s}")

    print(f"    Image : ({img[0]:7.1f}, {img[1]:7.1f})")

    print(f"    Field : ({fld[0]:6.2f}, {fld[1]:6.2f})")

    print()

