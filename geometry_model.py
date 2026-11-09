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

import pandas as pd

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
