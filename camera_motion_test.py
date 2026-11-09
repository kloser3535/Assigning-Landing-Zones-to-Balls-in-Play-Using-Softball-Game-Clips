import cv2
import pandas as pd
import numpy as np

from camera_motion import find_features, track_features

VIDEO_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos\FOR_INF_H.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)

FRAME = 200

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)

success, frame1 = cap.read()

success, frame2 = cap.read()

old_points, new_points = track_features(
    frame1,
    frame2
)

for old, new in zip(old_points, new_points):

    x1, y1 = old.ravel()
    x2, y2 = new.ravel()

    cv2.circle(
        frame2,
        (int(x2), int(y2)),
        3,
        (0,255,0),
        -1
    )

    cv2.line(
        frame2,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (255,0,0),
        1
    )

cv2.imshow(
    "Camera Motion Test",
    frame2
)

cv2.waitKey(0)
    
cap.release()

cv2.destroyAllWindows()
