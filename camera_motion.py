import cv2
import numpy as np

def find_features(frame):
    """
    Find good image features to track.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # the features that the OpenCV model is looking for are CORNERS (according to CHAT)
    features = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=10
    )

    return features

def track_features(previous_frame, current_frame):
    """
    Track image features from one frame to the next.
    """

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    features = find_features(previous_frame)

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        features,
        None
    )

    good_old = features[status.flatten() == 1]

    good_new = next_points[status.flatten() == 1]

    good_old = good_old.reshape(-1, 2)
    good_new = good_new.reshape(-1, 2)

    return good_old, good_new # good_old[i] and good_new[i] are the same feature in back-to-back frames

def estimate_camera_transform(old_points, new_points): # we are using an AFFINE TRANSFORMATION to determine how much the camera has moved between frames
    # I can speculate how much the camera moved based on the arrow visuals
    # However, I eventually want the computer to be able to determine how much the camera moved
    # And to do this, I have to quantify the camera movement which can be done with the AFFINE TRANSFORMATION and OTHER FUNCTIONS/COMPUTATIONS
    """
    Estimate the camera motion between two frames.
    """
    if len(old_points) < 3:
        return None
    
    transform, inliers = cv2.estimateAffinePartial2D(
        old_points,
        new_points
    )
    
    return transform

def summarize_motion(transform, old_points): # this function helps us compute movement using data that was produced by the affine matrix in the function above

    a = transform[0,0]
    b = transform[0,1]
    c = transform[1,0]
    d = transform[1,1]

    tx = transform[0,2]
    ty = transform[1,2]

    scale = np.sqrt(a*a + c*c)

    rotation = np.degrees(
        np.arctan2(c,a)
    )

    motion = {}
    motion["tracked_features"] = len(old_points)

    motion["pan_x"] = tx
    motion["pan_y"] = ty
    motion["scale"] = scale
    motion["rotation"] = rotation

    return motion

def draw_motion_vectors(frame, old_points, new_points): # we used somehting similar in the camera_motion_test script to visualize the outputs of our functions
    """
    Draw optical-flow vectors on the frame.
    """

    output = frame.copy()

    for old, new in zip(old_points, new_points):

        x1, y1 = old.ravel()
        x2, y2 = new.ravel()

        x1 = int(x1)
        y1 = int(y1)

        x2 = int(x2)
        y2 = int(y2)

        #
        # Motion line
        #

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1
        )

        #
        # New position
        #

        cv2.circle(
            output,
            (x2, y2),
            2,
            (0,255,0),
            -1
        )

    return output

def draw_motion_summary(frame, motion):
    """
    Draw camera-motion statistics.
    """

    x = 20
    y = 30

    line_height = 30

    pan_mag = np.hypot(
        motion["pan_x"],
        motion["pan_y"]
    )

    if pan_mag < 2:
        pan_color = (0,255,0)      # little movement

    elif pan_mag < 10:
        pan_color = (0,255,255)    # moderate

    else:
        pan_color = (0,0,255)      # large movement

    cv2.putText(
        frame,
        f"Frame: {motion['frame']}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"Pan X : {motion['pan_x']:.2f} px",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        pan_color,
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"Pan Y : {motion['pan_y']:.2f} px",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        pan_color,
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"Zoom : {motion['scale']:.4f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    y += line_height

    cv2.putText(
        frame,
        f"Rotation : {motion['rotation']:.2f} deg",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    return frame
