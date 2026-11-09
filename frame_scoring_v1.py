import numpy as np

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
)

def compute_frame_score(image_points):
    """
    Scores how usable a frame is for calibration.
    Higher = better geometry reliability.
    """

    if len(image_points) < 2:
        return 0.0

    pts = np.array(list(image_points.values()), dtype=np.float32)

    # 1. Number of landmarks (stability proxy)
    n_score = len(pts)

    # 2. Spatial spread (important for calibration quality)
    spread_x = np.std(pts[:, 0])
    spread_y = np.std(pts[:, 1])
    spread_score = spread_x + spread_y

    # 3. Bounding box size (penalize clustered points)
    bbox_area = (
        np.max(pts[:, 0]) - np.min(pts[:, 0])
    ) * (
        np.max(pts[:, 1]) - np.min(pts[:, 1])
    )

    # Final score
    return n_score * spread_score + 0.001 * bbox_area

def choose_model(image_points):
    n = len(image_points)

    if n >= 4:
        return "homography"

    elif n == 3:
        score = compute_frame_score(image_points)

        if score > 5000:   # you will tune this
            return "affine"
        else:
            return "similarity"

    elif n == 2:
        return "similarity"

    else:
        return None

def compute_transform(image_pts, field_pts, model_type):
    if model_type == "similarity":
        return estimate_similarity(image_pts, field_pts)

    elif model_type == "affine":
        return estimate_affine(image_pts, field_pts)

    elif model_type == "homography":
        return estimate_homography(image_pts, field_pts)

    return None
