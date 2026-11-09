import numpy as np
import cv2
import math

from zones import ZONE_DEFINITIONS

from field_geometry import get_fence_distance

def estimate_similarity(image_pts, field_pts):
    """
    Estimate a similarity transform from image pixels
    to field coordinates.
    """
    if len(image_pts) < 2:
        print("Need at least 2 landmarks for calibration.")
        return None

    transform, inliers = cv2.estimateAffinePartial2D(
        image_pts,
        field_pts
    )

    if transform is None:
        return None
    if not np.isfinite(transform).all():
        return None
    
    return transform

def estimate_affine(image_pts, field_pts):
    """
    Estimate full affine transform (allows shear).
    """

    if len(image_pts) < 3:
        print("Need at least 3 points for affine transform.")
        return None

    transform, inliers = cv2.estimateAffine2D(
        image_pts,
        field_pts
    )
    if transform is None:
        return None
    if not np.isfinite(transform).all():
        return None
    
    return transform

def estimate_homography(image_pts, field_pts):
    if len(image_pts) < 4:
        return None

    H, mask = cv2.findHomography(image_pts, field_pts, method=0)

    if H is None:
        return None
    if not np.isfinite(H).all():
        return None
    
    return H

def compute_transformation(image_pts, field_pts):

    num_landmarks = len(image_pts)

    if num_landmarks < 2:
        return None, None, np.inf

    elif num_landmarks == 2:
        model = "similarity"
        transform = estimate_similarity(image_pts, field_pts)

    elif num_landmarks == 3:
        model = "affine"
        transform = estimate_affine(image_pts, field_pts)

    else:
        model = "homography"
        transform = estimate_homography(image_pts, field_pts)

    landmark_error = mean_landmark_error(
        transform,
        image_pts,
        field_pts
    )

    return transform, model, landmark_error

def transform_ball(transforms, ball_pixel):

    if transforms is None:
        return None

    if ball_pixel is None:
        return None

    return project_ball(transforms, ball_pixel)

def transform_points(transform, points):
    """
    Transform image points into field coordinates.
    """

    if transform is None:
        return None

    points = np.array(points, dtype=np.float32)

    points = points.reshape(-1, 1, 2)

    if transform.shape == (3,3):
        transformed = cv2.perspectiveTransform(points, transform)
    else:
        transformed = cv2.transform(points, transform)

    return transformed.reshape(-1, 2)

def project_ball(transforms, ball_pixel):
    """
    Convert the ball from image pixels to field coordinates.

    Parameters
    ----------
    transform : ndarray
        Similarity, affine, or homography transform.

    ball_pixel : tuple
        (x, y) pixel coordinates of the ball.

    Returns
    -------
    ndarray
        (x, y) ball location in field coordinates.
    """

    projections = {}

    for model_name, transform in transforms.items():
        if transform is None:
            continue
        
        projected = apply_transform(
            ball_pixel,
            transform,
            model_name
        )

        projections[model_name] = projected

    return projections

# helper function
def apply_transform(point, transform, model_name):
    pt = np.array(
        [[point]],
        dtype=np.float32
    )

    if model_name in ["similarity", "affine"]:
        projected = cv2.transform(
            pt,
            transform
        )
    elif model_name == "homography":
        projected = cv2.perspectiveTransform(
            pt,
            transform
        )
    else:
        return None
    
    return tuple(projected[0][0])

def distance(point1, point2):
    """
    Euclidean distance between two field points.
    """

    return np.linalg.norm(point1 - point2)

def bearing_angle(point):
    """
    Compute the bearing angle (degrees)
    from home plate.
    """

    x = point[0]
    y = point[1]

    angle = np.degrees(
        np.arctan2(y, x)
    )

    return angle

def assign_zone(distance, bearing):
    """
    Return the zone containing the ball.
    """

    for zone_name, zone in ZONE_DEFINITIONS.items():

        if (
            zone["u_min"] <= distance <= zone["u_max"]
            and
            zone["theta_min"] <= bearing <= zone["theta_max"]
        ):

            return zone_name

    return None

def normalize_distance(distance_from_home, bearing):
    """
    Convert distance in feet to normalized field distance.
    """

    fence = get_fence_distance(bearing)

    return distance_from_home / fence

def mean_landmark_error(transform, image_pts, field_pts):
    if transform is None:
        return float("inf")

    projected = transform_points(transform, image_pts)

    errors = np.linalg.norm(projected - field_pts, axis=1)

    return float(np.mean(errors))

def compute_transformations(image_pts, field_pts):

    transformations = {
        "similarity": None,
        "affine": None,
        "homography": None,
    }

    num_landmarks = len(image_pts)

    if num_landmarks >= 2:
        similarity = estimate_similarity(image_pts, field_pts)
        transformations["similarity"] = similarity

    if num_landmarks >= 3:
        affine = estimate_affine(image_pts, field_pts)
        transformations["affine"] = affine

    if num_landmarks >= 4:
        homography = estimate_homography(image_pts, field_pts)
        transformations["homography"] = homography

    return transformations

def get_fence_distance(bearing_deg):
    """
    Fence distance in feet for a bearing angle in degrees.
    """

    theta = math.radians(bearing_deg)

    return fence_distance_radians(theta)

def fence_distance_radians(theta):
    """
    Fence distance (feet) for a bearing angle in radians.
    """
    
    FOUL_LINE_DISTANCE = 195.0
    CENTER_FIELD_DISTANCE = 220.0
    
    return (
        FOUL_LINE_DISTANCE
        + (CENTER_FIELD_DISTANCE - FOUL_LINE_DISTANCE)
        * math.sin(2 * theta)
    )

def apply_inverse_transform(field_point, transform, model_name):

    pt = np.array(
        [[point]],
        dtype=np.float32
    )

    if model_name in ["similarity", "affine"]:
        # similarity/affine
        inverse_transform = cv2.invertAffineTransform(transform)
        projected = cv2.transform(
            pt,
            inverse_transform
        )

    elif model_name == "homography":
        # homography
        inverse_transform = np.linalg.inv(transform)
        projected = cv2.perspectiveTransform(
            pt,
            inverse_transform
        )
    else:
        return None

    return tuple(projected[0][0])

def ball_reprojection_score(ball_pixel, transform, model_name):
    projected = cv2.transform(
        pt,
        inverse_transform
    )

    # gather necessary data
    x = field_point[0]
    y = field_point[1]

    distance = np.sqrt(x**2 + y**2)
    bearing = np.degrees(
        np.arctan2(y, x)
    )

    fence = get_fence_distance(bearing)

    # distance score (within fence)
    if distance <= fence:
        distance_score = 100
    else:
        penalty = distance - fence
        distance_score = max(
            0,
            100 - penalty
        )

    # bearing score (within fair territory)   
    if -2 <= bearing <= 92:
        bearing_score = 100
    else:
        bearing_score = 0

    return {
        "field_point": field_point,
        "distance": distance,
        "bearing": bearing,
        "distance_score": distance_score,
        "bearing_score": bearing_score
    }

def compare_transformations(image_pts, field_pts, ball_pixel, field_geometry):
    transforms = compute_transformations(image_pts, field_pts)
    results = []
    for model_name, transform in transforms.items():
        if transform is None:
            continue
        ball_field = project_ball(transforms, ball_pixel)

        # scoring elements:
            #(1) inside playing field
            #(2) inside fence (based on distance)
            #(3) inside fair territory (based on bearing angle)
    # distance score
    distance_ft = math.sqrt(
        ball_x_ft**2 +
        ball_y_ft**2
    )
    bearing = bearing_angle(point)
    fence_distance = get_fence_distance(bearing)
    difference = fence_distance - distance_ft
    if difference >= 0:
        fence_score = 1.0
    else:
        fence_score = 0.0
    if bearing < 0 or bearing > 90:
        foul_score = 0.0
    else:
        foul_score = 1.0

    observed_x = ball_pixel_x
    observed_y = ball_pixel_y
    
    pixel_error = np.sqrt(
        (predicted_x - observed_x)**2 +
        (predicted_y - observed_y)**2
    )
    ball_reprojection_score = score = 1 / (1 + pixel_error / 20) # 20 pixels is the baseline
    # that is the farthest away that I will allow the prediction and still accept it as truth
