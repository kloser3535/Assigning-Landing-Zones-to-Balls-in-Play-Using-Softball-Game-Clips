import math
import numpy as np

FOUL_LINE_DISTANCE = 195.0
CENTER_FIELD_DISTANCE = 220.0

INFIELD_FOUL = 88.0
INFIELD_CENTER = 100.0

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

def fence_distance_radians(theta):
    """
    Fence distance (feet) for a bearing angle in radians.
    """

    return (
        FOUL_LINE_DISTANCE
        + (CENTER_FIELD_DISTANCE - FOUL_LINE_DISTANCE)
        * math.sin(2 * theta)
    )

def get_fence_distance(bearing_deg):
    """
    Fence distance in feet for a bearing angle in degrees.
    """

    theta = math.radians(bearing_deg)

    return fence_distance_radians(theta)

def infield_radius(theta):

    return (
        INFIELD_FOUL
        + (INFIELD_CENTER - INFIELD_FOUL)
        * math.sin(2 * theta)
    )
