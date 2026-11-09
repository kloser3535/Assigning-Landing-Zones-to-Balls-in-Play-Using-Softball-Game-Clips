# inputs
# generate timeline csv for ball coordinates
# master_csv file for first touch
# combined_csv_files_best_landmarks - this should be all I need...
# prepare_landmarks = only keep best landmarks

# imports
import pandas as pd
import numpy as np
import cv2
import os
from itertools import combinations

from field_geometry import FIELD_POINTS
from calibration_models import compute_transformation

# weights
MAX_DISTANCE = 20
REPROJECTION_SCALE = 20

# geometry scpre v2 constants
REPROJECTION_WEIGHT = 38
COVERAGE_WEIGHT = 23
DETERMINANT_WEIGHT = 3
STRETCH_WEIGHT = 3
CONFIGURATION_WEIGHT = 28
EDGE_WEIGHT = 5
MAX_LANDMARKS = 5
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
EDGE_THRESHOLD = 10
SPACING_PENALTY_POINTS = 20

# ball score constants
CONFIDENCE_WEIGHT = 0.25
DISTANCE_WEIGHT = 0.40
BB_WEIGHT = 0.35

# ranking score constants
GEOMETRY_RANK_WEIGHT = 0.55
BALL_RANK_WEIGHT = 0.20
CONF_RANK_WEIGHT = 0.10
TIMING_RANK_WEIGHT = 0.15

#############################
MIN_LANDMARK_SPACING = {
    ("pitching_rubber", "first_base"): 100,
    ("pitching_rubber", "second_base"): 30,
    ("pitching_rubber", "third_base"): 100,
    ("pitching_rubber", "home_plate"): 100,

    ("home_plate", "second_base"): 175,

    ("home_plate", "first_base"): 430,
    ("home_plate", "third_base"): 430,
    ("third_base", "second_base"): 410,
    ("first_base", "second_base"): 410,

    ("first_base", "third_base"): 800
    
}

PAIR_WEIGHTS = {

    tuple(sorted(("home_plate","pitching_rubber"))): 0.40,
    tuple(sorted(("home_plate","first_base"))): 0.90,
    tuple(sorted(("home_plate","third_base"))): 0.90,
    tuple(sorted(("home_plate","second_base"))): 1.00,
    tuple(sorted(("pitching_rubber","second_base"))): 0.40,
    tuple(sorted(("first_base","second_base"))): 0.80,
    tuple(sorted(("third_base","second_base"))): 0.80,
    tuple(sorted(("first_base","third_base"))): 0.85,
    tuple(sorted(("pitching_rubber","first_base"))): 0.40,
    tuple(sorted(("pitching_rubber","third_base"))): 0.40,

}

LANDMARK_CONFIGURATION_SCORES = {
# full field
    frozenset([
        "home_plate",
        "first_base",
        "second_base",
        "third_base",
        "pitching_rubber"
    ]): 1.00,

# 4 landmarks    
    frozenset([
        "home_plate",
        "first_base",
        "second_base",
        "third_base"
    ]): 1.00,
    
    frozenset([
        "home_plate",
        "first_base",
        "second_base",
        "pitching_rubber"
    ]): 0.90,

    frozenset([
        "home_plate",
        "first_base",
        "third_base",
        "pitching_rubber"
    ]): 0.90,
        frozenset([
        "home_plate",
        "third_base",
        "second_base",
        "pitching_rubber"
    ]): 0.90,
        frozenset([
        "third_base",
        "first_base",
        "second_base",
        "pitching_rubber"
    ]): 0.95,
    
    # Excellent triangles
    frozenset([
        "home_plate",
        "first_base",
        "second_base"
    ]): 0.90,

    frozenset([
        "home_plate",
        "second_base",
        "third_base"
    ]): 0.90,

    frozenset([
        "first_base",
        "second_base",
        "third_base"
    ]): 0.85,

    # poor triangles
    frozenset([
        "home_plate",
        "first_base",
        "pitching_rubber"
    ]): 0.80,
    
    frozenset([
        "pitching_rubber",
        "first_base",
        "second_base"
    ]): 0.70,
    frozenset([
        "home_plate",
        "pitching_rubber",
        "third_base"
    ]): 0.80,
    frozenset([
        "pitching_rubber",
        "third_base",
        "second_base"
    ]): 0.70,

    # Good pairs
    frozenset([
        "first_base",
        "second_base"
    ]): 0.65,

    frozenset([
        "second_base",
        "third_base"
    ]): 0.65,

    frozenset([
        "home_plate",
        "first_base"
    ]): 0.75,

    frozenset([
        "home_plate",
        "third_base"
    ]): 0.75,


    # The trio you mentioned
    frozenset([
        "home_plate",
        "pitching_rubber",
        "second_base"
    ]): 0.50,


    # Weak pairs
    frozenset([
        "home_plate",
        "second_base"
    ]): 0.35,

    frozenset([
        "first_base",
        "third_base"
    ]): 0.25,


    # Bad rubber pairs
    frozenset([
        "home_plate",
        "pitching_rubber"
    ]): 0.05,

    frozenset([
        "first_base",
        "pitching_rubber"
    ]): 0.05,

    frozenset([
        "third_base",
        "pitching_rubber"
    ]): 0.05,
    
    frozenset([
        "second_base",
        "pitching_rubber"
    ]): 0.05,
    
}

# functions

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

def rank_frames(search_df, quality_level, use_timing):

    candidates = search_df[
        (search_df["quality_level"] == quality_level)
        &
        (search_df["detected"] == True)
    ].copy()
    
    if candidates.empty:
        return candidates
    if use_timing:

        candidates["ranking_score"] = (
            candidates["base_ranking_score"]
            +
            TIMING_RANK_WEIGHT * candidates["timing_score"]
        )
    else:
        candidates["ranking_score"] = (
            candidates["base_ranking_score"]
        )

    
    candidates = candidates.sort_values(
        by=[
            "ranking_score"
        ],
        ascending=False
    )
    
    return candidates

# helper function for coverage score
def triangle_area(p1, p2, p3):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return abs(
        x1*(y2-y3)
        + x2*(y3-y1)
        + x3*(y1-y2)
    ) / 2

def get_field_axis(image_points):

    # Best case
    if (
        "home_plate" in image_points and
        "second_base" in image_points
    ):
        return (
            image_points["home_plate"],
            image_points["second_base"]
        )

    # Next best
    if (
        "pitching_rubber" in image_points and
        "second_base" in image_points
    ):
        return (
            image_points["pitching_rubber"],
            image_points["second_base"]
        )

    # Last option
    if (
        "home_plate" in image_points and
        "pitching_rubber" in image_points
    ):
        return (
            image_points["home_plate"],
            image_points["pitching_rubber"]
        )

    return None

def score_landmark_configuration(hypothesis): # function has access to name, reason and points
    # takes a possible landmakr configuration and
    # (a) determines if still passes spacing and orientation test
    # (b) computes geometry score
    # (c) returns geometry score and other relevant info
        #so that another function can compare all of the relevant hypotheses for this frame

    name = hypothesis["name"]
    reason = hypothesis["reason"]
    image_points = hypothesis["points"]

    landmark_points = []

    for landmark, (x,y) in image_points.items():

        landmark_points.append(
            (
                landmark,
                x,
                y
            )
        )

    result = {
        "name": name,
        "reason": reason,
        "valid": False, # valid = false means that we will reject the frame from being a calibration candidate
        "image_points": image_points,
        "geometry_score_v2": 0,
        "orientation_pass": False,
        "spacing_pass": False,
        "spacing_violations": np.nan,
        "reprojection_score": 0,
        "coverage_score": 0,
        "determinant_score": 0,
        "stretch_score": 0,
        "configuration_score": 0,
        "edge_score": 0,
        "landmark_count": len(image_points)
    }

    # spacing evaluation
    spacing_violations = find_spacing_violations(image_points)
    result["spacing_violations"] = spacing_violations
    spacing_pass = (len(spacing_violations) == 0)
    result["spacing_pass"] = spacing_pass

    # orientation evaluation
    orientation_score = 1.0
    axis = get_field_axis(image_points)

    if axis is None:
        camera_angle = "unknown"
    else:
        (x1, y1), (x2, y2) = axis
        if y1 > y2:
            camera_angle = "behind_home"
        else:
            camera_angle = "outfield"

    center_xs = []
    for landmark in ["home_plate", "pitching_rubber", "second_base"]:

        if landmark in image_points:
            center_xs.append(image_points[landmark][0])

    if len(center_xs) > 0:
        center_axis_x = np.mean(center_xs)
    else:
        center_axis_x = None

    if ("first_base" in image_points and center_axis_x is not None):
        fb_x = image_points["first_base"][0]
        
        if camera_angle == "behind_home":
            if fb_x < center_axis_x:
                orientation_score = 0.0

        if camera_angle == "outfield":
            if fb_x > center_axis_x:
                orientation_score = 0.0
            
    if ("third_base" in image_points and center_axis_x is not None):
        tb_x = image_points["third_base"][0]
        
        if camera_angle == "behind_home":
            if tb_x > center_axis_x:
                orientation_score = 0.0
                
        elif camera_angle == "outfield":    
            if tb_x < center_axis_x:
                orientation_score = 0.0

    orientation_pass = (orientation_score == 1.0)
    result["orientation_pass"] = orientation_pass

    # skip to end of function if hypotheses does not pass spacing and orientation
    if not spacing_pass:
        return result
    if not orientation_pass:
        return result
    #______________________________________________________
    # start geometry_score_v2 calculation
    #______________________________________________________
    
    # (1) configuration_score
    visible_classes = set(image_points.keys())
    configuration_score = LANDMARK_CONFIGURATION_SCORES.get(
        frozenset(visible_classes),
        0.50
    )
    result["configuration_score"] = configuration_score

    # build correspondences
    image_pts, field_pts, names = build_correspondences(image_points)

    # compute transform and reject candidates with less than two landmarks
    if len(image_pts) >= 2:
        transform, model_type, transform_geometry_error = compute_transformation(
            image_pts,
            field_pts
        )
    else:
        transform = None
        model_type = None

    if transform is None:
        return result
    else:
        # (2) reprojection_score
        reprojection_score = np.exp(
            -transform_geometry_error / REPROJECTION_SCALE
        )
        
        # compute A
        if model_type in ["similarity", "affine"]:
            A = transform[:, :2]

        elif model_type == "homography":
            A = transform[:2, :2]

        else:
            A = None

        if A is None:
            return result
        if not np.isfinite(A).all():
            print("Invalid transformation matrix.")
            determinant_score = 0
            stretch_score = 0
        else:
            # (3) determinant_score    
            det = np.linalg.det(A)
            determinant_penalty = abs(det - 1)
            determinant_score = 1 / (1 + determinant_penalty)

            # (4) stretch_score
            u,s,v = np.linalg.svd(A)
            stretch_ratio = s[0]/s[1]
            stretch_penalty = abs(stretch_ratio - 1)
            stretch_score = 1 / (1 + stretch_penalty) # smooth decay for all observations over all videos because no max/min

        # (5) coverage_score
        if len(image_pts) == 2:
            landmark_names = list(image_points.keys())
            pts = list(image_points.values())

            landmark1 = landmark_names[0]
            landmark2 = landmark_names[1]
            
            (x1, y1) = pts[0]
            (x2, y2) = pts[1]

            pair = tuple(sorted([landmark1, landmark2]))
            pair_weight = PAIR_WEIGHTS.get(pair, 1.0)
            
            distance = np.sqrt(
                (x2 - x1)**2 +
                (y2 - y1)**2
            )

            image_diagonal = np.sqrt(
                IMAGE_WIDTH**2 +
                IMAGE_HEIGHT**2
            )

            distance_ratio = distance / image_diagonal

            distance_score = min(distance_ratio / 0.60, 1.0) # between 1 and 0, 0.5 (half of the diagonal) is my target coverage ratio
            
            coverage_score = min(distance_score * pair_weight, 1.0)
            # pair_weight in a dictionary at the top based on how strong the pair is for calibrating the ball based on what I have seen in the visual debugger
            # pair_weight helps me to pull down value of pitching rubber since I have decided that it is not a very good landmark
        elif len(image_pts) == 3:
            # triangle area and try to scale it on a scale of 0 to 1 (normalize? to all videos or to all frames in this one vide - have to ask more about scaling/logic)
            pts = list(image_points.values())
            coverage_area = triangle_area(
                pts[0],
                pts[1],
                pts[2]
            )
            coverage_ratio = coverage_area / (IMAGE_WIDTH * IMAGE_HEIGHT)
            
            coverage_score = min(coverage_ratio / 0.10, 1.0)
        elif len(image_pts) >=4:
            # convex hull (score the same way as triangle area but only compare to frames with 4/5 landmarks on them)
            points = np.array(
                [[x, y] for _, x, y in landmark_points],
                dtype=np.float32
            )
            # compute convex hull/area
            hull = cv2.convexHull(points)
            hull_area = cv2.contourArea(hull)
            
            image_area = IMAGE_WIDTH * IMAGE_HEIGHT
            coverage_ratio = hull_area / image_area

            if len(image_pts) == 4:
                coverage_score = min(coverage_ratio / 0.15, 1.0)

            if len(image_pts) == 5:
                coverage_score = min(coverage_ratio / 0.20, 1.0)                    

        # (6) edge_score
        # penalty calculation
        edge_count = 0

        for landmark, x, y in landmark_points:
            left = x
            right = IMAGE_WIDTH - x
            top = y
            bottom = IMAGE_HEIGHT - y

            edge_distance = min(
                left,
                right,
                top,
                bottom
            )

            if edge_distance < EDGE_THRESHOLD:
                edge_count += 1
        # compute score        
        edge_score = max(
            0,
            1 - edge_count * 0.20
        )

        # compute geometry_score_v2
        geometry_score_v2 = (
            REPROJECTION_WEIGHT * reprojection_score +
            COVERAGE_WEIGHT * coverage_score +
            DETERMINANT_WEIGHT * determinant_score +
            STRETCH_WEIGHT * stretch_score +
            CONFIGURATION_WEIGHT * configuration_score +
            EDGE_WEIGHT * edge_score
        )

    result["geometry_score_v2"] = geometry_score_v2
    result["reprojection_score"] = reprojection_score
    result["coverage_score"] = coverage_score
    result["stretch_score"] = stretch_score
    result["determinant_score"] = determinant_score
    result["edge_score"] = edge_score
    result["valid"] = True

    return result
        
def find_spacing_violations(image_points):

    violations=[]

    landmarks=list(image_points.keys())

    for i in range(len(landmarks)):

        for j in range(i+1,len(landmarks)):

            l1=landmarks[i]
            l2=landmarks[j]

            p1=image_points[l1]
            p2=image_points[l2]

            distance=np.sqrt(
                (p2[0]-p1[0])**2+
                (p2[1]-p1[1])**2
            )

            pair=(l1,l2)
            reverse=(l2,l1)

            if pair in MIN_LANDMARK_SPACING:
                minimum=MIN_LANDMARK_SPACING[pair]

            elif reverse in MIN_LANDMARK_SPACING:
                minimum=MIN_LANDMARK_SPACING[reverse]

            else:
                continue


            if distance < minimum:

                violations.append(
                    {
                    "landmark1":l1,
                    "landmark2":l2,
                    "distance":distance,
                    "minimum":minimum
                    }
                )

    return violations

def generate_landmark_hypotheses(
    image_points,
    evaluation_result
):
    """
    Generate possible landmark corrections.

    Returns:
        list of dictionaries:
        [
            {
                "name": hypothesis name,
                "points": modified image_points
            }
        ]
    """

    spacing_failed = not evaluation_result["spacing_pass"]
    orientation_failed = not evaluation_result["orientation_pass"]
    spacing_violations = evaluation_result.get(
        "spacing_violations",
        []
    )

    hypotheses = []

    # --------------------------------------------------
    # 1. Always include original detection
    # --------------------------------------------------
    hypotheses.append(
        {
        "name": "original",
        "reason": "original",
        "points": image_points.copy()
        }
    )
    # --------------------------------------------------
    # 2. Swap first and third base
    # --------------------------------------------------

    if orientation_failed:
        if (
            "first_base" in image_points
            and
            "third_base" in image_points
        ):

            swapped = image_points.copy()

            swapped["first_base"] = image_points["third_base"]

            swapped["third_base"] = image_points["first_base"]


            hypotheses.append(
                {
                    "name": "swap_first_third",
                    "reason": "orientation",
                    "points": swapped
                }
            )
        elif (
            "first_base" in image_points
            and
            "third_base" not in image_points
        ):
            relabel = image_points.copy()
            coords = relabel["first_base"]
            del relabel["first_base"]
            relabel["third_base"] = coords

            hypotheses.append(
                {
                    "name": "relabel_first_to_third",
                    "reason": "orientation",
                    "points": relabel
                }
            )
        elif (
            "third_base" in image_points
            and
            "first_base" not in image_points
        ):
            relabel = image_points.copy()
            coords = relabel["third_base"]
            del relabel["third_base"]
            relabel["first_base"] = coords

            hypotheses.append(
                {
                    "name": "relabel_third_to_first",
                    "reason": "orientation",
                    "points": relabel
                }
            )
        

    # --------------------------------------------------
    # 3. Remove individual landmarks
    # --------------------------------------------------
    if spacing_failed:

        landmarks_to_remove = set()

        for violation in spacing_violations:
            landmarks_to_remove.add(
                violation["landmark1"]
            )
            landmarks_to_remove.add(
                violation["landmark2"]
            )

        for landmark in landmarks_to_remove:

            reduced = image_points.copy()

            del reduced[landmark]


            # must keep enough landmarks
            if len(reduced) >= 2:

                hypotheses.append(
                    {
                        "name": f"remove_{landmark}",
                        "reason": "spacing",
                        "points": reduced
                    }
                )


    return hypotheses

def compare_landmark_hypotheses(hypotheses):

    valid_results = []

    all_results = []

    for hypothesis in hypotheses:

        result = score_landmark_configuration(
            hypothesis
        )

        result["hypothesis_reason"] = hypothesis["reason"]

        all_results.append(result)

        if result["valid"]:
            valid_results.append(result)


    # no hypothesis survived
    if len(valid_results) == 0:

        return {
            "valid": False,
            "reason": "No valid landmark configuration",
            "all_results": all_results
        }


    best_result = max(
        valid_results,
        key=lambda x: x["geometry_score_v2"]
    )


    best_result["all_results"] = all_results
    best_result["selected_hypothesis"] = best_result["name"]

    best_result["all_hypothesis_summary"] = [
        {
            "name": r["name"],
            "reason": r["reason"],
            "valid": r["valid"],
            "geometry_score": r["geometry_score_v2"]
        }
        for r in all_results
    ]

    return best_result

def evaluate_original_configuration(image_points):
    
    result = {
        "spacing_pass": True,
        "orientation_pass": True,
        "orientation_known": False,
        "needs_hypothesis": False,
        "reasons": []
    }

    # spacing test
    spacing_violations = find_spacing_violations(
        image_points
    )

    if len(spacing_violations) > 0:

        result["spacing_pass"] = False

        result["needs_hypothesis"] = True

        result["reasons"].append(
            "spacing"
        )

        result["spacing_violations"] = spacing_violations

    # orientation test
    orientation_score = None
    axis = get_field_axis(image_points)

    if axis is None:
        camera_angle = "unknown"
        result["orientation_known"] = False
    else:
        result["orientation_known"] = True
        orientation_score = 1.0
        (x1, y1), (x2, y2) = axis
        if y1 > y2:
            camera_angle = "behind_home"
        else:
            camera_angle = "outfield"

    center_xs = []
    for landmark in ["home_plate", "pitching_rubber", "second_base"]:

        if landmark in image_points:
            center_xs.append(image_points[landmark][0])

    if len(center_xs) > 0:
        center_axis_x = np.mean(center_xs)
    else:
        center_axis_x = None

    if ("first_base" in image_points and center_axis_x is not None):
        fb_x = image_points["first_base"][0]
        
        if camera_angle == "behind_home":
            if fb_x < center_axis_x:
                orientation_score = 0.0

        if camera_angle == "outfield":
            if fb_x > center_axis_x:
                orientation_score = 0.0
            
    if ("third_base" in image_points and center_axis_x is not None):
        tb_x = image_points["third_base"][0]
        
        if camera_angle == "behind_home":
            if tb_x > center_axis_x:
                orientation_score = 0.0
                
        elif camera_angle == "outfield":    
            if tb_x < center_axis_x:
                orientation_score = 0.0

    if (
        result["orientation_known"]
        and orientation_score == 0
    ):

        result["orientation_pass"] = False
        result["needs_hypothesis"] = True
        result["reasons"].append("orientation")

    return result

def get_frame_landmarks(frame_df):

    image_points = {}

    landmarks = [
        "home_plate",
        "first_base",
        "second_base",
        "third_base",
        "pitching_rubber"
    ]

    row = frame_df.iloc[0]

    for landmark in landmarks:

        x_col = landmark + "_center_x"
        y_col = landmark + "_center_y"

        x = row[x_col]
        y = row[y_col]

        if pd.notna(x) and pd.notna(y):

            image_points[landmark] = (
                float(x),
                float(y)
            )

    return image_points

def compute_ranking_score(
    geometry_score_v2,
    ball_score,
    avg_landmark_conf,
    timing_score
):

    if pd.isna(timing_score):
        timing_score = 0


    ranking_score = (
        GEOMETRY_RANK_WEIGHT * geometry_score_v2
        +
        BALL_RANK_WEIGHT * ball_score
        +
        CONF_RANK_WEIGHT * (avg_landmark_conf * 100)
        +
        TIMING_RANK_WEIGHT * timing_score
    )


    return ranking_score

# input/output files
BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

MASTER_DETECTION_FOLDER = fr"{BASE_FOLDER}\master_detection_csv_files"
FIRST_TOUCH_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\first_touch_summary.csv"
OUTPUT_FOLDER = fr"{BASE_FOLDER}\calibration_score_csv_files"
CANDIDATE_FOLDER = fr"{BASE_FOLDER}\calibration_candidates_topten_csv_files"

first_touch_df = pd.read_csv(FIRST_TOUCH_CSV)

# processing video
def process_video(video_name, master_csv, candidate_csv, output_csv, first_touch_df):

    df = pd.read_csv(master_csv)
    
    match = first_touch_df[
        first_touch_df["video"] == video_name
    ]

    if match.empty:
        print(f"No first touch found for {video_name}")
        return None

    first_touch_frame = match.iloc[0]["predicted_frame"]

    frames = sorted(df["frame"].unique())

    score_table = pd.DataFrame({
        "frame": frames
    })
    
    print()
    print("Number of frames:", len(frames))
   
    score_table["camera_state"] = ""
    score_table["detected"] = False
    score_table["searchable"] = False
    
    score_table["landmark_count"] = 0
    score_table["avg_landmark_conf"] = np.nan
    
    score_table["ball_conf"] = np.nan
    score_table["area_score"] = np.nan
    score_table["distance_score"] = np.nan

    score_table["distance_from_first_touch"] = np.nan

    score_table["ball_score"] = np.nan
    score_table["timing_score"] = np.nan

    score_table["reprojection_score"] = np.nan
    score_table["coverage_score"] = np.nan
    score_table["determinant_score"] = np.nan
    score_table["stretch_score"] = np.nan
    score_table["configuration_score"] = np.nan
    score_table["geometry_score_v2"] = np.nan

    score_table["quality_level"] = ""
    score_table["base_ranking_score"] = np.nan
    score_table["ranking_score"] = np.nan
    score_table["search_region"] = ""

    score_table["selected_hypothesis"] = ""
    score_table["hypothesis_reason"] = ""
    score_table["original_spacing_pass"] = False
    score_table["original_orientation_pass"] = False
    score_table["original_needs_hypothesis"] = False

    score_table["hypothesis_count"] = 0
    score_table["all_hypothesis_scores"] = ""
    score_table["final_landmarks"] = ""
    
    previous_ball_area = None
    
    previous_ball_x = None
    previous_ball_y = None

    hard_cut_start_frame = None
    large_pan_start_frame = None
    
    for frame_index, frame in enumerate(frames):

        frame_df = df[
            df["frame"] == frame
        ]
        row = frame_df.iloc[0]

        camera_state = frame_df.iloc[0]["camera_state"]
        score_table.loc[frame_index, "camera_state"] = camera_state

        window_start = first_touch_frame - MAX_DISTANCE
        window_end = first_touch_frame + MAX_DISTANCE
        if (
            hard_cut_start_frame is None
            and camera_state == "Following Ball"
        ):
            hard_cut_start_frame = frame

        if (
            large_pan_start_frame is None
            and camera_state == "Large Pan"
        ):
            large_pan_start_frame = frame

        if camera_state in ["Pitch View", "Hard Cut", "Unknown"]:
            searchable = False
        else:
            searchable = True
            
        score_table.loc[frame_index, "searchable"] = searchable

        if not searchable:
            region = "pitch"
        elif frame < window_start:
            region = "before"
        elif frame <= window_end:
            region = "window"
        else:
            region = "after"

        score_table.loc[frame_index, "search_region"] = region

        # landmark stats
        
        # -----------------------------------
        # 1. Extract landmarks
        # -----------------------------------

        image_points = get_frame_landmarks(frame_df)
        visible_classes = set(image_points.keys())

        # -----------------------------------
        # 2. Ball detection check
        # -----------------------------------

        if not frame_df.iloc[0]["ball_detected"]:

            score_table.loc[frame_index,"quality_level"] = "F"
            continue

        # 3. evaluate spacing and orientation of original configuration
        original_result = evaluate_original_configuration(image_points)

        score_table.loc[
            frame_index,
            "original_spacing_pass"
        ] = original_result["spacing_pass"]


        score_table.loc[
            frame_index,
            "original_orientation_pass"
        ] = original_result["orientation_pass"]


        score_table.loc[
            frame_index,
            "original_needs_hypothesis"
        ] = original_result["needs_hypothesis"]

        # 4. genearte hypotheses if the original configuration did not pass spacing and orientation
        if (
            original_result["spacing_pass"]
            and
            original_result["orientation_pass"]
        ):

            hypotheses = [
                {
                    "name": "original",
                    "reason": "original_valid",
                    "points": image_points
                }
            ]

        else:

            hypotheses = generate_landmark_hypotheses(
                image_points,
                original_result
            )

        # 5. give hypotheses a geometry score
        best_result = compare_landmark_hypotheses(hypotheses)

        # 6. keep frames that were able to do a calibration transformation
        if not best_result["valid"]:
            score_table.loc[frame_index,"quality_level"] = "F"
            continue

        image_points = best_result["image_points"]
        geometry_score_v2 = best_result["geometry_score_v2"]
        configuration_score = best_result["configuration_score"]

        visible_classes = set(image_points.keys())
        landmark_count = len(visible_classes)
        score_table.loc[frame_index, "landmark_count"] = landmark_count
       
        score_table.loc[frame_index, "geometry_score_v2"] = geometry_score_v2
        score_table.loc[frame_index, "configuration_score"] = configuration_score
        score_table.loc[frame_index, "reprojection_score"] = best_result["reprojection_score"]
        score_table.loc[frame_index, "coverage_score"] = best_result["coverage_score"]
        score_table.loc[frame_index, "determinant_score"] = best_result["determinant_score"]
        score_table.loc[frame_index, "stretch_score"] = best_result["stretch_score"]
        score_table.loc[frame_index, "edge_score"] = best_result["edge_score"]

        score_table.loc[frame_index, "selected_hypothesis"] = best_result["selected_hypothesis"]
        score_table.loc[frame_index, "hypothesis_reason"] = best_result["hypothesis_reason"]

        score_table.loc[
            frame_index,
            "final_landmarks"
        ] = ",".join(
            sorted(image_points.keys())
        )

        score_table.loc[
            frame_index,
            "all_hypothesis_scores"
        ] = str(
            best_result.get(
                "all_hypothesis_summary",
                []
            )
        )

        # 7. compute ball_score (since it is part of ranking score)
        # ball stats
        ball_detected = False
        ball_conf = 0

        distance_change = np.nan
        area_change = np.nan

        distance_score = 0
        area_score = 0
        ball_score = 0
        
        if frame_df.iloc[0]["ball_detected"]:

            ball_detected = True

            ball_conf = frame_df.iloc[0]["ball_confidence"]

            ball_x = frame_df.iloc[0]["ball_center_x"]
            ball_y = frame_df.iloc[0]["ball_center_y"]

            ball_x1 = frame_df.iloc[0]["ball_x1"]
            ball_y1 = frame_df.iloc[0]["ball_y1"]

            ball_x2 = frame_df.iloc[0]["ball_x2"]
            ball_y2 = frame_df.iloc[0]["ball_y2"]

            ball_width = ball_x2 - ball_x1
            ball_height = ball_y2 - ball_y1

            ball_area = (
                ball_width *
                ball_height
            )

            if previous_ball_area is not None:
                area_change = abs(
                    ball_area -
                    previous_ball_area
                )
            else:
                area_change = np.nan
            if previous_ball_x is not None:
                distance_change = np.sqrt(
                    (ball_x - previous_ball_x)**2 +
                    (ball_y - previous_ball_y)**2
                )
            else:
                distance_change = np.nan 
                
            previous_ball_area = ball_area
            previous_ball_x = ball_x
            previous_ball_y = ball_y
        
        if pd.notna(area_change):
            if area_change > 50:
                area_score = 0
            else:
                area_score = 100
        if pd.notna(distance_change):
            if distance_change <= 75:
                distance_score = 100
            elif distance_change <= 150:
                distance_score = 50
            else:
                distance_score = 0

        score_table.loc[frame_index, "detected"] = ball_detected
        score_table.loc[frame_index, "area_score"] = area_score
        score_table.loc[frame_index, "distance_score"] = distance_score
        score_table.loc[frame_index, "ball_conf"] = ball_conf

        if ball_detected:
            ball_confidence_score = ball_conf * 100
            ball_score =(
                CONFIDENCE_WEIGHT * ball_confidence_score +
                DISTANCE_WEIGHT * distance_score +
                BB_WEIGHT * area_score
            )

        score_table.loc[frame_index, "ball_score"] = ball_score

        # 8. assign tiers
        
        has_orientation_reference = (
            "home_plate" in visible_classes
            or
            "pitching_rubber" in visible_classes
        )

        if landmark_count >= 4 and geometry_score_v2 >= 80 and configuration_score >= 0.90 and has_orientation_reference:
            quality = "A"

        elif landmark_count >= 3 and geometry_score_v2 >= 75 and configuration_score >= 0.80 and has_orientation_reference:
            quality = "B"

        elif landmark_count >= 2 and geometry_score_v2 >= 60 and configuration_score >= 0.60 and has_orientation_reference:
            if (
                "pitching_rubber" in visible_classes # make sure visible_classes still exists in new script
                and landmark_count == 2
            ):
                quality = "D-"
            else:
                quality = "C"

        elif landmark_count >= 2: # and has_orientation_reference
            if (
                "pitching_rubber" in visible_classes
                and landmark_count == 2
            ):
                quality = "D-"
            else:
                quality = "D"
        elif landmark_count >= 2:
            quality = "D-"
        else:
            quality = "F"

        score_table.loc[frame_index, "quality_level"] = quality
        
        # 9. average landmark confidence
        if landmark_count == 0:
            avg_landmark_conf = 0
        else:
            confidence_values = []
            for landmark in visible_classes:
                conf_col = landmark + "_confidence"
                conf = frame_df.iloc[0][conf_col]
                if pd.notna(conf):
                    confidence_values.append(conf)
            if len(confidence_values) > 0:
                avg_landmark_conf = np.mean(confidence_values)
            else:
                avg_landmark_conf = 0

        score_table.loc[frame_index, "avg_landmark_conf"] = avg_landmark_conf

        # 10. ranking score
        base_ranking_score = compute_ranking_score(
            geometry_score_v2,
            ball_score,
            avg_landmark_conf,
            np.nan   # timing added later so a perfect ranking score right now would be 85
        )
        
        score_table.loc[frame_index,"base_ranking_score"] = base_ranking_score
        
        # 11. distance from first touch (for timing score)
        distance = abs(frame - first_touch_frame)        
        score_table.loc[frame_index, "distance_from_first_touch"] = distance

    # now we are OUTSIDE THE FRAME LOOP
    print("Hard cut starts at:", hard_cut_start_frame)
    print("Large pan starts at:", large_pan_start_frame)
    if hard_cut_start_frame is not None:
        score_table.loc[
            (score_table["frame"] >= hard_cut_start_frame) &
            (score_table["frame"] <= hard_cut_start_frame + 3),
            "searchable"
        ] = False

    if large_pan_start_frame is not None:
        score_table.loc[
            (score_table["frame"] >= large_pan_start_frame) &
            (score_table["frame"] <= large_pan_start_frame + 5),
            "searchable"
        ] = False
        
    primary_search = score_table[(
        (score_table["search_region"].isin(["before", "window"])) &
        (score_table["searchable"] == True)
    )]
    backup_search = score_table[(
        (score_table["search_region"] == "after") &
        (score_table["searchable"] == True)
    )]

    #timing score
    primary_search = primary_search.copy()
        
    primary_search_start = primary_search["frame"].min()
    primary_search_end = primary_search["frame"].max()

    max_distance = max(
        first_touch_frame - primary_search_start,
        primary_search_end - first_touch_frame
    )
    if max_distance == 0:
        max_distance = 1


    primary_search["timing_score"] = (
        1 -
        primary_search["distance_from_first_touch"] / max_distance
    )

    primary_search["timing_score"] = (
        primary_search["timing_score"]
        .clip(lower=0)
        * 100
    )

    primary_candidates = primary_search[
        (primary_search["quality_level"].isin(
            ["A","B","C","D","D-"]
        ))
        &
        (primary_search["detected"] == True)
        &
        (primary_search["base_ranking_score"].notna())
    ]

    if not primary_candidates.empty:
        search_pool = primary_search
        use_timing = True

    else:
        search_pool = backup_search.copy()
        search_pool["timing_score"] = 0
        use_timing = False
        
    print("\nQUALITY COUNTS")
    print(score_table["quality_level"].value_counts())
    print()
    print(search_pool["quality_level"].value_counts())

    all_ranked_frames = []

    for grade in ["A", "B", "C", "D", "D-"]:

        ranked = rank_frames(
            search_pool,
            grade,
            use_timing
        )

        if not ranked.empty:
            ranked["quality_level"] = grade
            score_table.loc[ranked.index, "ranking_score"] = ranked["ranking_score"]
            score_table.loc[ranked.index, "timing_score"] = ranked["timing_score"]
            all_ranked_frames.append(ranked)

    if len(all_ranked_frames) == 0:
        print("No calibration frame found for", video_name)
        score_table.to_csv(
            output_csv,
            index=False
        )
        return None

    calibration_candidates = pd.concat(
        all_ranked_frames,
        ignore_index=True
    )

    quality_priority = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "D-": 5,
    }

    calibration_candidates["quality_priority"] = (
        calibration_candidates["quality_level"]
        .map(quality_priority)
    )

    calibration_candidates = (
        calibration_candidates
        .sort_values(
            by=[
                "quality_priority",
                "ranking_score",
                "timing_score"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
    )
    
    calibration_candidates = (
        calibration_candidates
        .head(10)
    )

    calibration_candidates.to_csv(
        candidate_csv,
        index=False
    )

    score_table.to_csv(
        output_csv,
        index=False
    )


    print()
    print("=" * 60)
    print(f"Top Calibration Candidates for {video_name}")
    print("=" * 60)

    print(
        calibration_candidates[
            [
                "frame",
                "landmark_count",
                "quality_level",
                "search_region",
                "geometry_score_v2",
                "ranking_score",
                "ball_score",
                "timing_score"
            ]
        ].to_string(index=False)
    )
    
    return calibration_candidates

def main():

    csv_files = sorted(

        f

        for f in os.listdir(MASTER_DETECTION_FOLDER)

        if f.endswith(".csv")

    )

    all_results = []

    for csv_name in csv_files:
        
        video_name = os.path.splitext(csv_name)[0]

        master_csv = os.path.join(
            MASTER_DETECTION_FOLDER,
            csv_name
        )

        output_csv = os.path.join(
            OUTPUT_FOLDER,
            csv_name
        )

        candidate_csv = os.path.join(
            CANDIDATE_FOLDER,
            csv_name
        )

        summary = process_video(
            video_name,
            master_csv,
            candidate_csv,
            output_csv,
            first_touch_df
        )

if __name__ == "__main__":

    main()
