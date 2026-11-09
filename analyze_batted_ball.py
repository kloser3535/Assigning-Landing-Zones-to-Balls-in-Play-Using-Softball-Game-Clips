import pandas as pd
import numpy as np
import math

from calibration_models import (
    compute_transformation,
    project_ball,
    distance,
    bearing_angle,
    normalize_distance,
    assign_zone,
    compute_transformations,
    get_fence_distance,
    fence_distance_radians,
    mean_landmark_error,
)

from field_geometry import FIELD_POINTS

# constants/dictionaries

MODEL_PRIORITY = {
    "similarity": 1,
    "affine": 2,
    "homography": 3,
}

MODEL_SCORE_TOLERANCE = 5

# files
BASE_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject"

LANDMARK_BALL_FOLDER = fr"{BASE_FOLDER}\combined_csv_files_best_landmarks"
CALIBRATION_FRAME_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\calibration_frame_summary.csv"
OUTPUT_CSV = fr"{BASE_FOLDER}\ball_landing_csv_files"

def process_video(video_name):

    print()
    print("======================")
    print(video_name)
    print("======================")


    csv_file = os.path.join(
        LANDMARK_BALL_FOLDER,
        video_name + ".csv"
    )


    top5_file = os.path.join(
        TOP5_FOLDER,
        video_name + ".csv"
    )


    if not os.path.exists(csv_file):

        print("Missing detection CSV")
        return None


    if not os.path.exists(top5_file):

        print("Missing calibration candidates")
        return None

    calibration_df = pd.read_csv(CALIBRATION_FRAME_CSV)
    results = analyze_batted_ball(
        video_name,
        csv_file,
        top5_file
    )


    return results

def build_correspondences(image_points):
    """
    Build matching image and field point arrays
    for every visible landmark.
    """

    image_pts = []
    field_pts = []
    names = []

    for landmark in FIELD_POINTS:

        # Skip landmarks that aren't visible
        if landmark not in image_points:
            continue

        image_pts.append(image_points[landmark])
        field_pts.append(FIELD_POINTS[landmark])
        names.append(landmark)

    return (
        np.array(image_pts, dtype=np.float32),
        np.array(field_pts, dtype=np.float32),
        names
    )

def analyze_batted_ball(
        df,
        calibration_candidates
):

    # loops through for all of my candidates frames (up to 10 - however many are saved)
    for _, candidate in calibration_candidates.iterrows():

        calibration_frame = int(candidate["frame"])
        
        print("Trying calibration frame:", calibration_frame)

        result = analyze_with_calibration_frame(df, calibration_frame)

        if result["zone"] != "unassigned":
            return result

    # only after ALL frames fail        
    return {
        "zone": "unassigned",
        "calibration_frame": None,
        "model": None
    }


def analyze_with_calibration_frame(df, calibration_frame):

    frame_df = df[
        df["frame"] == calibration_frame
    ]

    image_points = {}
    ball_pixel = None


    row = frame_df.iloc[0]


    # --------------------------
    # Ball
    # --------------------------

    if (
        pd.notna(row["ball_center_x"])
        and
        pd.notna(row["ball_center_y"])
    ):

        ball_pixel = (
            row["ball_center_x"],
            row["ball_center_y"]
        )


    # --------------------------
    # Landmarks
    # --------------------------

    for landmark in [
        "home_plate",
        "first_base",
        "second_base",
        "third_base",
        "pitching_rubber"
    ]:

        x_col = landmark + "_center_x"
        y_col = landmark + "_center_y"

        if (
            pd.notna(row[x_col])
            and
            pd.notna(row[y_col])
        ):

            image_points[landmark] = (
                row[x_col],
                row[y_col]
            )
    print("Landmarks:")
    print(image_points)

    print("Ball:")
    print(ball_pixel)

    # build correspondences with image and field points
    image_pts, field_pts, names = build_correspondences(image_points)

    print("Visible landmarks:", names)

    num_landmarks = len(image_pts)

    # compute all transformations
    if num_landmarks < 2:
        print("Not enough landmarks for transformation")
        return None
    
    transforms = compute_transformations(image_pts, field_pts)
    
    if all(t is None for t in transforms.values()):
        print("Transformation failed")
        return None

    # project ball with all models
    candidate_landing_points = project_ball(transforms, ball_pixel)

    if all(c is None for c in candidate_landing_points.values()):
        print("Projection failed.")
        return None
    print(candidate_landing_points)

    # score different candidate landing points
    scores = score_transformations(candidate_landing_points, transforms, image_pts, field_pts) #, ball_pixel)
    print(scores)
    
    # pick winner
    # rank models based on score
    ranked_models = sorted(
        scores.items(),
        key=lambda x: x[1]["overall_score"],
        reverse=True
    )
    
    top_model = ranked_models[0][0]
    top_score = ranked_models[0][1]["overall_score"]

    # if only one model applied
    if len(ranked_models) == 1:
        best_model = top_model
    # for multiple models
    else:
        second_model = ranked_models[1][0]
        second_score = ranked_models[1][1]["overall_score"]
        score_gap = top_score - second_score
        # if score gap is greater than 5
        if score_gap > MODEL_SCORE_TOLERANCE:
            best_model = top_model
        # pick the most complex model is score gap is less than 5
        else:
            if MODEL_PRIORITY[top_model] > MODEL_PRIORITY[second_model]:
                best_model = top_model
            else:
                best_model = second_model

    models_to_try = [best_model]

    for model_name, _ in ranked_models:

        if model_name != best_model:
            models_to_try.append(model_name)

    for model_name in models_to_try:
        
        landing_point = candidate_landing_points.get(model_name)

        if landing_point is None:
            continue

        # compute necessary data (bearing and distance from home) to assign zone
        distance_home = distance(landing_point, FIELD_POINTS["home_plate"])
        bearing = bearing_angle(landing_point)
        u = normalize_distance(distance_home, bearing)
        zone = assign_zone(u, bearing)

        if zone is not None:

            print(
                f"Using {model_name}"
            )
            print()

            summary = {
                "zone": zone,
                "model": model_name,
                "calibration_frame": calibration_frame,
                "camera_state": frame_df.iloc[0]["camera_state"],
                "landing_point": landing_point,
                "landmark_error": scores[model_name]["reprojection_error"],
                "ball_x_ft": landing_point[0],
                "ball_y_ft": landing_point[1],
                "distance_from_home": distance_home,
                "bearing_angle": bearing,
                "normalized_distance": u,
                "model_score": scores[model_name]["overall_score"]
            }

            print(summary)
            
            return summary

    print("All models failed.")

    return {
        "zone": "unassigned",
        "model": None,
        "calibration_frame": calibration_frame
    }

def score_transformations(candidate_landing_points, transforms, image_pts, field_pts):
    scores = {}
    
    for model_name, ball_field in candidate_landing_points.items():
        
        # gather necessary data
        transform = transforms[model_name]
        x = ball_field[0]
        y = ball_field[1]

        # distance score (within fence)
        distance = np.sqrt(x**2 + y**2)
        bearing = bearing_angle(ball_field)
        fence = get_fence_distance(bearing)
        
        if distance <= fence:
            fence_score = 100
        else:
            penalty = distance - fence
            fence_score = max(
                0,
                100 - penalty
            )

        # bearing score (within fair territory)   
        if -2 <= bearing <= 92:
            fair_score = 100
        else:
            fair_score = 0

        # reprojection_score
        transform_geometry_error = mean_landmark_error(transform, image_pts, field_pts)
        reprojection_score = max(
            0,
            100 * (1 - transform_geometry_error/30) # I actually think 30 pixels was bad if I remember right BUT THIS COULD CHANGE
        )
    
        # overall_score
        overall_score = (
            0.37 * fence_score +
            0.37 * fair_score +
            0.26 * reprojection_score
        )

        scores[model_name] = {
            "distance": distance,
            "bearing": bearing,
            "fence_score": fence_score,
            "fair_score": fair_score,
            "reprojection_error": transform_geometry_error,
            "reprojection_score": reprojection_score,
            "overall_score": overall_score
        }
                    
    return scores

if __name__ == "__main__":
    csv_file = r"C:\Users\klose\Downloads\ScreenTrackProject\combined_csv_files_best_landmarks\FOR_INF_H.csv"
    df = pd.read_csv(csv_file)
    results = analyze_batted_ball(
        df,
        149
    )
    print(results)
