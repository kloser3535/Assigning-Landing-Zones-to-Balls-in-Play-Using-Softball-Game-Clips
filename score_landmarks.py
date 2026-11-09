import pandas as pd

CSV_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\training_run_2_video_application_results\version_2_CSVs\GM_INF_O.csv"

df = pd.read_csv(CSV_FILE)

df["area"] = df["width"] * df["height"]

AREA_LIMITS = {

    "home_plate": 1650,

    "first_base": 2000,

    "second_base": 2000,

    "third_base": 1000,

    "pitching_rubber": 700

}

selected_rows = []

frames = sorted(df["frame"].unique())

def score_candidate(row):

    score = row["confidence"]

    predicted = row["class"]

    area = row["area"]

    limit = AREA_LIMITS.get(predicted, None)

    if limit is not None:

        if area > limit:

            excess = area - limit

            score -= excess / 3000

    return score

for frame in frames:

    frame_df = df[df["frame"] == frame]
    classes = frame_df["class"].unique()
    for landmark in classes:

        candidates = frame_df[
            frame_df["class"] == landmark
        ]

        candidates = candidates.copy()

        candidates["score"] = candidates.apply(
            score_candidate,
            axis=1
        )

        winner = candidates.loc[
            candidates["score"].idxmax()
        ]

        selected_rows.append(winner)
    cleaned_df = pd.DataFrame(selected_rows)

cleaned_df.to_csv(

    "cleaned_landmarks.csv",

    index=False

)
