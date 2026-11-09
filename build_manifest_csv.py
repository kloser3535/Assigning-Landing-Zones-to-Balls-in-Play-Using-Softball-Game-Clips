import os
import pandas as pd

BASE_FOLDER = r"C:\Summer_Scholars_Project"

OUTPUT_CSV = r"C:\Summer_Scholars_Project\manifest.csv"

def create_video_id(video_path):

    relative_path = os.path.relpath(
        video_path,
        BASE_FOLDER
    )

    clean = relative_path.replace("\\","_")

    clean = os.path.splitext(clean)[0]

    return clean

def valid_video(video_path):

    parts = video_path.split(os.sep)

    path_string = "_".join(parts).lower()


    if "broadcast" not in path_string:
        return False


    if "inf" not in path_string:
        return False


    return True

records = []


for root, dirs, files in os.walk(BASE_FOLDER):

    for file in files:

        if not file.endswith(".mp4"):
            continue


        video_path = os.path.join(
            root,
            file
        )


        if not valid_video(video_path):
            continue


        video_id = create_video_id(video_path)


        records.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "processed": False,
                "landmark_done": False,
                "ball_done": False
            }
        )


manifest = pd.DataFrame(records)


manifest.to_csv(
    OUTPUT_CSV,
    index=False
)


print(
    "Videos found:",
    len(manifest)
)
