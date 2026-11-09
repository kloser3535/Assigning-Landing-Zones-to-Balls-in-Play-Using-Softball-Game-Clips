import os
import shutil
import pandas as pd


BASE_FOLDER = r"C:\Summer_Scholars_Project"

OUTPUT_FOLDER = r"C:\YOLO_BATCH"

VIDEO_OUTPUT = os.path.join(
    OUTPUT_FOLDER,
    "videos"
)


os.makedirs(
    VIDEO_OUTPUT,
    exist_ok=True
)


records = []


for root, dirs, files in os.walk(BASE_FOLDER):

    # Only keep broadcast INF clips
    path_lower = root.lower()

    if (
        "broadcast" not in path_lower
        or "inf" not in path_lower
    ):
        continue


    for file in files:

        if not file.endswith(".mp4"):
            continue


        old_path = os.path.join(
            root,
            file
        )


        # Example:
        # DAY_H_GB_Out_INF_LEFT_angle_broadcast_video_1

        relative = os.path.relpath(
            old_path,
            BASE_FOLDER
        )


        video_id = (
            relative
            .replace("\\","_")
            .replace(".mp4","")
        )


        new_name = video_id + ".mp4"


        new_path = os.path.join(
            VIDEO_OUTPUT,
            new_name
        )


        shutil.copy2(
            old_path,
            new_path
        )


        records.append(
            {
            "video_id": video_id,
            "original_path": old_path,
            "colab_video_path": new_name,
            "landmark_done": False,
            "ball_done": False
            }
        )


manifest = pd.DataFrame(records)


manifest.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "manifest.csv"
    ),
    index=False
)


print(
    "Created manifest with",
    len(manifest),
    "videos"
)
