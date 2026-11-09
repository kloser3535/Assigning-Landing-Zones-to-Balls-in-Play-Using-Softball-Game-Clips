import pandas as pd
import os
import re


MANIFEST = (
    r"C:\Users\klose\Downloads\ScreenTrackProject\manifest.csv"
)


manifest = pd.read_csv(MANIFEST)


for index, row in manifest.iterrows():

    video_path = row["original_path"]

    # ------------------------------------
    # Get video number from path
    # ------------------------------------

    video_file = os.path.basename(video_path)

    video_stem = os.path.splitext(video_file)[0]

    try:

        video_number = int(video_stem)

    except ValueError:

        print(
            "Could not extract number:",
            video_file
        )

        continue


    # ------------------------------------
    # Find Export.csv folder
    # ------------------------------------

    video_folder = os.path.dirname(
        os.path.dirname(video_path)
    )

    export_csv = os.path.join(
        video_folder,
        "Export.csv"
    )


    if not os.path.exists(export_csv):

        print(
            "Missing Export:",
            export_csv
        )

        continue


    # ------------------------------------
    # Find matching Export row
    # ------------------------------------

    export_df = pd.read_csv(export_csv)


    matches = export_df[
        export_df["#"] == video_number
    ]


    if len(matches) == 0:

        print(
            "No export match:",
            video_number,
            export_csv
        )

        continue


    export_row = matches.index[0]


    # ------------------------------------
    # Save to manifest
    # ------------------------------------

    manifest.loc[
        index,
        "export_csv"
    ] = export_csv


    manifest.loc[
        index,
        "export_row"
    ] = export_row


manifest.to_csv(
    MANIFEST,
    index=False
)


print("Manifest updated.")
