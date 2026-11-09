import pandas as pd

MANIFEST = r"C:\Users\klose\Downloads\ScreenTrackProject\manifest.csv"

manifest = pd.read_csv(MANIFEST)

new_columns = {
    "export_csv": "",
    "export_row": -1,
    "pipeline_done": False,
}

for column, default_value in new_columns.items():

    if column not in manifest.columns:

        manifest[column] = default_value

manifest.to_csv(
    MANIFEST,
    index=False
)

print("Manifest updated.")
