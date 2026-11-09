import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\landmark_annotations_csv_files\landmark_evaluation_records_updated.csv"

df = pd.read_csv(CSV_FILE)

CAMERA_INFO = {

    "home_cf": ("home", "middle", "outfield"),
    "home_full_field": ("home", "middle", "infield"),
    "home_left_infield": ("home", "left", "infield"),
    "home_mid_infield": ("home", "middle", "infield"),
    "home_right_infield": ("home", "right", "infield"),
    "home_lf": ("home", "left", "outfield"),
    "home_rf": ("home", "right", "outfield"),

    "of_left_infield": ("outfield", "left", "infield"),
    "of_mid_infield": ("outfield", "middle", "infield"),
    "of_right_infield": ("outfield", "right", "infield"),
    "of_lf": ("outfield", "left", "outfield"),
    "of_rf": ("outfield", "right", "outfield"),
    "cf_right_infield": ("outfield", "right", "infield")

}

df["camera_location"] = (
    df["camera_view"]
      .map(lambda x: CAMERA_INFO[x][0])
)

df["field_side"] = (
    df["camera_view"]
      .map(lambda x: CAMERA_INFO[x][1])
)

df["field_region"] = (
    df["camera_view"]
      .map(lambda x: CAMERA_INFO[x][2])
)

print("=" * 60)
print("AREA ANALYSIS")
print("=" * 60)

print()

print("Area by Label")

print("-" * 40)

area_summary = (
    df.groupby("label")["area"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max"
    )
    .round(3)
)

print(area_summary)

print()

print("=" * 60)
print("AREA BY PREDICTED CLASS AND LABEL")
print("=" * 60)

area_by_class = (
    df.groupby(["predicted_class", "label"])["area"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max"
    )
    .round(3)
)

print(area_by_class)

print()

print("=" * 60)
print("AREA BY CAMERA_LOCATION AND LABEL AND CLASS")
print("=" * 60)

location_summary = (
    df.groupby(["camera_location", "label", "predicted_class"])["area"]
      .agg(
          Count="count",
          Mean="mean",
          Median="median",
          Std="std",
          Min="min",
          Max="max"
      )
      .round(3)
)

print(location_summary)

print()

print("=" * 60)
print("AREA BY FIELD_SIDE AND LABEL AND CLASS")
print("=" * 60)

side_summary = (
    df.groupby(["field_side", "label", "predicted_class"])["area"]
      .agg(
          Count="count",
          Mean="mean",
          Median="median",
          Std="std",
          Min="min",
          Max="max"
      )
      .round(3)
)

print(side_summary)

print()

print("=" * 60)
print("AREA BY FIELD_REGION AND LABEL AND CLASS")
print("=" * 60)

region_summary = (
    df.groupby(["field_region", "label", "predicted_class"])["area"]
      .agg(
          Count="count",
          Mean="mean",
          Median="median",
          Std="std",
          Min="min",
          Max="max"
      )
      .round(3)
)

print(region_summary)

plt.figure(figsize=(8,5))

for label in ["correct", "background", "mislabelled"]:

    subset = df[df["label"] == label]

    plt.hist(
        subset["area"],
        bins=20,
        alpha=0.5,
        label=label
    )

plt.xlabel("Area")

plt.ylabel("Detections")

plt.title("Area Distribution")

plt.legend()

plt.show()

area_summary.to_csv(
    "area_by_label.csv"
)

area_by_class.to_csv(
    "confidence_by_class_label.csv"
)

location_summary.to_csv(
    "area_by_camera_location_label_class.csv"
)
side_summary.to_csv(
    "area_by_field_side_label_class.csv"
)
region_summary.to_csv(
    "area_by_field_region_label_class.csv"
)
