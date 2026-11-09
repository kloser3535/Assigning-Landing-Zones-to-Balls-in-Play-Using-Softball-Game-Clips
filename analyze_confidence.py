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

print()

print("=" * 60)

print("LANDMARK ANNOTATION SUMMARY")

print("=" * 60)

print()

print("Total detections:", len(df))

print()

print("*" * 30)

print()

print("Detection Categories")

print("-" * 30)

print(df["label"].value_counts())

print()

print("*" * 30)

print()

print("Percentages")

print("-" * 30)

percentages = df["label"].value_counts(normalize=True) * 100

print(percentages.round(1))

print()

print("*" * 30)

print()

print("Camera Views")

print("-" * 30)

print(df["camera_view"].value_counts())

print()

print("*" * 30)

print()

print("Predicted Classes")

print("-" * 30)

print(df["predicted_class"].value_counts())

print()

print("*" * 30)

print()

print("Actual Landmarks")

print("-" * 30)

print(df["actual"].value_counts())

print()

print("*" * 30)

print()

print("Missing Values")

print("-" * 30)

print(df.isnull().sum())

print()
print("=" * 60)
print("CONFIDENCE ANALYSIS")
print("=" * 60)

print()

print("Overall Confidence Statistics")

print(df["confidence"].describe())

print()

print("Confidence by Label")

print("-" * 40)

confidence_summary = (
    df.groupby("label")["confidence"]
      .describe()
      .round(3)
)

print(confidence_summary)

print()

print("=" * 60)
print("CONFIDENCE BY PREDICTED CLASS")
print("=" * 60)

confidence_by_class = (
    df.groupby("predicted_class")["confidence"]
      .describe()
      .round(3)
)

print(confidence_by_class)

print()

print("=" * 60)
print("CONFIDENCE BY VIEW AND CLASS AND LABEL")
print("=" * 60)

table = (
    df.groupby(
        [
            "camera_view",
            "predicted_class",
            "label"
        ]
    )["confidence"]
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

print(table)

print()

print("=" * 60)
print("CONFIDENCE BY CAMERA_LOCATION AND LABEL")
print("=" * 60)

location_summary = (
    df.groupby(["camera_location", "label"])["confidence"]
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
print("CONFIDENCE BY FIELD_SIDE AND LABEL")
print("=" * 60)

side_summary = (
    df.groupby(["field_side", "label"])["confidence"]
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
print("CONFIDENCE BY FIELD_REGION AND LABEL")
print("=" * 60)

region_summary = (
    df.groupby(["field_region", "label"])["confidence"]
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
        subset["confidence"],
        bins=20,
        alpha=0.5,
        label=label
    )

plt.xlabel("Confidence")

plt.ylabel("Detections")

plt.title("Confidence Distribution")

plt.legend()

plt.show()

confidence_summary.to_csv(
    "confidence_by_label.csv"
)

confidence_by_class.to_csv(
    "confidence_by_class.csv"
)

table.to_csv(
    "confidence_by_camera_class_label.csv"
)

location_summary.to_csv(
    "confidence_by_camera_location_class.csv"
)
side_summary.to_csv(
    "confidence_by_field_side_label.csv"
)
region_summary.to_csv(
    "confidence_by_field_region_label.csv"
)
