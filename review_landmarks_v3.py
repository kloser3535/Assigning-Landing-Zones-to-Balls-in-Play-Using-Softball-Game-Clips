import cv2
import pandas as pd
import os

LANDMARKS = {
    "1": "home_plate",
    "2": "first_base",
    "3": "second_base",
    "4": "third_base",
    "5": "pitching_rubber"
}

FALSE_OBJECTS = {
    "1": "shoe",
    "2": "chalk",
    "3": "batter_circle",
    "4": "player",
    "5": "fence",
    "6": "bat",
    "7": "other_base",
    "8": "duplicate",
    "9": "unknown"
}

VIDEO_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\testing_videos"

CSV_FOLDER = r"C:\Users\klose\Downloads\ScreenTrackProject\training_run_2_video_application_results\version_2_CSVs"

video_files = sorted([
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(".mp4")
])

OUTPUT_CSV = r"C:\Users\klose\Downloads\ScreenTrackProject\landmark_annotations_csv_files\landmark_evaluation_records.csv"

print("\nAvailable Videos\n")

for i, video in enumerate(video_files, start=1):
    print(f"{i}. {video}")

choice = int(input("\nSelect a video number: "))

selected_video = video_files[choice - 1]

VIDEO_FILE = os.path.join(
    VIDEO_FOLDER,
    selected_video
)

CSV_FILE = os.path.join(
    CSV_FOLDER,
    selected_video.replace(".mp4", ".csv")
)

print("\nSelected video:")
print(VIDEO_FILE)

print("\nSelected CSV:")
print(CSV_FILE)

df = pd.read_csv(CSV_FILE)

cap = cv2.VideoCapture(VIDEO_FILE)

VIDEO_NAME = os.path.basename(VIDEO_FILE)

if not cap.isOpened():
    print("Could not open video.")
    quit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video has", total_frames, "frames.")

current_frame = 0

def show_frame(frame_number):

    global boxes

    boxes = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        return

    frame_df = df[df["frame"] == frame_number]

    print("\n" + "=" * 70)
    print(f"FRAME {frame_number}")
    print("=" * 70)

    print(
        f"{'ID':<4}"
        f"{'CLASS':<18}"
        f"{'CONF':<8}"
        f"{'X':<8}"
        f"{'Y':<8}"
        f"{'WIDTH':<8}"
        f"{'HEIGHT':<8}"
        f"{'AREA':<10}"
        f"{'ASPECT':<10}"
    )

    print("-" * 70)    

    for i, (_, row) in enumerate(frame_df.iterrows()):

        area = row["width"] * row["height"]

        if row["height"] > 0:
            aspect_ratio = row["width"] / row["height"]
        else:
            aspect_ratio = 0
        
        print(
            f"{i:<4}"
            f"{row['class']:<18}"
            f"{row['confidence']:<8.2f}"
            f"{row['x_center']:<8.1f}"
            f"{row['y_center']:<8.1f}"
            f"{row['width']:<8.1f}"
            f"{row['height']:<8.1f}"
            f"{area:<8.2f}"
            f"{aspect_ratio:<8.2f}"
        )

        x1 = int(row["x_center"] - row["width"] / 2)
        y1 = int(row["y_center"] - row["height"] / 2)

        x2 = int(row["x_center"] + row["width"] / 2)
        y2 = int(row["y_center"] + row["height"] / 2)

        boxes.append({

            "id": i,

            "x1": x1,

            "y1": y1,

            "x2": x2,

            "y2": y2

        })

        review_key = (frame_number, i)

        color = (0,255,0)

        if review_key in reviewed:

            if reviewed[review_key] == "y":

                color = (255,0,0)

            else:

                color = (0,0,255)

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        cv2.putText(
            frame,
            str(i),
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            frame,
            row["class"],
            (x1, y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

    cv2.putText(
        frame,
        f"Frame {frame_number}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("Landmark Review Tool", frame)

    cv2.setMouseCallback(
        "Landmark Review Tool",
        mouse_click
    )

def mouse_click(event, x, y, flags, param):

    global selected_detection

    if event == cv2.EVENT_LBUTTONDOWN:

        for box in boxes:

            if (box["x1"] <= x <= box["x2"] and
                box["y1"] <= y <= box["y2"]):

                selected_detection = box["id"]

                annotate_detection(selected_detection)

                break

                print()

                print("=" * 60)
                print("Selected Detection:", selected_detection)
                print("=" * 60)

                break

annotations = []

reviewed = {}

camera_view = ""

boxes = []
selected_detection = None

def annotate_detection(detection_id):

    global annotations

    frame_df = df[df["frame"] == current_frame]

    review_key = (current_frame, detection_id)

    if detection_id < 0 or detection_id >= len(frame_df):
        print("Invalid detection.")
        return    

    if review_key in reviewed:

        print()

        print("This detection has already been reviewed.")

        return

    if detection_id < 0 or detection_id >= len(frame_df):
        print("Invalid detection.")
        return

    row = frame_df.iloc[detection_id]

    print()
    print("=" * 60)
    print("Detection", detection_id)
    print("=" * 60)

    print("Predicted class:", row["class"])
    print("Confidence:", row["confidence"])

    real = input("Real landmark? (y/n): ").lower()

    if real == "y":

        print("\nActual landmark:")

        for key, value in LANDMARKS.items():
            print(f"{key}: {value}")

        choice = input("Choice: ")

        actual = LANDMARKS.get(choice, "unknown")

        false_object = ""

    else:

        print("\nFalse object:")

        for key, value in FALSE_OBJECTS.items():
            print(f"{key}: {value}")

        choice = input("Choice: ")

        false_object = FALSE_OBJECTS.get(choice, "unknown")

        actual = ""

    area = row["width"] * row["height"]

    aspect_ratio = (
        row["width"] / row["height"]
        if row["height"] > 0 else 0
    )

    annotations.append({

        "video_file": VIDEO_NAME,
        "frame": current_frame,
        "camera_view": camera_view,
        "detection": detection_id,

        "predicted_class": row["class"],

        "confidence": row["confidence"],

        "x_center": row["x_center"],
        "y_center": row["y_center"],

        "width": row["width"],
        "height": row["height"],

        "area": area,
        "aspect_ratio": aspect_ratio,

        "real": real,
        "actual": actual,
        "false_object": false_object

    })

    reviewed[review_key] = real

    import os

    annotation_df = pd.DataFrame(annotations)

    if os.path.exists(OUTPUT_CSV):

        old_df = pd.read_csv(OUTPUT_CSV)

        annotation_df = pd.concat(
            [old_df, annotation_df],
            ignore_index=True
        )

    annotation_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    annotations.clear()

    print()
    print("Annotation saved.")

while True:

    show_frame(current_frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('d'):
        if current_frame < total_frames - 1:
            current_frame += 1
    elif key == ord('a'):
        if current_frame > 0:
            current_frame -= 1
    elif key == ord('l'):

        frame_df = df[df["frame"] == current_frame]

        if len(frame_df) == 0:
            print("No detections in this frame.")
            continue

        print("\nAvailable detections:\n")

        for i, (_, row) in enumerate(frame_df.iterrows()):
            print(
                f"{i}: "
                f"{row['class']} "
                f"(confidence={row['confidence']:.2f})"
            )

        detection_id = int(input("\nDetection ID: "))

        annotate_detection(detection_id)
    elif key == ord('v'):

        print("\nSelect camera view:")

        print("1 = pitch")
        print("2 = left_infield")
        print("3 = right_infield")
        print("4 = center_field")
        print("5 = outfield")
        print("6 = full_field")
        print("7 = other")

        choice = input("Choice: ")

        CAMERA_VIEWS = {
            "1": "pitch",
            "2": "left_infield",
            "3": "right_infield",
            "4": "center_field",
            "5": "outfield",
            "6": "full_field",
            "7": "other"
        }

        camera_view = CAMERA_VIEWS.get(choice, "unknown")

        print(f"\nCamera view set to: {camera_view}")
    elif key == ord('g'):

        frame = int(input("Jump to frame: "))

        if 0 <= frame < total_frames:
            current_frame = frame
        
cap.release()

cv2.destroyAllWindows()



