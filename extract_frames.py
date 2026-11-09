import cv2
import os

video_folder = r"C:\Users\klose\Downloads\ScreenTrackProject\annotated_videos_v2"
output_folder = r"C:\Users\klose\Downloads\ScreenTrackProject\annotated_frames_v2"

os.makedirs(output_folder, exist_ok=True)

for video_name in os.listdir(video_folder):

    if not video_name.lower().endswith((".mp4", ".mov", ".avi")):
        continue

    video_path = os.path.join(video_folder, video_name)

    print(f"Processing {video_name}")

    cap = cv2.VideoCapture(video_path)

    frame_num = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Save one frame every 30 frames
        if frame_num % 15 == 0:

            filename = f"{os.path.splitext(video_name)[0]}_{frame_num}.jpg"

            output_path = os.path.join(output_folder, filename)

            cv2.imwrite(output_path, frame)

            saved_count += 1

        frame_num += 1

    cap.release()

    print(f"Saved {saved_count} frames.")

print("Finished extracting frames.")
