import cv2
import pandas as pd

IMAGE_FILE = r"C:\Users\klose\Downloads\ScreenTrackProject\gold_frame.png"

image = cv2.imread(IMAGE_FILE)

if image is None:
    raise Exception("Could not open image.")

LANDMARKS = {
    ord("1"): "home_plate",
    ord("2"): "first_base",
    ord("3"): "second_base",
    ord("4"): "third_base",
    ord("5"): "pitching_rubber"
}

points = {}
current_click = None

def mouse_callback(event, x, y, flags, param):

    global current_click

    if event == cv2.EVENT_LBUTTONDOWN:

        current_click = (x, y)

        print(f"Clicked: ({x}, {y})")

cv2.namedWindow("Landmark Annotator")
cv2.setMouseCallback("Landmark Annotator", mouse_callback)

while True:

    display = image.copy()

    for name, pt in points.items():

        cv2.circle(display, pt, 6, (0,255,0), -1)

        cv2.putText(
            display,
            name,
            (pt[0]+8, pt[1]-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )
    if current_click is not None:

        cv2.circle(display, current_click, 6, (0,0,255), -1)
        
    cv2.imshow("Landmark Annotator", display)

    key = cv2.waitKeyEx(20)

    if key == 27:
        break

    elif key in LANDMARKS:

        if current_click is None:
            print("Click first.")
            continue

        landmark = LANDMARKS[key]

        points[landmark] = current_click

        print(f"{landmark} saved.")
        
    elif key == ord("s"):

        rows = []

        for landmark, pt in points.items():

            rows.append({
                "landmark": landmark,
                "center_x": pt[0],
                "center_y": pt[1]
            })

        df = pd.DataFrame(rows)

        df.to_csv("gold_landmarks.csv", index=False)

        print("Saved gold_landmarks.csv")

cv2.destroyAllWindows()
