import os
import shutil

DATASET_DIR = r"C:\Users\klose\Downloads\ScreenTrackProject\softball-landmark-version-2"

train_images = os.path.join(DATASET_DIR, "train", "images")
train_labels = os.path.join(DATASET_DIR, "train", "labels")

valid_images = set(os.listdir(os.path.join(DATASET_DIR, "valid", "images")))
test_images = set(os.listdir(os.path.join(DATASET_DIR, "test", "images")))

# Create new folders
clean_img_dir = os.path.join(DATASET_DIR, "train_clean", "images")
clean_lbl_dir = os.path.join(DATASET_DIR, "train_clean", "labels")

os.makedirs(clean_img_dir, exist_ok=True)
os.makedirs(clean_lbl_dir, exist_ok=True)

count = 0

for img_file in os.listdir(train_images):

    if img_file in valid_images or img_file in test_images:
        continue

    label_file = os.path.splitext(img_file)[0] + ".txt"

    shutil.copy2(
        os.path.join(train_images, img_file),
        os.path.join(clean_img_dir, img_file)
    )

    src_label = os.path.join(train_labels, label_file)

    if os.path.exists(src_label):
        shutil.copy2(
            src_label,
            os.path.join(clean_lbl_dir, label_file)
        )

    count += 1

print("Training images copied:", count)
