import os
import random
import shutil

# CHANGE THIS TO YOUR DATASET FOLDER
DATASET_DIR = r"C:\Users\klose\Downloads\ScreenTrackProject\softball-landmark-version-2"

random.seed(42)

images_dir = os.path.join(DATASET_DIR, "train", "images")
labels_dir = os.path.join(DATASET_DIR, "train", "labels")

# Create output folders
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)

# Get all image files
image_files = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(image_files)

n = len(image_files)

n_train = int(0.70 * n)
n_valid = int(0.20 * n)

train_files = image_files[:n_train]
valid_files = image_files[n_train:n_train+n_valid]
test_files = image_files[n_train+n_valid:]

splits = {
    "valid": valid_files,
    "test": test_files
}

for split_name, files in splits.items():

    for img_file in files:

        label_file = os.path.splitext(img_file)[0] + ".txt"

        src = os.path.join(images_dir, img_file)
        dst = os.path.join(DATASET_DIR, split_name, "images", img_file)

        print("COPYING:", src)
        print("TO:", dst)

        shutil.copy2(src, dst)

        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            shutil.copy2(
                label_path,
                os.path.join(DATASET_DIR, split_name, "labels", label_file)
            )

print("Dataset split complete!")
print(f"Train: {len(train_files)}")
print(f"Valid: {len(valid_files)}")
print(f"Test:  {len(test_files)}")
