import os
import random
import shutil

random.seed(42)

# Existing dataset
SOURCE_DATASET = r"C:\Users\klose\Downloads\ScreenTrackProject\softball_ball_version_1"

# New dataset to create
DEST_DATASET = r"C:\Users\klose\Downloads\ScreenTrackProject\softball_ball_version_1_70_20_10"

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(DEST_DATASET, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DATASET, split, "labels"), exist_ok=True)

all_files = []

for split in ["train", "valid", "test"]:

    image_dir = os.path.join(SOURCE_DATASET, split, "images")
    label_dir = os.path.join(SOURCE_DATASET, split, "labels")

    for img in os.listdir(image_dir):

        if img.lower().endswith((".jpg", ".jpeg", ".png")):

            all_files.append({
                "image": os.path.join(image_dir, img),
                "label": os.path.join(
                    label_dir,
                    os.path.splitext(img)[0] + ".txt"
                ),
                "filename": img
            })

random.shuffle(all_files)

n = len(all_files)

n_train = int(0.70 * n)
n_valid = int(0.20 * n)

train_files = all_files[:n_train]
valid_files = all_files[n_train:n_train+n_valid]
test_files = all_files[n_train+n_valid:]

splits = {
    "train": train_files,
    "valid": valid_files,
    "test": test_files
}

for split_name, files in splits.items():

    for item in files:

        shutil.copy2(
            item["image"],
            os.path.join(
                DEST_DATASET,
                split_name,
                "images",
                item["filename"]
            )
        )

        if os.path.exists(item["label"]):

            shutil.copy2(
                item["label"],
                os.path.join(
                    DEST_DATASET,
                    split_name,
                    "labels",
                    os.path.basename(item["label"])
                )
            )

print("\nFinished!")

print(f"Total images: {len(all_files)}")
print(f"Train: {len(train_files)}")
print(f"Valid: {len(valid_files)}")
print(f"Test : {len(test_files)}")
