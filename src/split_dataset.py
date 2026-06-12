import os
import random
import shutil

# Reproducible random split
random.seed(42)

# Source folders
SOURCE_DIRS = {
    "helmet": "helmet",
    "no_helmet": "no_helmet"
}

# Destination folder
DEST_ROOT = "dataset"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for class_name, source_dir in SOURCE_DIRS.items():

    files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(files)

    total = len(files)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    # Create folders
    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(DEST_ROOT, split, class_name),
            exist_ok=True
        )

    # Copy training files
    for file in train_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(DEST_ROOT, "train", class_name, file)
        )

    # Copy validation files
    for file in val_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(DEST_ROOT, "val", class_name, file)
        )

    # Copy test files
    for file in test_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(DEST_ROOT, "test", class_name, file)
        )

    print(f"\nClass: {class_name}")
    print(f"Total : {total}")
    print(f"Train : {len(train_files)}")
    print(f"Val   : {len(val_files)}")
    print(f"Test  : {len(test_files)}")

print("\nDataset split completed successfully!")