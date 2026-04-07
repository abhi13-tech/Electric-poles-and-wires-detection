import os
import random
import shutil
from collections import defaultdict
import pandas as pd

# ================= CONFIG =================
OUTPUT_DIR = "Pole_Splits"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.10
TEST_RATIO = 0.20

REQUIRE_ALL_VIEWS = False   #3 images per pole
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def load_and_group_images(image_dir):
    groups = defaultdict(list)
    for fname in os.listdir(image_dir):
        if not fname.startswith("pole_"):
            continue

        try:
            parts = fname.split("_")
            pole_id = int(parts[1])
            view_id = int(parts[2])

            groups[pole_id].append({
                "filename": fname,
                "view": view_id
            })
        except:
            continue

    return groups

def validate_groups(groups, require_all_views=True):
    valid_groups = {}

    for pole_id, items in groups.items():
        views = [x["view"] for x in items]

        if require_all_views:
            if sorted(views) != [1, 2, 3]:
                continue

        valid_groups[pole_id] = items

    print(f"Valid poles: {len(valid_groups)} / {len(groups)}")
    return valid_groups

def split_poles(groups):
    pole_ids = list(groups.keys())
    random.shuffle(pole_ids)

    n = len(pole_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_ids = pole_ids[:train_end]
    val_ids = pole_ids[train_end:val_end]
    test_ids = pole_ids[val_end:]

    return train_ids, val_ids, test_ids

def collect_files(groups, pole_ids):
    rows = []

    for pid in pole_ids:
        for item in groups[pid]:
            rows.append({
                "pole_id": pid,
                "view": item["view"],
                "filename": item["filename"]
            })

    return pd.DataFrame(rows)

def attach_labels(df, label_csv_path):
    labels = pd.read_csv(label_csv_path)  # must contain: pole_id, label
    df = df.merge(labels, on="pole_id", how="left")
    return df

def save_split(df, split_name, data_type):
    split_dir = os.path.join(OUTPUT_DIR, data_type, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for _, row in df.iterrows():
        src = os.path.join(IMAGE_DIR, row["filename"])
        dst = os.path.join(split_dir, row["filename"])
        shutil.copy(src, dst)

    df.to_csv(os.path.join(split_dir, f"{split_name}.csv"), index=False)

#--------------------Images-----------------------

IMAGE_DIR = "Pole Detection.yolov8/train/images"
groups = load_and_group_images(IMAGE_DIR)
groups = validate_groups(groups, REQUIRE_ALL_VIEWS)

train_ids, val_ids, test_ids = split_poles(groups)

train_df = collect_files(groups, train_ids)
val_df = collect_files(groups, val_ids)
test_df = collect_files(groups, test_ids)

save_split(train_df, "train", "images")
save_split(val_df, "val", "images")
save_split(test_df, "test", "images")

print("Done.")
print(f"Train poles: {len(train_ids)}")
print(f"Val poles: {len(val_ids)}")
print(f"Test poles: {len(test_ids)}")

#------------------Labels----------------------

IMAGE_DIR = "Pole Detection.yolov8/train/labels"

groups = load_and_group_images(IMAGE_DIR)
groups = validate_groups(groups, REQUIRE_ALL_VIEWS)

train_df = collect_files(groups, train_ids)
val_df = collect_files(groups, val_ids)
test_df = collect_files(groups, test_ids)

save_split(train_df, "train", "labels")
save_split(val_df, "val", "labels")
save_split(test_df, "test", "labels")

print("Done.")
print(f"Train poles: {len(train_ids)}")
print(f"Val poles: {len(val_ids)}")
print(f"Test poles: {len(test_ids)}")
