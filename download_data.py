"""
Dataset Download & Organization Script
Downloads the Microsoft Cats vs Dogs dataset and splits it into
train/validation/test directories for the ML pipeline.

Usage:
    python download_data.py
"""

import os
import shutil
import random
import urllib.request
import zipfile
from PIL import Image

DATA_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_FILE = "kagglecatsanddogs_5340.zip"
EXTRACTED_DIR = "PetImages"

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 42


def download_dataset():
    """Download the dataset ZIP from Microsoft servers."""
    if not os.path.exists(ZIP_FILE):
        print(f"Downloading dataset from Microsoft (~800 MB)...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
        print("Download complete.")
    else:
        print("ZIP file already exists, skipping download.")


def extract_dataset():
    """Extract the ZIP archive."""
    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    else:
        print("Dataset already extracted, skipping.")


def is_valid_image(filepath):
    """Verify a file is a valid, non-corrupt image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def organize_data():
    """Split raw images into train/validation/test directories."""
    random.seed(SEED)

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in ["cats", "dogs"]:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    label_map = {"cats": "Cat", "dogs": "Dog"}

    for label, folder_name in label_map.items():
        source_dir = os.path.join(EXTRACTED_DIR, folder_name)
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} not found, skipping.")
            continue

        # Collect only valid images
        all_files = sorted(os.listdir(source_dir))
        valid_files = []
        skipped = 0
        for f in all_files:
            fpath = os.path.join(source_dir, f)
            if os.path.isfile(fpath) and is_valid_image(fpath):
                valid_files.append(f)
            else:
                skipped += 1

        print(f"  {label}: {len(valid_files)} valid images ({skipped} skipped as corrupt)")

        random.shuffle(valid_files)

        n = len(valid_files)
        train_end = int(n * SPLIT_RATIOS["train"])
        val_end = train_end + int(n * SPLIT_RATIOS["validation"])

        splits = {
            "train": valid_files[:train_end],
            "validation": valid_files[train_end:val_end],
            "test": valid_files[val_end:],
        }

        split_dirs = {"train": TRAIN_DIR, "validation": VAL_DIR, "test": TEST_DIR}

        for split_name, files in splits.items():
            dest = os.path.join(split_dirs[split_name], label)
            for f in files:
                shutil.copy2(os.path.join(source_dir, f), os.path.join(dest, f))
            print(f"    -> {split_name}: {len(files)} images")


def print_summary():
    """Print a summary of the organized dataset."""
    print("\n========== Dataset Summary ==========")
    total = 0
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join("data", split)
        print(f"\n  {split.upper()}:")
        for cls in ["cats", "dogs"]:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                count = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
                print(f"    {cls}: {count} images")
                total += count
    print(f"\n  TOTAL: {total} images")
    print("=====================================")


def cleanup():
    """Remove downloaded ZIP and extracted raw folder."""
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        print(f"Removed {ZIP_FILE}")
    if os.path.exists(EXTRACTED_DIR):
        shutil.rmtree(EXTRACTED_DIR)
        print(f"Removed {EXTRACTED_DIR}/")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    organize_data()
    print_summary()

    response = input("\nClean up downloaded ZIP and raw files? (y/n): ").strip().lower()
    if response == "y":
        cleanup()
    print("\nDone! You can now run: python train.py")
