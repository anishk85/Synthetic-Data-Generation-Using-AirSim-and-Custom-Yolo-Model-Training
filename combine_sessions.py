"""
Combine multiple auto_session folders and split into train/val/test
Combines only the 4 specified auto_session folders
"""

import os
import shutil
from pathlib import Path
import random

# Configuration
SESSIONS_TO_COMBINE = [
    "auto_session_20260217_194413",
    "auto_session_20260217_194844",
    "auto_session_20260217_195641",
    "auto_session_20260217_200924"
]

FLIGHT_LOGS_DIR = Path(r"C:\Users\anish\Documents\AirSim\flightLogs")
OUTPUT_DIR = Path(r"C:\Users\anish\Documents\AirSim\dataset")

# Split ratios for 873 total images
TRAIN_COUNT = 650  # ~74%
VAL_COUNT = 140    # ~16%
# TEST_COUNT will be the rest (~10%)

def collect_all_files():
    """Collect all image and label files from the specified sessions"""
    all_files = []  # List of tuples (image_path, label_path)
    
    for session_name in SESSIONS_TO_COMBINE:
        session_dir = FLIGHT_LOGS_DIR / session_name
        images_dir = session_dir / "images"
        labels_dir = session_dir / "labels"
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
            
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping...")
            continue
        
        # Get all PNG files
        image_files = sorted(images_dir.glob("*.png"))
        print(f"Found {len(image_files)} images in {session_name}")
        
        for img_path in image_files:
            # Find corresponding label file
            label_path = labels_dir / (img_path.stem + ".txt")
            
            if label_path.exists():
                all_files.append((img_path, label_path))
            else:
                print(f"Warning: No label for {img_path.name}, skipping...")
    
    return all_files

def create_dataset_structure():
    """Create train/val/test folder structure"""
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    print(f"Created dataset structure at {OUTPUT_DIR}")

def copy_files(file_pairs, split_name, start_idx=0):
    """Copy image and label files to the specified split folder"""
    images_dir = OUTPUT_DIR / split_name / 'images'
    labels_dir = OUTPUT_DIR / split_name / 'labels'
    
    copied = 0
    for i, (img_path, label_path) in enumerate(file_pairs):
        # Create new sequential filename
        new_name = f"{split_name}_{start_idx + i:05d}"
        
        # Copy image
        shutil.copy2(img_path, images_dir / f"{new_name}.png")
        
        # Copy label
        shutil.copy2(label_path, labels_dir / f"{new_name}.txt")
        
        copied += 1
    
    print(f"Copied {copied} files to {split_name}")
    return copied

def main():
    print("="*60)
    print("Combining Auto Sessions into Train/Val/Test Split")
    print("="*60)
    
    # Collect all files from the 4 sessions
    print("\nCollecting files from sessions...")
    all_files = collect_all_files()
    
    total_files = len(all_files)
    print(f"\nTotal files collected: {total_files}")
    
    if total_files == 0:
        print("ERROR: No files found to combine!")
        return
    
    # Shuffle to randomize distribution
    random.seed(42)  # For reproducibility
    random.shuffle(all_files)
    
    # Calculate splits
    test_count = total_files - TRAIN_COUNT - VAL_COUNT
    
    print(f"\nSplit distribution:")
    print(f"  Train: {TRAIN_COUNT} images ({TRAIN_COUNT/total_files*100:.1f}%)")
    print(f"  Val:   {VAL_COUNT} images ({VAL_COUNT/total_files*100:.1f}%)")
    print(f"  Test:  {test_count} images ({test_count/total_files*100:.1f}%)")
    
    # Create output structure
    print("\nCreating dataset structure...")
    create_dataset_structure()
    
    # Split the data
    train_files = all_files[:TRAIN_COUNT]
    val_files = all_files[TRAIN_COUNT:TRAIN_COUNT+VAL_COUNT]
    test_files = all_files[TRAIN_COUNT+VAL_COUNT:]
    
    # Copy files to respective folders
    print("\nCopying files...")
    copy_files(train_files, 'train', start_idx=0)
    copy_files(val_files, 'val', start_idx=0)
    copy_files(test_files, 'test', start_idx=0)
    
    print("\n" + "="*60)
    print("Dataset combination complete!")
    print(f"Output location: {OUTPUT_DIR}")
    print("="*60)
    
    # Verify the output
    print("\nVerifying output...")
    for split in ['train', 'val', 'test']:
        img_count = len(list((OUTPUT_DIR / split / 'images').glob('*.png')))
        label_count = len(list((OUTPUT_DIR / split / 'labels').glob('*.txt')))
        print(f"  {split}: {img_count} images, {label_count} labels")

if __name__ == "__main__":
    main()
