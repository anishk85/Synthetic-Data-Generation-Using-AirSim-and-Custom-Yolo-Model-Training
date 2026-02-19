"""
Dataset Preparation Script for YOLOv11 Training
Organizes AirSim collected data into YOLO format with train/val/test splits
"""

import os
import shutil
import random
from pathlib import Path
import glob

# Configuration
SOURCE_SESSIONS = "flightLogs/session_*"  # Pattern to match session folders
DATASET_ROOT = "dataset"
TRAIN_SPLIT = 0.7  # 70% training
VAL_SPLIT = 0.2    # 20% validation
TEST_SPLIT = 0.1   # 10% testing

def create_dataset_structure():
    """Create the YOLO dataset folder structure"""
    print("=" * 60)
    print("Creating YOLO Dataset Structure")
    print("=" * 60)
    
    folders = [
        f"{DATASET_ROOT}/train/images",
        f"{DATASET_ROOT}/train/labels",
        f"{DATASET_ROOT}/val/images",
        f"{DATASET_ROOT}/val/labels",
        f"{DATASET_ROOT}/test/images",
        f"{DATASET_ROOT}/test/labels"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print()

def collect_all_data():
    """Collect all images and labels from session folders"""
    print("=" * 60)
    print("Collecting Data from Session Folders")
    print("=" * 60)
    
    session_folders = glob.glob(SOURCE_SESSIONS)
    all_images = []
    all_labels = []
    
    for session in session_folders:
        img_folder = os.path.join(session, "images")
        lbl_folder = os.path.join(session, "labels")
        
        if os.path.exists(img_folder) and os.path.exists(lbl_folder):
            images = glob.glob(os.path.join(img_folder, "*.png"))
            labels = glob.glob(os.path.join(lbl_folder, "*.txt"))
            
            print(f"Session: {session}")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")
            
            all_images.extend(images)
            all_labels.extend(labels)
    
    print(f"\nTotal collected:")
    print(f"  Images: {len(all_images)}")
    print(f"  Labels: {len(all_labels)}")
    print()
    
    return all_images, all_labels

def split_dataset(images, labels):
    """Split dataset into train/val/test sets"""
    print("=" * 60)
    print("Splitting Dataset")
    print("=" * 60)
    
    # Create pairs of (image, label) with same base name
    data_pairs = []
    for img in images:
        img_name = Path(img).stem
        lbl_path = None
        for lbl in labels:
            if Path(lbl).stem == img_name:
                lbl_path = lbl
                break
        if lbl_path:
            data_pairs.append((img, lbl_path))
    
    print(f"Valid image-label pairs: {len(data_pairs)}")
    
    if len(data_pairs) == 0:
        print("ERROR: No valid image-label pairs found!")
        return [], [], []
    
    # Shuffle for random split
    random.shuffle(data_pairs)
    
    # Calculate split indices
    total = len(data_pairs)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)
    
    train_data = data_pairs[:train_end]
    val_data = data_pairs[train_end:val_end]
    test_data = data_pairs[val_end:]
    
    print(f"Train: {len(train_data)} ({TRAIN_SPLIT*100:.0f}%)")
    print(f"Val:   {len(val_data)} ({VAL_SPLIT*100:.0f}%)")
    print(f"Test:  {len(test_data)} ({TEST_SPLIT*100:.0f}%)")
    print()
    
    return train_data, val_data, test_data

def copy_files(data_pairs, split_name):
    """Copy files to the appropriate split folder"""
    print(f"Copying {split_name} data...")
    
    for idx, (img_path, lbl_path) in enumerate(data_pairs):
        # Use sequential numbering for organized dataset
        new_name = f"{split_name}_{idx:05d}"
        
        # Copy image
        img_ext = Path(img_path).suffix
        dst_img = os.path.join(DATASET_ROOT, split_name, "images", f"{new_name}{img_ext}")
        shutil.copy2(img_path, dst_img)
        
        # Copy label
        dst_lbl = os.path.join(DATASET_ROOT, split_name, "labels", f"{new_name}.txt")
        shutil.copy2(lbl_path, dst_lbl)
    
    print(f"✓ Copied {len(data_pairs)} {split_name} files\n")

def verify_dataset():
    """Verify the created dataset"""
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        img_count = len(glob.glob(f"{DATASET_ROOT}/{split}/images/*.png"))
        lbl_count = len(glob.glob(f"{DATASET_ROOT}/{split}/labels/*.txt"))
        print(f"{split.capitalize()}:")
        print(f"  Images: {img_count}")
        print(f"  Labels: {lbl_count}")
        print(f"  Match: {'✓' if img_count == lbl_count else '✗ MISMATCH!'}")
        print()

def analyze_dataset():
    """Analyze class distribution in the dataset"""
    print("=" * 60)
    print("Class Distribution Analysis")
    print("=" * 60)
    
    classes = ['Car', 'Hedge', 'Tree', 'House']
    class_counts = {i: 0 for i in range(4)}
    total_objects = 0
    
    for split in ['train', 'val', 'test']:
        label_files = glob.glob(f"{DATASET_ROOT}/{split}/labels/*.txt")
        
        for lbl_file in label_files:
            with open(lbl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if class_id < 4:  # Only count valid classes
                            class_counts[class_id] += 1
                            total_objects += 1
    
    print(f"Total objects: {total_objects}\n")
    for class_id, count in class_counts.items():
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"Class {class_id} ({classes[class_id]}): {count} ({percentage:.1f}%)")
    print()

def main():
    print("\n" + "=" * 60)
    print("YOLO DATASET PREPARATION FOR AIRSIM DATA")
    print("=" * 60 + "\n")
    
    # Step 1: Create structure
    create_dataset_structure()
    
    # Step 2: Collect all data
    images, labels = collect_all_data()
    
    if len(images) == 0:
        print("ERROR: No images found! Please collect data first using oddDatasetmanualcontrol.py")
        return
    
    # Step 3: Split dataset
    train_data, val_data, test_data = split_dataset(images, labels)
    
    # Step 4: Copy files
    copy_files(train_data, "train")
    copy_files(val_data, "val")
    copy_files(test_data, "test")
    
    # Step 5: Verify
    verify_dataset()
    
    # Step 6: Analyze
    analyze_dataset()
    
    print("=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print(f"\nDataset location: {os.path.abspath(DATASET_ROOT)}")
    print("Next step: Run train_yolo.py to start training")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
