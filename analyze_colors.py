import cv2
import numpy as np
import glob
import os

# Find latest session
sessions = sorted(glob.glob('flightLogs/session_*'))
if not sessions:
    print("No sessions found!")
    exit()

latest_session = sessions[-1]
print(f"Analyzing: {latest_session}")

# Get a few sample images (not the first one which had 0 detections)
image_files = sorted(glob.glob(f'{latest_session}/images/*.png'))
sample_images = image_files[10:15]  # Get images 10-14

print(f"\nAnalyzing {len(sample_images)} sample images...\n")

# Collect all unique RGB colors
all_colors = set()

for img_path in sample_images:
    # Load as RGB (OpenCV loads as BGR, so we need to convert)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get unique colors
    pixels = img_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    for color in unique_colors:
        all_colors.add(tuple(color))

# Sort colors for display
sorted_colors = sorted(list(all_colors))

print(f"Found {len(sorted_colors)} unique RGB colors across all samples:\n")
print("="*60)

for color in sorted_colors:
    # Skip black/background colors
    if color == (0, 0, 0):
        continue
    print(f"RGB: {color}")

print("\n" + "="*60)
print("\nCurrent mappings in code:")
print("  (72, 121, 89)  - Cars")
print("  (80, 250, 232) - Hedges")
print("  (34, 177, 76)  - Trees (NEEDS VERIFICATION)")
print("  (250, 20, 234) - Houses (NEEDS VERIFICATION)")
