"""
Real-time debugging - check what colors are actually in the segmentation during flight
"""
import airsim
import cv2
import numpy as np

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ Connected!")

# Configure segmentation (same as oddDatasetmanualcontrol.py)
print("\nSetting up segmentation...")
client.simSetSegmentationObjectID(".*", 0, False)
client.simSetSegmentationObjectID("Car.*", 1, True)
client.simSetSegmentationObjectID(".*Vehicle.*", 1, True)
client.simSetSegmentationObjectID("Hedge.*", 2, True)
client.simSetSegmentationObjectID(".*Hedge.*", 2, True)
client.simSetSegmentationObjectID("Birch.*", 3, True)
client.simSetSegmentationObjectID("Tree.*", 3, True)
client.simSetSegmentationObjectID(".*Oak.*", 3, True)
client.simSetSegmentationObjectID(".*Pine.*", 3, True)
client.simSetSegmentationObjectID(".*Fir.*", 3, True)
client.simSetSegmentationObjectID("House.*", 4, True)
client.simSetSegmentationObjectID(".*House.*", 4, True)
client.simSetSegmentationObjectID("Small_House.*", 4, True)
client.simSetSegmentationObjectID(".*Building.*", 4, True)
print("✓ Done")

# Current mapping from oddDatasetmanualcontrol.py
RGB_TO_AIRSIM_ID = {
    (111, 89, 156): 1,      # Cars
    (113, 147, 147): 2,     # Hedges
    (105, 102, 73): 3,      # Trees
    (88, 83, 67): 4,        # Houses
}

print("\nCapturing segmentation image...")
responses = client.simGetImages([
    airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
])

seg_data = responses[0].image_data_uint8
seg = np.frombuffer(seg_data, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)

print(f"Segmentation shape: {seg.shape}")
print(f"Segmentation dtype: {seg.dtype}")

# Get unique colors
unique_colors, counts = np.unique(seg.reshape(-1, 3), axis=0, return_counts=True)

print(f"\n{'='*60}")
print(f"Found {len(unique_colors)} unique colors in LIVE segmentation:")
print(f"{'='*60}")

total_pixels = seg.shape[0] * seg.shape[1]
for color, count in zip(unique_colors, counts):
    percentage = (count / total_pixels) * 100
    r, g, b = color
    
    # Check if this color is in our mapping
    color_tuple = tuple(color)
    if color_tuple in RGB_TO_AIRSIM_ID:
        class_id = RGB_TO_AIRSIM_ID[color_tuple]
        print(f"✓ MATCHED: ({r:3d}, {g:3d}, {b:3d}) -> ID={class_id} - {count:7d} pixels ({percentage:5.2f}%)")
    else:
        if count > 1000:  # Only show significant colors
            print(f"  UNKNOWN: ({r:3d}, {g:3d}, {b:3d}) - {count:7d} pixels ({percentage:5.2f}%)")

print(f"\n{'='*60}")
print("Expected mappings:")
print(f"{'='*60}")
for rgb, airsim_id in RGB_TO_AIRSIM_ID.items():
    class_names = {1: "Car", 2: "Hedge", 3: "Tree", 4: "House"}
    print(f"  {rgb} -> {class_names[airsim_id]}")

print(f"\n{'='*60}")
print("DIAGNOSIS:")
print(f"{'='*60}")

# Check if ANY of our expected colors exist
matched = False
for rgb in RGB_TO_AIRSIM_ID.keys():
    if rgb in [tuple(c) for c in unique_colors]:
        matched = True
        break

if matched:
    print("✓ Some colors match! Detection should work.")
else:
    print("❌ NO COLORS MATCH!")
    print("\nPossible issues:")
    print("  1. AirSim was restarted (segmentation IDs reset)")
    print("  2. Color format is different (BGR vs RGB)")
    print("  3. Segmentation needs to be reconfigured")
    print("\nSOLUTION: Run identify_colors.py again and update RGB values!")

# Save for inspection
cv2.imwrite("debug_live_seg.png", seg)
print(f"\n✓ Saved debug_live_seg.png for inspection")
