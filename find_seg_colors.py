"""
Quick script to capture one segmentation image and find all unique RGB colors.
Run this while AirSim is running and the drone is in the air.
"""
import airsim
import cv2
import numpy as np

try:
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ Connected!")
    
    # Configure segmentation (same as oddDatasetmanualcontrol.py)
    print("\nSetting up segmentation IDs...")
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
    print("✓ Segmentation configured")
    
    # Capture segmentation image
    print("\nCapturing segmentation image...")
    responses = client.simGetImages([
        airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
    ])
    
    seg_data = responses[0].image_data_uint8
    seg = np.frombuffer(seg_data, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
    
    # Save for inspection
    cv2.imwrite("debug_segmentation.png", seg)
    print("✓ Saved debug_segmentation.png")
    
    # Find unique RGB colors
    unique_colors = np.unique(seg.reshape(-1, 3), axis=0)
    
    print(f"\n{'='*60}")
    print(f"Found {len(unique_colors)} unique RGB colors:")
    print(f"{'='*60}")
    
    for color in unique_colors:
        r, g, b = color
        # Count how many pixels have this color
        mask = np.all(seg == color, axis=2)
        count = np.sum(mask)
        percentage = (count / (seg.shape[0] * seg.shape[1])) * 100
        print(f"RGB: ({r:3d}, {g:3d}, {b:3d}) - {count:7d} pixels ({percentage:5.2f}%)")
    
    print(f"\n{'='*60}")
    print("Current mappings in oddDatasetmanualcontrol.py:")
    print(f"{'='*60}")
    print("  (72, 121, 89)   -> ID=1 (Car)")
    print("  (80, 250, 232)  -> ID=2 (Hedge)")
    print("  (34, 177, 76)   -> ID=3 (Tree) ❌ WRONG")
    print("  (250, 20, 234)  -> ID=4 (House) ❌ WRONG")
    
    print(f"\n{'='*60}")
    print("ACTION NEEDED:")
    print(f"{'='*60}")
    print("Look at the colors above (excluding (0,0,0) which is background)")
    print("You should see colors for Car and Hedge that match.")
    print("Add the other non-black colors as Tree and House mappings.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure:")
    print("  1. AirSim is running")
    print("  2. Drone has been spawned (press Y to takeoff first)")
    import traceback
    traceback.print_exc()
