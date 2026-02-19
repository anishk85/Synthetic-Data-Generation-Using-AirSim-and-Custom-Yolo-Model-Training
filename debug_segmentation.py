import airsim
import numpy as np
import cv2
from collections import Counter

# Connect to AirSim
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ Connected successfully!\n")

# 1. List all objects in the scene
print("=" * 60)
print("SCENE OBJECTS")
print("=" * 60)
try:
    all_objects = client.simListSceneObjects()
    print(f"Total objects found: {len(all_objects)}\n")
    
    # Group objects by prefix
    prefixes = {}
    for obj in all_objects:
        # Get first word/prefix
        parts = obj.split('_')
        prefix = parts[0] if parts else obj
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(obj)
    
    # Show grouped objects
    for prefix, objects in sorted(prefixes.items()):
        print(f"\n{prefix}* ({len(objects)} objects):")
        for obj in objects[:5]:  # Show first 5 examples
            print(f"  - {obj}")
        if len(objects) > 5:
            print(f"  ... and {len(objects) - 5} more")
            
except Exception as e:
    print(f"Error listing objects: {e}")

# 2. Capture and analyze segmentation colors
print("\n" + "=" * 60)
print("SEGMENTATION COLOR ANALYSIS")
print("=" * 60)
try:
    # Take RGB and segmentation images
    responses = client.simGetImages([
        airsim.ImageRequest("FrontCam", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
    ])
    
    rgb_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
        responses[0].height, responses[0].width, 3
    )
    seg_img = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(
        responses[1].height, responses[1].width, 3
    )
    
    # Save images for inspection
    cv2.imwrite("debug_rgb.png", rgb_img)
    cv2.imwrite("debug_segmentation.png", seg_img)
    print("✓ Saved debug_rgb.png and debug_segmentation.png")
    
    # Find unique RGB colors in segmentation
    seg_reshaped = seg_img.reshape(-1, 3)
    unique_colors = np.unique(seg_reshaped, axis=0)
    
    # Count pixels for each color
    color_counts = {}
    for color in unique_colors:
        mask = np.all(seg_img == color, axis=-1)
        count = np.sum(mask)
        color_counts[tuple(color)] = count
    
    # Sort by pixel count
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {len(unique_colors)} unique colors in segmentation:")
    print("\nColor (R, G, B) -> Pixel Count (% of image)")
    print("-" * 60)
    total_pixels = seg_img.shape[0] * seg_img.shape[1]
    
    for color, count in sorted_colors[:10]:  # Show top 10 colors
        percentage = (count / total_pixels) * 100
        print(f"{str(color):30s} -> {count:8d} ({percentage:5.2f}%)")
        
except Exception as e:
    print(f"Error analyzing segmentation: {e}")

# 3. Test current segmentation setup
print("\n" + "=" * 60)
print("TESTING CURRENT SEGMENTATION SETUP")
print("=" * 60)

try:
    # Reset all to 0
    client.simSetSegmentationObjectID(".*", 0, False)
    
    # Apply current setup
    client.simSetSegmentationObjectID(".*House", 1, True)
    client.simSetSegmentationObjectID(".*Roof", 1, True)
    client.simSetSegmentationObjectID(".*Wall", 1, True)
    client.simSetSegmentationObjectID(".*Door", 1, True)
    client.simSetSegmentationObjectID(".*Window", 1, True)
    client.simSetSegmentationObjectID("Tree.*", 2, True)
    client.simSetSegmentationObjectID(".*Vegetation", 2, True)  # Fixed!
    client.simSetSegmentationObjectID("InstanceFoliageAction.*", 2, True)
    client.simSetSegmentationObjectID("Car.*", 3, True)
    client.simSetSegmentationObjectID("Road.*", 4, True)
    
    print("✓ Segmentation IDs configured")
    print("\nTesting pattern matches:")
    
    # Test which objects match each pattern
    patterns = [
        ("Houses", [".*House", ".*Roof", ".*Wall", ".*Door", ".*Window"]),
        ("Trees", ["Tree.*", ".*Vegetation", "InstanceFoliageAction.*"]),
        ("Cars", ["Car.*"]),
        ("Roads", ["Road.*"])
    ]
    
    for name, pattern_list in patterns:
        print(f"\n{name}:")
        matched = set()
        for pattern in pattern_list:
            try:
                matches = client.simListSceneObjects(pattern)
                if matches:
                    matched.update(matches)
            except:
                pass
        
        if matched:
            print(f"  ✓ Matched {len(matched)} objects")
            for obj in list(matched)[:3]:
                print(f"    - {obj}")
            if len(matched) > 3:
                print(f"    ... and {len(matched) - 3} more")
        else:
            print(f"  ✗ No matches found!")
            
except Exception as e:
    print(f"Error testing segmentation: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\n📝 Next steps:")
print("  1. Review the object names to find correct patterns")
print("  2. Check the segmentation colors to update RGB_TO_AIRSIM_ID")
print("  3. Update oddDatasetmanualcontrol.py with correct patterns")
print("=" * 60)
