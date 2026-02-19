"""Discover AirSim's color palette for segmentation IDs 0-4."""
import airsim
import numpy as np
import time

client = airsim.MultirotorClient()
client.confirmConnection()

objects = client.simListSceneObjects()
cars = [o for o in objects if o.startswith("Car") and "Porch" not in o]
hedges = [o for o in objects if o.startswith("Hedge")]
birches = [o for o in objects if o.startswith("Birch")]
trees = [o for o in objects if o.startswith("Tree")]
houses = [o for o in objects if "House" in o]

def capture_seg():
    resp = client.simGetImages([
        airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
    ])
    seg = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
    return seg.reshape(resp[0].height, resp[0].width, 3)

def get_colors(seg):
    unique, counts = np.unique(seg.reshape(-1, 3), axis=0, return_counts=True)
    return {tuple(c): int(cnt) for c, cnt in zip(unique, counts)}

# Step 1: Reset everything to ID 0
print("Resetting all objects to ID 0...")
client.simSetSegmentationObjectID(r".*", 0, True)
time.sleep(2.0)

# Take off to see objects from the air
print("Taking off...")
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-10, 2).join()
time.sleep(2.0)

# Capture baseline (all ID 0)
seg0 = capture_seg()
colors0 = get_colors(seg0)
print(f"\nBaseline (all ID 0): {len(colors0)} colors")
for c, cnt in sorted(colors0.items(), key=lambda x: x[1], reverse=True):
    print(f"  {c}: {cnt} px ({cnt/921600*100:.1f}%)")

# Step 2: Set Cars to ID 1 and discover its color
print("\n--- Setting Cars to ID 1 ---")
for obj in cars:
    client.simSetSegmentationObjectID(obj, 1, False)
# Also try regex-based approach
client.simSetSegmentationObjectID(r".*Car.*", 1, True)
time.sleep(1.0)

seg1 = capture_seg()
colors1 = get_colors(seg1)
new_colors = {c: cnt for c, cnt in colors1.items() if c not in colors0}
print(f"New colors after Car=1: {new_colors}")

# Step 3: Set Hedges to ID 2
print("\n--- Setting Hedges to ID 2 ---")
for obj in hedges:
    client.simSetSegmentationObjectID(obj, 2, False)
client.simSetSegmentationObjectID(r".*Hedge.*", 2, True)
time.sleep(1.0)

seg2 = capture_seg()
colors2 = get_colors(seg2)
new_colors2 = {c: cnt for c, cnt in colors2.items() if c not in colors1}
print(f"New colors after Hedge=2: {new_colors2}")

# Step 4: Set Trees/Birch to ID 3
print("\n--- Setting Trees to ID 3 ---")
for obj in birches + trees:
    client.simSetSegmentationObjectID(obj, 3, False)
client.simSetSegmentationObjectID(r".*Birch.*", 3, True)
client.simSetSegmentationObjectID(r".*Tree.*", 3, True)
time.sleep(1.0)

seg3 = capture_seg()
colors3 = get_colors(seg3)
new_colors3 = {c: cnt for c, cnt in colors3.items() if c not in colors2}
print(f"New colors after Tree=3: {new_colors3}")

# Step 5: Set Houses to ID 4
print("\n--- Setting Houses to ID 4 ---")
for obj in houses:
    client.simSetSegmentationObjectID(obj, 4, False)
client.simSetSegmentationObjectID(r".*House.*", 4, True)
time.sleep(1.0)

seg4 = capture_seg()
colors4 = get_colors(seg4)
new_colors4 = {c: cnt for c, cnt in colors4.items() if c not in colors3}
print(f"New colors after House=4: {new_colors4}")

# Final summary
print("\n" + "=" * 60)
print("FINAL COLOR PALETTE")
print("=" * 60)
print(f"All colors in final frame:")
for c, cnt in sorted(colors4.items(), key=lambda x: x[1], reverse=True):
    pct = cnt / 921600 * 100
    if pct > 0.05:
        print(f"  {c}: {cnt} px ({pct:.1f}%)")

# Landing
print("\nLanding...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Done!")
