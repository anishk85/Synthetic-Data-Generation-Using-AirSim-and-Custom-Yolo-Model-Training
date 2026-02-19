"""Diagnostic script to understand AirSim segmentation API behavior."""
import airsim
import numpy as np
import time

client = airsim.MultirotorClient()
client.confirmConnection()

# Get some known actor names
objects = client.simListSceneObjects()
cars = [o for o in objects if o.startswith("Car") and "Porch" not in o][:5]
hedges = [o for o in objects if o.startswith("Hedge")][:3]
birches = [o for o in objects if o.startswith("Birch")][:3]
houses_q = [o for o in objects if o.startswith("House")][:3]
houses_s = [o for o in objects if o.startswith("Small_House")][:3]
trees = [o for o in objects if o.startswith("Tree")][:3]

print("=" * 60)
print("TEST 1: Set by exact actor name (is_name_regex=False)")
print("=" * 60)
for name in cars[:3]:
    result = client.simSetSegmentationObjectID(name, 1, False)
    print(f"  {name} -> {result}")

print()
print("=" * 60)
print("TEST 2: Set by exact actor name (is_name_regex=True)")
print("=" * 60)
for name in cars[:3]:
    # Escape special regex chars in exact names
    result = client.simSetSegmentationObjectID(name, 1, True)
    print(f"  {name} (regex) -> {result}")

print()
print("=" * 60)
print("TEST 3: Broad regex patterns")
print("=" * 60)
patterns = [
    (r".*Car.*", 1),
    (r".*Hedge.*", 2),
    (r".*Birch.*", 3),
    (r".*Tree.*", 3),
    (r".*House.*", 4),
    (r".*", 0),
    (r"[\w]*", 0),
    (r"Car", 1),
]
for pat, oid in patterns:
    result = client.simSetSegmentationObjectID(pat, oid, True)
    print(f"  Pattern '{pat}' -> ID {oid}: {result}")

print()
print("=" * 60)
print("TEST 4: Get segmentation ID for actor names vs mesh-style names")
print("=" * 60)
test_names = cars[:3] + hedges[:2] + birches[:2] + houses_q[:2] + houses_s[:2] + trees[:2]
for name in test_names:
    sid = client.simGetSegmentationObjectID(name)
    print(f"  {name}: ID={sid}")

print()
print("=" * 60)
print("TEST 5: Reset all to 0, capture, then set Car and re-capture")
print("=" * 60)

# Reset everything
r = client.simSetSegmentationObjectID(r".*", 0, True)
print(f"  Reset all with '.*': {r}")
time.sleep(1.0)

# Capture baseline
resp = client.simGetImages([
    airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
])
seg = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
seg = seg.reshape(resp[0].height, resp[0].width, 3)
unique0, counts0 = np.unique(seg.reshape(-1, 3), axis=0, return_counts=True)
print(f"  After reset: {len(unique0)} unique colors")
for c, cnt in sorted(zip(unique0, counts0), key=lambda x: x[1], reverse=True):
    print(f"    {tuple(c)}: {cnt} px")

# Now set all things with "Car" in name individually (is_name_regex=False)
print("\n  Setting individual car objects...")
set_count = 0
for obj in objects:
    if obj.startswith("Car") and "Porch" not in obj:
        r = client.simSetSegmentationObjectID(obj, 5, False)
        if r:
            set_count += 1
print(f"  Set {set_count}/{len([o for o in objects if o.startswith('Car') and 'Porch' not in o])} car objects (exact name, False)")

time.sleep(0.5)

# Capture again
resp = client.simGetImages([
    airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
])
seg2 = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
seg2 = seg2.reshape(resp[0].height, resp[0].width, 3)
unique2, counts2 = np.unique(seg2.reshape(-1, 3), axis=0, return_counts=True)
print(f"\n  After setting cars (exact, False): {len(unique2)} unique colors")
for c, cnt in sorted(zip(unique2, counts2), key=lambda x: x[1], reverse=True):
    print(f"    {tuple(c)}: {cnt} px")

# Now try with is_name_regex=True for individual car names
r = client.simSetSegmentationObjectID(r".*", 0, True)  # Reset again
time.sleep(0.5)

set_count2 = 0
for obj in objects:
    if obj.startswith("Car") and "Porch" not in obj:
        r = client.simSetSegmentationObjectID(obj, 5, True)
        if r:
            set_count2 += 1
print(f"\n  Set {set_count2}/{len([o for o in objects if o.startswith('Car') and 'Porch' not in o])} car objects (exact name, True)")

time.sleep(0.5)

# Capture again
resp = client.simGetImages([
    airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False)
])
seg3 = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
seg3 = seg3.reshape(resp[0].height, resp[0].width, 3)
unique3, counts3 = np.unique(seg3.reshape(-1, 3), axis=0, return_counts=True)
print(f"  After setting cars (exact, True): {len(unique3)} unique colors")
for c, cnt in sorted(zip(unique3, counts3), key=lambda x: x[1], reverse=True):
    print(f"    {tuple(c)}: {cnt} px")

# Check if new colors appeared
set0 = set([tuple(c) for c in unique0])
set2 = set([tuple(c) for c in unique2])
set3 = set([tuple(c) for c in unique3])
print(f"\n  Baseline colors: {set0}")
print(f"  New colors after exact-False: {set2 - set0}")
print(f"  New colors after exact-True: {set3 - set0}")

print("\nDone!")
