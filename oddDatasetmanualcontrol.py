import airsim
import keyboard
import time 
import csv
import cv2
import math 
import os
import numpy as np
from datetime import datetime

# Create session directories
session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_DIR = "flightLogs"
SESSION_DIR = os.path.join(BASE_DIR, f"session_{session_time}")
IMG_DIR = os.path.join(SESSION_DIR, "images")
LBL_DIR = os.path.join(SESSION_DIR, "labels")

# Ensure all directories exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

log_path = os.path.join(SESSION_DIR, "flight_log.csv")
print("=" * 60)
print(f"[SESSION] Starting new session")
print(f"[SESSION] Folder: {SESSION_DIR}")
print(f"[SESSION] Log file: {log_path}")
print("=" * 60)

# Connect to AirSim
print("\n[AIRSIM] Connecting to AirSim...")
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[AIRSIM] ✓ Connected successfully!")
except Exception as e:
    print(f"[ERROR] Failed to connect to AirSim: {e}")
    exit(1)

# Create and initialize log file
try:
    logFile = open(log_path, "w", newline="")
    writer = csv.writer(logFile)
    writer.writerow([
        "time", "command", "imageSaved", "imageName", "vx", "vy", "vz", "yawRate",
        "x", "y", "z", "roll", "pitch", "yaw",
        "gpsLat", "gpsLon", "gpsAlt"
    ])
    logFile.flush()
    print(f"[LOG] ✓ Log file created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create log file: {e}")
    exit(1)  

# Setup segmentation IDs for object detection
print("\n[SEGMENTATION] Setting up object IDs...")
try:
    # Step 1: Reset ALL objects to background (ID=0) using broad regex
    success = client.simSetSegmentationObjectID(r".*", 0, True)
    print(f"[SEGMENTATION] Reset all objects (regex): {success}")
    time.sleep(1.0)

    # Step 2: Set class IDs using regex patterns that match mesh/object names
    class_patterns = {
        1: [r".*Car.*"],       # Cars/Vehicles
        2: [r".*Hedge.*"],     # Hedges
        3: [r".*Birch.*", r".*Tree.*"],  # Trees (Birch + Tree meshes)
        4: [r".*House.*"],     # Houses/Buildings (includes Small_House, Medium_House, etc.)
    }
    class_labels = {1: "Car", 2: "Hedge", 3: "Tree", 4: "House"}

    for class_id, patterns in class_patterns.items():
        for pat in patterns:
            r = client.simSetSegmentationObjectID(pat, class_id, True)
            print(f"  {class_labels[class_id]} pattern '{pat}' -> ID {class_id}: {r}")

    # Step 3: Also set individual actor names for completeness
    all_objects = client.simListSceneObjects()
    individual_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for obj_name in all_objects:
        name = obj_name.lower()
        target_id = None
        if "car" in name and "porch" not in name:
            target_id = 1
        elif "hedge" in name:
            target_id = 2
        elif "birch" in name or (name.startswith("tree") or "_tree" in name):
            target_id = 3
        elif "house" in name:
            target_id = 4
        if target_id is not None:
            client.simSetSegmentationObjectID(obj_name, target_id, False)
            individual_counts[target_id] += 1

    print(f"[SEGMENTATION] Individual objects set: "
          f"Car={individual_counts[1]}, Hedge={individual_counts[2]}, "
          f"Tree={individual_counts[3]}, House={individual_counts[4]}")

    time.sleep(1.0)

    # Step 4: Use discovered color palette (deterministic per AirSim ID)
    # These were discovered by sequentially setting IDs and capturing:
    #   ID 0 (background) = (0, 0, 0)
    #   ID 1 (Car)         = (6, 108, 153)
    #   ID 2 (Hedge)       = (191, 105, 112)
    #   ID 3 (Tree)        = (72, 121, 89)
    #   ID 4 (House)       = (64, 225, 190)
    RGB_TO_AIRSIM_ID = {
        (6, 108, 153): 1,     # Car
        (191, 105, 112): 2,   # Hedge
        (72, 121, 89): 3,     # Tree
        (64, 225, 190): 4,    # House
    }
    AIRSIM_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3}

    print(f"\n[SEGMENTATION] ✓ Color mapping ({len(RGB_TO_AIRSIM_ID)} classes):")
    for color, aid in RGB_TO_AIRSIM_ID.items():
        print(f"  {color} -> {class_labels[aid]} (YOLO class {AIRSIM_TO_YOLO[aid]})")

except Exception as e:
    print(f"[WARNING] Segmentation setup failed: {e}")
    import traceback
    traceback.print_exc()
    print("[WARNING] Using empty color mapping - detection won't work!")
    RGB_TO_AIRSIM_ID = {}
    AIRSIM_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3}

# Image dimensions must match actual AirSim camera output (1280x720)
IMG_WIDTH = 1280
IMG_HEIGHT = 720
CAPTURE_INTERVAL=2

# Minimum area thresholds for each class (in pixels)
MIN_AREA = {
    0: 150,    # Cars: smaller for distant vehicles, better detection
    1: 200,    # Hedges: moderate size to avoid small debris
    2: 250,    # Trees: moderate to avoid small branches
    3: 400,    # Houses: larger for buildings
}

# Maximum aspect ratio (width/height or height/width) - per class
MAX_ASPECT_RATIO = {
    0: 4.0,    # Cars: reasonable aspect ratio for vehicles
    1: 10.0,   # Hedges: can be elongated (hedge rows)
    2: 3.0,    # Trees: usually vertical, fairly compact
    3: 5.0,    # Houses: can be wide or tall but not extreme
}

frameId=len(os.listdir(IMG_DIR))
lastCapture=0
lastStatusUpdate=time.time()  # For showing recording status periodically
VEL=2.5
YAW_RATE=25
vx=vy=vz=0.0
yawRate=0.0
currentCmd="idle"
flying=False
running=True
recording=False
imageSaved=0
imageName=""


def segmentationRGBToClassMap(SegRgb):
    h, w, _ = SegRgb.shape
    classMap = np.full((h, w), -1, dtype=np.int32)
    
    class_labels = {0: "Car", 1: "Hedge", 2: "Tree", 3: "House"}
    class_pixels = {}
    
    for rgb, airsimId in RGB_TO_AIRSIM_ID.items():
        mask = np.all(SegRgb == rgb, axis=-1)
        pixel_count = int(np.sum(mask))
        yolo_id = AIRSIM_TO_YOLO[airsimId]
        classMap[mask] = yolo_id
        if pixel_count > 0:
            class_pixels[yolo_id] = class_pixels.get(yolo_id, 0) + pixel_count
    
    # Debug output for first 3 frames
    global frameId
    if frameId < 3:
        total_classified = np.sum(classMap >= 0)
        total_pixels = h * w
        details = ", ".join([f"{class_labels[cid]}:{px}" for cid, px in sorted(class_pixels.items())])
        print(f"[DEBUG] Frame {frameId}: {total_classified}/{total_pixels} pixels classified "
              f"({total_classified/total_pixels*100:.1f}%) | {details}")
        if total_classified == 0:
            unique_colors = np.unique(SegRgb.reshape(-1, 3), axis=0)
            print(f"[DEBUG] Frame colors: {[tuple(c) for c in unique_colors[:10]]}")
            print(f"[DEBUG] Expected: {list(RGB_TO_AIRSIM_ID.keys())}")
    
    return classMap


def merge_boxes(boxes, iou_threshold=0.3):
    """Merge overlapping or nearby boxes using IoU threshold"""
    if len(boxes) == 0:
        return boxes
    
    def box_iou(box1, box2):
        # Convert from center format to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def merge_two_boxes(box1, box2):
        # Convert to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Calculate merged bounding box
        merged_xmin = min(x1_min, x2_min)
        merged_ymin = min(y1_min, y2_min)
        merged_xmax = max(x1_max, x2_max)
        merged_ymax = max(y1_max, y2_max)
        
        # Convert back to center format
        merged_x = (merged_xmin + merged_xmax) / 2
        merged_y = (merged_ymin + merged_ymax) / 2
        merged_w = merged_xmax - merged_xmin
        merged_h = merged_ymax - merged_ymin
        
        return (merged_x, merged_y, merged_w, merged_h)
    
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        current_box = boxes[i]
        merged_with = [i]
        
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if box_iou(current_box, boxes[j]) > iou_threshold:
                current_box = merge_two_boxes(current_box, boxes[j])
                merged_with.append(j)
        
        for idx in merged_with:
            used[idx] = True
        merged.append(current_box)
    
    return merged


def segToBoxes(ClassMap, classId):
    """Extract bounding boxes from segmentation mask with improved morphological cleaning"""
    mask = (ClassMap == classId).astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    # Use class-specific kernel sizes for better cleaning
    kernel_sizes = {
        0: 7,   # Cars: larger kernel for smooth shapes
        1: 5,   # Hedges: moderate kernel
        2: 5,   # Trees: moderate kernel
        3: 9    # Houses: largest kernel for buildings
    }
    kernel_size = kernel_sizes.get(classId, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Remove small noise with opening (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes with closing (dilation followed by erosion)
    # Apply closing twice for better hole filling
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = MIN_AREA.get(classId, 150)
    max_aspect = MAX_ASPECT_RATIO.get(classId, 5.0)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter by area
        if w * h <= min_area:
            continue
        
        # Filter by aspect ratio (class-specific)
        aspect_ratio = max(w / h, h / w) if h > 0 else 0
        if aspect_ratio > max_aspect:
            continue
        
        # Add small padding to boxes for better coverage (2% on each side)
        padding = 0.02
        x_pad = int(w * padding)
        y_pad = int(h * padding)
        x = max(0, x - x_pad)
        y = max(0, y - y_pad)
        w = min(IMG_WIDTH - x, w + 2 * x_pad)
        h = min(IMG_HEIGHT - y, h + 2 * y_pad)
        
        # Normalize to YOLO format (center_x, center_y, width, height)
        boxes.append((
            (x + w / 2) / IMG_WIDTH,
            (y + h / 2) / IMG_HEIGHT,
            w / IMG_WIDTH,
            h / IMG_HEIGHT
        ))
    
    # Merge overlapping boxes (especially useful for fragmented detections)
    # Use class-specific IoU thresholds
    iou_thresholds = {
        0: 0.4,   # Cars: higher threshold (less aggressive merging)
        1: 0.3,   # Hedges: moderate merging
        2: 0.35,  # Trees: moderate merging
        3: 0.5    # Houses: very conservative (buildings shouldn't merge much)
    }
    iou_thresh = iou_thresholds.get(classId, 0.3)
    if len(boxes) > 1:
        boxes = merge_boxes(boxes, iou_threshold=iou_thresh)
    
    return boxes

def captureDataset():
    global frameId
    try:
        responses = client.simGetImages([
            airsim.ImageRequest("FrontCam", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("FrontCam", airsim.ImageType.Segmentation, False, False) 
        ])
        rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)
        seg = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).reshape(responses[1].height, responses[1].width, 3)

        classMap = segmentationRGBToClassMap(seg)
        imgName = f"{frameId:05d}.png"
        cv2.imwrite(os.path.join(IMG_DIR, imgName), rgb)
        labelPath = os.path.join(LBL_DIR, imgName.replace(".png", ".txt"))
        
        total_boxes = 0
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        class_names = {0: "Car", 1: "Hedge", 2: "Tree", 3: "House"}
        
        with open(labelPath, "w") as f:
            for classId in [0, 1, 2, 3]:
                boxes = segToBoxes(classMap, classId)
                class_counts[classId] = len(boxes)
                total_boxes += len(boxes)
                for box in boxes:
                    f.write(f"{classId} {' '.join(map(str, box))}\n")
        
        # Detailed output
        details = ", ".join([f"{class_names[cid]}:{class_counts[cid]}" for cid in [0, 1, 2, 3] if class_counts[cid] > 0])
        print(f"[DATASET] Frame {frameId:05d} saved - Total: {total_boxes} ({details})")
        frameId += 1
        return imgName
    except Exception as e:
        print(f"[ERROR] Failed to capture dataset: {e}")
        return ""


print("\n" + "=" * 60)
print("MANUAL CONTROL ACTIVE")
print("=" * 60)
print("Controls:")
print("  Y = Takeoff")
print("  L = Land")
print("  W/S = Forward/Backward")
print("  A/D = Left/Right")
print("  Z/C = Up/Down")
print("  Q/E = Rotate Left/Right")
print("  R = Toggle Recording (saves images & labels)")
print("  ESC = Exit")
print("=" * 60)
print("\n📸 QUICK START:")
print("  1. Press Y to takeoff")
print("  2. Press R to START recording")
print("  3. Fly around to collect data")
print("  4. Press R again to STOP recording")
print("  5. Press L to land")
print("=" * 60 + "\n")

log_entries = 0

while running:
    vx = vy = vz = 0.0
    YawRate = 0.0
    currentCmd = "idle"
    
    # Show recording status every 10 seconds if recording is active
    if recording and flying and (time.time() - lastStatusUpdate >= 10):
        print(f"[STATUS] Recording: ON | Images captured: {frameId} | Flying: {flying}")
        lastStatusUpdate = time.time()
    
    time.sleep(0.05)

    if keyboard.is_pressed("y") and not flying:
        print("\n[AIRSIM] Taking off...")
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(-10, 2).join()
        flying = True
        currentCmd = "takeoff"
        print("[AIRSIM] ✓ Drone in the air!\n")
    
    if keyboard.is_pressed("l") and flying:
        print("\n[AIRSIM] Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        flying = False
        currentCmd = "land"
        print("[AIRSIM] ✓ Drone landed\n")

    # Recording toggle - works anytime
    if keyboard.is_pressed("r"):
        recording = not recording
        status = "ON ✓" if recording else "OFF"
        print(f"\n{'='*60}")
        print(f"[RECORDING] {status}")
        if recording:
            if flying:
                print("[RECORDING] Images will be saved every 2 seconds")
            else:
                print("[RECORDING] Enabled - will start capturing after takeoff")
            print("[RECORDING] Press R again to stop recording")
        else:
            print("[RECORDING] Image capture stopped")
        print(f"{'='*60}\n")
        time.sleep(0.3)

    # Exit - works anytime
    if keyboard.is_pressed("esc"):
        running = False
        break

    if not flying:
        time.sleep(0.05)
        continue

    if keyboard.is_pressed("w"):
        vx = VEL
        currentCmd = "forward"
    elif keyboard.is_pressed("s"):
        vx = -VEL
        currentCmd = "backward"
    
    if keyboard.is_pressed("a"):
        vy = -VEL
        currentCmd = "left"
    elif keyboard.is_pressed("d"):
        vy = VEL
        currentCmd = "right"
    
    if keyboard.is_pressed("z"):
        vz = -VEL
        currentCmd = "up"
    elif keyboard.is_pressed("c"):
        vz = VEL
        currentCmd = "down"
    
    if keyboard.is_pressed("q"):
        yawRate = YAW_RATE
        currentCmd = "yawLeft"
    elif keyboard.is_pressed("e"):
        yawRate = -YAW_RATE
        currentCmd = "yawRight"

    # Handle movement and yaw
    if yawRate != 0:
        client.rotateByYawRateAsync(yawRate, 0.1)
        
    if vx != 0 or vy != 0 or vz != 0:
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1)
    elif yawRate == 0:
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.1)
    
    # Capture dataset if recording
    if recording and (time.time() - lastCapture >= CAPTURE_INTERVAL):
        imageName = captureDataset()
        imageSaved = 1
        lastCapture = time.time()
    else:
        imageSaved = 0
        imageName = ""

    # Get telemetry
    try:
        pose = client.simGetVehiclePose()
        gps = client.getGpsData()
        pos = pose.position
        q = pose.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(q)
        
        # Log data
        current_time = time.time()
        writer.writerow([
            current_time, currentCmd,
            imageSaved, imageName,
            vx, vy, vz, yawRate,
            pos.x_val, pos.y_val, pos.z_val,
            roll, pitch, yaw,
            gps.gnss.geo_point.latitude,
            gps.gnss.geo_point.longitude,
            gps.gnss.geo_point.altitude
        ])
        logFile.flush()
        log_entries += 1
        
        # Show progress every 50 entries
        if log_entries % 50 == 0:
            print(f"[LOG] Logged {log_entries} entries...")
    except Exception as e:
        print(f"[ERROR] Failed to log data: {e}")

# Cleanup
print("\n" + "=" * 60)
print("SHUTTING DOWN")
print("=" * 60)

if flying:
    print("[AIRSIM] Landing drone...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[AIRSIM] ✓ Drone landed safely")

# Close log file
try:
    logFile.close()
    print(f"[LOG] ✓ Log file closed successfully")
    print(f"[LOG] Total entries logged: {log_entries}")
    print(f"[LOG] Saved to: {log_path}")
except Exception as e:
    print(f"[ERROR] Failed to close log file: {e}")

print("\n" + "=" * 60)
print("SESSION COMPLETE")
print("=" * 60)
print(f"Session folder: {SESSION_DIR}")
print(f"Images collected: {frameId}")
print(f"Log entries: {log_entries}")
print("=" * 60)

if frameId == 0:
    print("\n⚠️  WARNING: No images were collected!")
    print("⚠️  Did you forget to press 'R' to start recording?")
    print("=" * 60)
    


                      


