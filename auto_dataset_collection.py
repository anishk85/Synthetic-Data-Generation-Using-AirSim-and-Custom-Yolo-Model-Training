"""
Automated dataset collection script for AirSim.
Flies the drone in a systematic pattern and automatically captures labeled images.
Stops after collecting 750 images.
"""
import airsim
import keyboard
import time
import csv
import cv2
import os
import numpy as np
from datetime import datetime

# Configuration
TARGET_IMAGES = 750
CAPTURE_INTERVAL = 1.5  # seconds between captures while flying
FLIGHT_SPEED = 3.0  # m/s


# Generate diverse flight paths covering a large area
def generate_flight_plan():
    """Generate a comprehensive flight plan with multiple altitudes and patterns."""
    waypoints = []
    
    # Phase 1: Low altitude grid (closer to objects) - altitude 8m
    alt = -8
    for y in range(-80, 100, 40):
        for x in range(-80, 100, 50):
            waypoints.append((x, y, alt))
    
    # Phase 2: Medium altitude lawnmower - altitude 15m
    alt = -15
    direction = 1
    for y in range(-100, 120, 25):
        if direction == 1:
            xs = range(-100, 120, 40)
        else:
            xs = range(100, -120, -40)
        for x in xs:
            waypoints.append((x, y, alt))
        direction *= -1
    
    # Phase 3: High altitude overview - altitude 25m
    alt = -25
    for y in range(-120, 140, 50):
        for x in range(-120, 140, 60):
            waypoints.append((x, y, alt))
    
    # Phase 4: Diagonal passes at 12m
    alt = -12
    for i in range(-100, 120, 30):
        waypoints.append((i, i, alt))
    for i in range(-100, 120, 30):
        waypoints.append((i, -i, alt))
    
    # Phase 5: Circular sweeps around center at various radii
    import math
    for radius in [30, 60, 90, 120]:
        alt = -10 if radius < 50 else -18
        for angle_deg in range(0, 360, 20):
            angle = math.radians(angle_deg)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            waypoints.append((x, y, alt))
    
    # Phase 6: Extended far exploration at 15m
    alt = -15
    direction = 1
    for y in range(-150, 170, 35):
        if direction == 1:
            xs = range(-150, 170, 45)
        else:
            xs = range(150, -170, -45)
        for x in xs:
            waypoints.append((x, y, alt))
        direction *= -1
    
    # Phase 7: Low passes along roads/streets at 6m
    alt = -6
    for x in range(-100, 120, 15):
        waypoints.append((x, 0, alt))
    for y in range(-100, 120, 15):
        waypoints.append((0, y, alt))
    for x in range(-100, 120, 15):
        waypoints.append((x, 50, alt))
    for x in range(-100, 120, 15):
        waypoints.append((x, -50, alt))
    
    return waypoints

# Create session directories
session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
BASE_DIR = "flightLogs"
SESSION_DIR = os.path.join(BASE_DIR, f"auto_session_{session_time}")
IMG_DIR = os.path.join(SESSION_DIR, "images")
LBL_DIR = os.path.join(SESSION_DIR, "labels")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

log_path = os.path.join(SESSION_DIR, "flight_log.csv")
print("=" * 60)
print(f"[AUTO SESSION] Starting automated collection")
print(f"[AUTO SESSION] Target: {TARGET_IMAGES} images")
print(f"[AUTO SESSION] Folder: {SESSION_DIR}")
print(f"[AUTO SESSION] Log file: {log_path}")
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
        "time", "waypoint", "imageSaved", "imageName",
        "x", "y", "z", "roll", "pitch", "yaw",
    ])
    logFile.flush()
    print(f"[LOG] ✓ Log file created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create log file: {e}")
    exit(1)

# Setup segmentation IDs for object detection
print("\n[SEGMENTATION] Setting up object IDs...")
try:
    # Step 1: Reset ALL objects to background (ID=0)
    success = client.simSetSegmentationObjectID(r".*", 0, True)
    print(f"[SEGMENTATION] Reset all objects (regex): {success}")
    time.sleep(1.0)

    # Step 2: Set class IDs using regex patterns
    class_patterns = {
        1: [r".*Car.*"],
        2: [r".*Hedge.*"],
        3: [r".*Birch.*", r".*Tree.*"],
        4: [r".*House.*"],
    }
    class_labels = {1: "Car", 2: "Hedge", 3: "Tree", 4: "House"}

    for class_id, patterns in class_patterns.items():
        for pat in patterns:
            r = client.simSetSegmentationObjectID(pat, class_id, True)
            print(f"  {class_labels[class_id]} pattern '{pat}' -> ID {class_id}: {r}")

    # Step 3: Set individual actor names for completeness
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

    # Step 4: Use discovered color palette
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

# Image dimensions
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Minimum area thresholds for each class (in pixels)
MIN_AREA = {
    0: 150,    # Cars
    1: 200,    # Hedges
    2: 250,    # Trees
    3: 400,    # Houses
}

# Maximum aspect ratio (width/height or height/width) - per class
MAX_ASPECT_RATIO = {
    0: 4.0,    # Cars
    1: 10.0,   # Hedges
    2: 3.0,    # Trees
    3: 5.0,    # Houses
}

frameId = 0


def segmentationRGBToClassMap(SegRgb):
    h, w, _ = SegRgb.shape
    classMap = np.full((h, w), -1, dtype=np.int32)
    
    class_labels_map = {0: "Car", 1: "Hedge", 2: "Tree", 3: "House"}
    class_pixels = {}
    
    for rgb, airsimId in RGB_TO_AIRSIM_ID.items():
        mask = np.all(SegRgb == rgb, axis=-1)
        pixel_count = int(np.sum(mask))
        yolo_id = AIRSIM_TO_YOLO[airsimId]
        classMap[mask] = yolo_id
        if pixel_count > 0:
            class_pixels[yolo_id] = class_pixels.get(yolo_id, 0) + pixel_count
    
    return classMap


def merge_boxes(boxes, iou_threshold=0.3):
    """Merge overlapping boxes using IoU threshold"""
    if len(boxes) == 0:
        return boxes
    
    def box_iou(box1, box2):
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
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
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        merged_xmin = min(x1_min, x2_min)
        merged_ymin = min(y1_min, y2_min)
        merged_xmax = max(x1_max, x2_max)
        merged_ymax = max(y1_max, y2_max)
        
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
    """Extract bounding boxes from segmentation mask"""
    mask = (ClassMap == classId).astype(np.uint8) * 255
    
    kernel_sizes = {0: 7, 1: 5, 2: 5, 3: 9}
    kernel_size = kernel_sizes.get(classId, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = MIN_AREA.get(classId, 150)
    max_aspect = MAX_ASPECT_RATIO.get(classId, 5.0)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        if w * h <= min_area:
            continue
        
        aspect_ratio = max(w / h, h / w) if h > 0 else 0
        if aspect_ratio > max_aspect:
            continue
        
        padding = 0.02
        x_pad = int(w * padding)
        y_pad = int(h * padding)
        x = max(0, x - x_pad)
        y = max(0, y - y_pad)
        w = min(IMG_WIDTH - x, w + 2 * x_pad)
        h = min(IMG_HEIGHT - y, h + 2 * y_pad)
        
        boxes.append((
            (x + w / 2) / IMG_WIDTH,
            (y + h / 2) / IMG_HEIGHT,
            w / IMG_WIDTH,
            h / IMG_HEIGHT
        ))
    
    iou_thresholds = {0: 0.4, 1: 0.3, 2: 0.35, 3: 0.5}
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
        
        details = ", ".join([f"{class_names[cid]}:{class_counts[cid]}" for cid in [0, 1, 2, 3] if class_counts[cid] > 0])
        if details:
            print(f"[CAPTURE] Frame {frameId:05d} - Total: {total_boxes} ({details})")
        else:
            print(f"[CAPTURE] Frame {frameId:05d} - No objects detected")
        frameId += 1
        return imgName
    except Exception as e:
        print(f"[ERROR] Failed to capture: {e}")
        return ""


VEL = 2.5
YAW_RATE = 25


def check_collision():
    """Check if drone has collided."""
    collision_info = client.simGetCollisionInfo()
    return collision_info.has_collided


def manual_override():
    """Enter manual control mode with continuous capture. Returns 'resume' or 'skip'."""
    global frameId
    print("\n" + "=" * 60)
    print("⚠️  MANUAL OVERRIDE ACTIVE (still capturing!)")
    print("=" * 60)
    print("Controls:")
    print("  W/S = Forward/Backward")
    print("  A/D = Left/Right")
    print("  Z/C = Up/Down")
    print("  Q/E = Rotate Left/Right")
    print("  G   = Resume automated flight")
    print("  N   = Skip to next waypoint")
    print("=" * 60 + "\n")
    
    # Reset collision state
    client.simSetVehiclePose(client.simGetVehiclePose(), True)
    time.sleep(0.3)
    
    last_capture = time.time()
    result = "resume"
    
    while True:
        vx = vy = vz = 0.0
        yawRate = 0.0
        
        if keyboard.is_pressed("g"):
            print("\n[MANUAL] Resuming automated flight...")
            result = "resume"
            time.sleep(0.5)  # Debounce
            break
        
        if keyboard.is_pressed("n"):
            print("\n[MANUAL] Skipping to next waypoint...")
            result = "skip"
            time.sleep(0.5)  # Debounce
            break
        
        if keyboard.is_pressed("w"):
            vx = VEL
        elif keyboard.is_pressed("s"):
            vx = -VEL
        
        if keyboard.is_pressed("a"):
            vy = -VEL
        elif keyboard.is_pressed("d"):
            vy = VEL
        
        if keyboard.is_pressed("z"):
            vz = -VEL  # Up
        elif keyboard.is_pressed("c"):
            vz = VEL   # Down
        
        if keyboard.is_pressed("q"):
            yawRate = YAW_RATE
        elif keyboard.is_pressed("e"):
            yawRate = -YAW_RATE
        
        if yawRate != 0:
            client.rotateByYawRateAsync(yawRate, 0.1)
        
        if vx != 0 or vy != 0 or vz != 0:
            client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1)
        elif yawRate == 0:
            client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.1)
        
        # Keep capturing while in manual mode
        if frameId < TARGET_IMAGES and (time.time() - last_capture) >= CAPTURE_INTERVAL:
            captureDataset()
            last_capture = time.time()
        
        time.sleep(0.05)
    
    return result


def fly_to_waypoint_with_capture(wx, wy, wz, wp_idx):
    """Fly to waypoint while capturing images. Handles collisions."""
    global frameId
    
    # Start moving (non-blocking)
    client.moveToPositionAsync(wx, wy, wz, FLIGHT_SPEED)
    
    # Calculate travel time
    pose = client.simGetVehiclePose()
    dx = wx - pose.position.x_val
    dy = wy - pose.position.y_val
    dz = wz - pose.position.z_val
    distance = (dx**2 + dy**2 + dz**2) ** 0.5
    travel_time = max(distance / FLIGHT_SPEED, 1.0)
    
    # Capture continuously while flying
    captures_this_leg = 0
    max_captures_per_leg = 5
    capture_spacing = travel_time / (max_captures_per_leg + 1)
    capture_spacing = max(capture_spacing, CAPTURE_INTERVAL)
    
    start_time = time.time()
    last_capture_time = 0
    stuck_check_time = time.time()
    last_pos = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
    
    while (time.time() - start_time) < travel_time + 5:  # +5s buffer
        if frameId >= TARGET_IMAGES:
            break
        
        # Check for manual override key (H = help/manual)
        if keyboard.is_pressed("h"):
            print("[FLIGHT] Manual override requested!")
            action = manual_override()
            if action == "skip":
                return  # Skip this waypoint
            # Resume movement toward waypoint
            client.moveToPositionAsync(wx, wy, wz, FLIGHT_SPEED)
            start_time = time.time()  # Reset timer
            last_capture_time = 0
            captures_this_leg = 0
            stuck_check_time = time.time()
            continue
        
        # Check for N key to skip waypoint directly
        if keyboard.is_pressed("n"):
            print(f"[FLIGHT] Skipping waypoint ({wx:.0f}, {wy:.0f}, {wz:.0f})...")
            time.sleep(0.3)
            return
        
        # Check for collision
        if check_collision():
            print(f"\n⚠️  COLLISION DETECTED! Press H for manual control or waiting 3s to auto-recover...")
            
            # Wait for user to press H or N, or auto-recover after 3s
            collision_time = time.time()
            recovered = False
            while (time.time() - collision_time) < 3.0:
                if keyboard.is_pressed("h"):
                    action = manual_override()
                    if action == "skip":
                        return
                    recovered = True
                    break
                if keyboard.is_pressed("n"):
                    print("[FLIGHT] Skipping waypoint...")
                    time.sleep(0.3)
                    return
                time.sleep(0.1)
            
            if not recovered:
                # Auto-recover: back up and go higher
                print("[FLIGHT] Auto-recovering: backing up and climbing...")
                client.moveByVelocityBodyFrameAsync(-2, 0, -2, 2).join()  # Back up and climb
                time.sleep(1)
            
            # Resume toward waypoint
            client.moveToPositionAsync(wx, wy, wz, FLIGHT_SPEED)
            start_time = time.time()
            last_capture_time = 0
            captures_this_leg = 0
            stuck_check_time = time.time()
            continue
        
        # Check if stuck (no movement for 5 seconds)
        if time.time() - stuck_check_time > 5.0:
            cur_pose = client.simGetVehiclePose()
            cur_pos = (cur_pose.position.x_val, cur_pose.position.y_val, cur_pose.position.z_val)
            moved = ((cur_pos[0]-last_pos[0])**2 + (cur_pos[1]-last_pos[1])**2 + (cur_pos[2]-last_pos[2])**2) ** 0.5
            
            if moved < 0.5:  # Less than 0.5m in 5 seconds = stuck
                print(f"\n⚠️  DRONE APPEARS STUCK! Press H=manual, N=skip waypoint")
                wait_start = time.time()
                handled = False
                while (time.time() - wait_start) < 3.0:
                    if keyboard.is_pressed("h"):
                        action = manual_override()
                        if action == "skip":
                            return
                        handled = True
                        break
                    if keyboard.is_pressed("n"):
                        print("[FLIGHT] Skipping waypoint...")
                        time.sleep(0.3)
                        return
                    time.sleep(0.1)
                if not handled:
                    # Auto-recover
                    print("[FLIGHT] Auto-recovering: climbing up...")
                    client.moveByVelocityBodyFrameAsync(0, 0, -3, 2).join()
                    time.sleep(1)
                
                client.moveToPositionAsync(wx, wy, wz, FLIGHT_SPEED)
                start_time = time.time()
                last_capture_time = 0
                captures_this_leg = 0
            
            last_pos = cur_pos
            stuck_check_time = time.time()
        
        elapsed = time.time() - start_time
        
        # Capture at evenly spaced intervals while in transit
        if elapsed - last_capture_time >= capture_spacing and captures_this_leg < max_captures_per_leg:
            captureDataset()
            captures_this_leg += 1
            last_capture_time = elapsed
            
            # Log telemetry
            try:
                p = client.simGetVehiclePose()
                pos = p.position
                q = p.orientation
                roll, pitch, yaw = airsim.to_eularian_angles(q)
                writer.writerow([
                    time.time(), f"wp{wp_idx}_transit", 1, f"{frameId-1:05d}.png",
                    pos.x_val, pos.y_val, pos.z_val,
                    roll, pitch, yaw
                ])
                logFile.flush()
            except:
                pass
        
        time.sleep(0.1)
    
    time.sleep(0.3)


# Main flight routine
print("\n" + "=" * 60)
print("AUTOMATED FLIGHT STARTING")
print("=" * 60)
print("  H = Manual override (fly manually, still captures!)")
print("  N = Skip to next waypoint")
print("  G = Resume auto (when in manual mode)")
print("  ESC = Abort and land")
print("=" * 60)

try:
    # Enable API control
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Takeoff
    print("\n[FLIGHT] Taking off...")
    client.takeoffAsync().join()
    print("[FLIGHT] ✓ Airborne!")
    
    # Move to starting altitude
    print(f"[FLIGHT] Climbing to initial altitude...")
    client.moveToZAsync(-15, 3).join()
    time.sleep(1)
    
    # Generate waypoints
    waypoints = generate_flight_plan()
    print(f"\n[FLIGHT] Generated {len(waypoints)} waypoints across multiple patterns")
    print(f"[FLIGHT] Flight speed: {FLIGHT_SPEED} m/s")
    print(f"[FLIGHT] Capture interval: {CAPTURE_INTERVAL}s (continuous while flying)")
    print(f"[FLIGHT] Target: {TARGET_IMAGES} images")
    print("=" * 60 + "\n")
    
    # Fly through waypoints with continuous capture
    for wp_idx in range(len(waypoints)):
        if frameId >= TARGET_IMAGES:
            print(f"\n[FLIGHT] ✓ Target reached: {frameId} images collected!")
            break
        
        # Check for ESC to abort
        if keyboard.is_pressed("esc"):
            print("\n[FLIGHT] ESC pressed - aborting...")
            break
        
        wx, wy, wz = waypoints[wp_idx]
        
        if wp_idx % 10 == 0:
            print(f"\n[FLIGHT] Waypoint {wp_idx+1}/{len(waypoints)} -> ({wx:.0f}, {wy:.0f}, {wz:.0f}m) | Images: {frameId}/{TARGET_IMAGES}")
        
        fly_to_waypoint_with_capture(wx, wy, wz, wp_idx)
    
    # If still haven't reached target, do another pass
    if frameId < TARGET_IMAGES:
        print(f"\n[FLIGHT] First pass done ({frameId}/{TARGET_IMAGES}). Starting extended exploration...")
        
        import random
        random.seed(42)
        
        while frameId < TARGET_IMAGES:
            if keyboard.is_pressed("esc"):
                print("\n[FLIGHT] ESC pressed - aborting...")
                break
            
            x = random.uniform(-150, 150)
            y = random.uniform(-150, 150)
            z = random.choice([-6, -8, -10, -12, -15, -20, -25])
            
            print(f"[FLIGHT] Exploring ({x:.0f}, {y:.0f}, {z:.0f}m) | Images: {frameId}/{TARGET_IMAGES}")
            fly_to_waypoint_with_capture(x, y, z, -1)
    
    print("\n" + "=" * 60)
    print("AUTOMATED FLIGHT COMPLETE")
    print("=" * 60)
    print(f"Images collected: {frameId}/{TARGET_IMAGES}")
    
except KeyboardInterrupt:
    print(f"\n[FLIGHT] Interrupted by user. Images collected: {frameId}")
except Exception as e:
    print(f"\n[ERROR] Flight error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Landing
    print("\n[FLIGHT] Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[FLIGHT] ✓ Landed safely")
    
    # Close log file
    try:
        logFile.close()
        print(f"[LOG] ✓ Log file closed")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print(f"Session folder: {SESSION_DIR}")
    print(f"Images collected: {frameId}")
    print("=" * 60)
