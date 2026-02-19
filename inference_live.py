"""
Live AirSim Inference Script
Real-time object detection using trained YOLOv11 model
Controls drone manually while running live detection
"""

import airsim
from ultralytics import YOLO
import cv2
import numpy as np
import time
import keyboard
import os
from datetime import datetime

# Configuration
MODEL_PATH = 'runs/detect/runs/train/airsim_yolo_20260217_212003/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DETECTION_INTERVAL = 0.1  # Run detection every 100ms
SAVE_DETECTIONS = True
OUTPUT_DIR = 'inference_results/live'

# Flight Configuration
VELOCITY = 2.0
YAW_RATE = 25

# Class names and colors
CLASS_NAMES = {0: 'Car', 1: 'Hedge', 2: 'Tree', 3: 'House'}
CLASS_COLORS = {
    0: (0, 255, 255),    # Car - Yellow
    1: (255, 0, 255),    # Hedge - Magenta
    2: (0, 255, 0),      # Tree - Green
    3: (255, 0, 0)       # House - Blue
}

# Global statistics
stats = {
    'total_frames': 0,
    'total_detections': 0,
    'class_detections': {0: 0, 1: 0, 2: 0, 3: 0},
    'avg_inference_time': 0,
    'start_time': None,
    'saved_frames': 0
}

def load_model():
    """Load trained YOLO model"""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print(f"Please ensure training is complete and model exists at: {MODEL_PATH}")
        return None
    
    print(f"Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded successfully")
    return model

def connect_airsim():
    """Connect to AirSim and setup drone"""
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)
    
    # Move to starting position (altitude 10m)
    client.moveToPositionAsync(0, 0, -10, 3).join()
    
    print("✓ Drone ready for live inference")
    return client

def process_keyboard_input():
    """Get velocity commands from keyboard"""
    vx = vy = vz = yaw_rate = 0
    
    # Translation
    if keyboard.is_pressed('w'): vx = VELOCITY
    if keyboard.is_pressed('s'): vx = -VELOCITY
    if keyboard.is_pressed('a'): vy = -VELOCITY
    if keyboard.is_pressed('d'): vy = VELOCITY
    if keyboard.is_pressed('q'): vz = -VELOCITY * 0.75
    if keyboard.is_pressed('e'): vz = VELOCITY * 0.75
    
    # Rotation
    if keyboard.is_pressed('left'): yaw_rate = -YAW_RATE
    if keyboard.is_pressed('right'): yaw_rate = YAW_RATE
    
    return vx, vy, vz, yaw_rate

def draw_detections(image, results, inference_time):
    """Draw bounding boxes and statistics on image"""
    if image is None:
        return None, 0
    
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # Extract predictions
    boxes = results[0].boxes
    
    detection_count = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Draw each detection
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Count detections
        detection_count[class_id] += 1
        
        # Get color and label
        color = CLASS_COLORS[class_id]
        label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Update global statistics
    stats['total_frames'] += 1
    frame_detections = sum(detection_count.values())
    stats['total_detections'] += frame_detections
    for class_id in stats['class_detections']:
        stats['class_detections'][class_id] += detection_count[class_id]
    
    # Update average inference time
    alpha = 0.1  # Smoothing factor
    stats['avg_inference_time'] = (alpha * inference_time + 
                                   (1 - alpha) * stats['avg_inference_time'])
    
    return annotated, frame_detections

def capture_and_detect(client, model):
    """Capture image from AirSim and run detection"""
    try:
        # Capture RGB image
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ])
        
        if not responses:
            print("DEBUG: No response from simGetImages")
            return None, None, 0
            
        if len(responses[0].image_data_uint8) == 0:
            print("DEBUG: Empty image data")
            return None, None, 0
        
        # Decode image - AirSim returns RGB data, reshape it
        img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
            responses[0].height, responses[0].width, 3
        )
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if img_bgr is None or img_bgr.size == 0:
            print("DEBUG: Image conversion failed")
            return None, None, 0
        
        # Run YOLO inference
        start_time = time.time()
        results = model.predict(
            source=img_bgr,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        # Draw detections
        annotated, num_detections = draw_detections(img_bgr, results, inference_time)
        
        return img_bgr, annotated, num_detections
    
    except Exception as e:
        print(f"ERROR in capture_and_detect: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def save_detection_frame(image, frame_number):
    """Save detection frame to disk"""
    if not SAVE_DETECTIONS:
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"live_detection_{frame_number:05d}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, image)
    stats['saved_frames'] += 1

def print_session_summary():
    """Print final statistics"""
    print("\n" + "=" * 60)
    print("LIVE INFERENCE SESSION SUMMARY")
    print("=" * 60)
    
    if stats['start_time']:
        duration = time.time() - stats['start_time']
        print(f"Session Duration: {duration:.1f} seconds")
    
    print(f"Total Frames Processed: {stats['total_frames']}")
    print(f"Total Detections: {stats['total_detections']}")
    
    if stats['total_frames'] > 0:
        print(f"Average Detections/Frame: {stats['total_detections']/stats['total_frames']:.2f}")
        print(f"Average Inference Time: {stats['avg_inference_time']*1000:.1f}ms")
        print(f"Average FPS: {1.0/stats['avg_inference_time']:.1f}")
    
    print(f"\nPer-Class Detection Counts:")
    for class_id, class_name in CLASS_NAMES.items():
        count = stats['class_detections'][class_id]
        percentage = (count / stats['total_detections'] * 100) if stats['total_detections'] > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    if SAVE_DETECTIONS and stats['saved_frames'] > 0:
        print(f"\nSaved Frames: {stats['saved_frames']}")
        print(f"Output Directory: {OUTPUT_DIR}")
    
    print("=" * 60)

def main():
    print("\n" + "=" * 60)
    print("LIVE YOLO INFERENCE IN AIRSIM")
    print("=" * 60 + "\n")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Connect to AirSim
    try:
        client = connect_airsim()
    except Exception as e:
        print(f"ERROR: Could not connect to AirSim: {e}")
        print("Please ensure AirSim is running")
        return
    
    print("\n" + "=" * 60)
    print("LIVE DETECTION ACTIVE")
    print("=" * 60)
    print("Controls:")
    print("  W/A/S/D - Forward/Left/Backward/Right")
    print("  Q/E - Up/Down")
    print("  Arrow Keys - Rotate")
    print("  C - Save current detection frame")
    print("  X - Exit and land")
    print("=" * 60 + "\n")
    
    stats['start_time'] = time.time()
    frame_count = 0
    last_detection_time = 0
    
    print("Warming up camera... capturing first frame")
    
    # Test camera connection
    print("Testing camera connection...")
    test_responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ])
    if test_responses and len(test_responses[0].image_data_uint8) > 0:
        print(f"✓ Camera working! Image size: {len(test_responses[0].image_data_uint8)} bytes")
    else:
        print("✗ ERROR: Camera not responding!")
        print("Make sure AirSim is fully loaded and the drone has spawned")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        return
    
    time.sleep(1)
    print("Starting detection loop... (Press X in OpenCV window to exit)")
    
    # Create resizable window
    window_name = 'YOLOv11 Live Detection - AirSim'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Set window size
    
    try:
        while True:
            # Process keyboard input
            vx, vy, vz, yaw_rate = process_keyboard_input()
            
            # Apply movement
            client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1, 
                                               airsim.DrivetrainType.MaxDegreeOfFreedom,
                                               airsim.YawMode(True, yaw_rate))
            
            # Run detection at specified interval
            current_time = time.time()
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                original, annotated, num_detections = capture_and_detect(client, model)
                
                if annotated is not None and original is not None:
                    # Resize image to fit window for better visibility
                    display_img = cv2.resize(annotated, (1280, 720))
                    
                    # Display result
                    cv2.imshow(window_name, display_img)
                    
                    # Manual capture on 'C' key
                    if keyboard.is_pressed('c'):
                        save_detection_frame(annotated, frame_count)
                        print(f"✓ Saved frame {frame_count} ({num_detections} detections)")
                    
                    frame_count += 1
                    last_detection_time = current_time
                else:
                    # No image captured yet, show message
                    if frame_count == 0:
                        print("Waiting for camera feed...")
            
            # Check for exit (cv2 window or keyboard)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x'):
                print("\nX key pressed in window, exiting...")
                break
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        
        print("\nLanding drone...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        
        # Print summary
        print_session_summary()

if __name__ == "__main__":
    main()
