"""
Image-wise Inference Script
Runs trained YOLOv11 model on test images and saves predictions
"""

from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = 'runs/detect/runs/train/airsim_yolo_20260217_212003/weights/best.pt'
TEST_IMAGES = 'dataset/test/images'
OUTPUT_DIR = 'inference_results/images'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Class names and colors
CLASS_NAMES = {0: 'Car', 1: 'Hedge', 2: 'Tree', 3: 'House'}
CLASS_COLORS = {
    0: (0, 255, 255),    # Car - Yellow
    1: (255, 0, 255),    # Hedge - Magenta
    2: (0, 255, 0),      # Tree - Green
    3: (255, 0, 0)       # House - Blue
}

def create_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_model():
    """Load trained YOLO model"""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first using train_yolo.py")
        return None
    
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded successfully")
    return model

def get_test_images():
    """Get list of test images"""
    image_paths = glob.glob(os.path.join(TEST_IMAGES, '*.png'))
    
    if not image_paths:
        print(f"ERROR: No images found in {TEST_IMAGES}")
        return []
    
    print(f"Found {len(image_paths)} test images")
    return sorted(image_paths)

def draw_predictions(image, results):
    """Draw bounding boxes and labels on image"""
    annotated = image.copy()
    
    # Extract predictions
    boxes = results[0].boxes
    
    detection_count = {0: 0, 1: 0, 2: 0, 3: 0}
    
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
    
    # Draw summary at top
    summary = f"Detections: Car:{detection_count[0]} Hedge:{detection_count[1]} Tree:{detection_count[2]} House:{detection_count[3]}"
    cv2.rectangle(annotated, (10, 10), (750, 45), (0, 0, 0), -1)
    cv2.putText(annotated, summary, (15, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated, sum(detection_count.values()), detection_count

def run_inference_on_images(model, image_paths):
    """Run inference on all test images"""
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE ON TEST IMAGES")
    print("=" * 60 + "\n")
    
    total_detections = 0
    class_totals = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for idx, image_path in enumerate(image_paths):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not read {image_path}")
            continue
        
        # Run inference
        results = model.predict(
            image, 
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        # Draw predictions
        annotated, num_detections, class_counts = draw_predictions(image, results)
        
        # Update totals
        total_detections += num_detections
        for class_id in class_totals:
            class_totals[class_id] += class_counts[class_id]
        
        # Save annotated image
        filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
        cv2.imwrite(output_path, annotated)
        
        # Print progress
        print(f"[{idx+1}/{len(image_paths)}] {filename}: {num_detections} objects detected "
              f"(C:{class_counts[0]} H:{class_counts[1]} T:{class_counts[2]} Hs:{class_counts[3]})")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total Images Processed: {len(image_paths)}")
    print(f"Total Detections: {total_detections}")
    print(f"Average Detections/Image: {total_detections/len(image_paths):.2f}")
    print(f"\nPer-Class Detections:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_name}: {class_totals[class_id]} ({class_totals[class_id]/total_detections*100:.1f}%)")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)

def show_sample_predictions(image_paths, num_samples=5):
    """Display sample predictions interactively"""
    print(f"\nDisplaying {num_samples} sample predictions...")
    print("Press any key to see next image, ESC to exit")
    
    sample_indices = np.linspace(0, len(image_paths)-1, num_samples, dtype=int)
    
    for idx in sample_indices:
        output_path = os.path.join(OUTPUT_DIR, f"pred_{os.path.basename(image_paths[idx])}")
        
        if not os.path.exists(output_path):
            continue
        
        img = cv2.imread(output_path)
        
        # Resize for display if too large
        h, w = img.shape[:2]
        if w > 1280:
            scale = 1280 / w
            img = cv2.resize(img, (1280, int(h * scale)))
        
        cv2.imshow('YOLO Inference Results', img)
        key = cv2.waitKey(0)
        
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def batch_inference_mode(model):
    """Run YOLO's built-in batch prediction"""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE MODE (YOLO Built-in)")
    print("=" * 60 + "\n")
    
    # Run batch prediction
    results = model.predict(
        source=TEST_IMAGES,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        save_txt=True,
        save_conf=True,
        project='inference_results',
        name='batch_yolo',
        exist_ok=True
    )
    
    print(f"\n✓ Batch inference complete")
    print(f"Results saved to: inference_results/batch_yolo")

def main():
    print("\n" + "=" * 60)
    print("YOLO IMAGE INFERENCE")
    print("=" * 60 + "\n")
    
    # Create output directory
    create_output_dir()
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get test images
    image_paths = get_test_images()
    if not image_paths:
        return
    
    # Print model info
    print(f"\nModel Configuration:")
    print(f"  Architecture: YOLOv11n")
    print(f"  Parameters: 2.58M")
    print(f"  Input Size: 640x640")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  IoU Threshold: {IOU_THRESHOLD}")
    print(f"  Classes: {len(CLASS_NAMES)}")
    
    # Run inference
    run_inference_on_images(model, image_paths)
    
    # Show sample predictions
    show_sample_predictions(image_paths, num_samples=5)
    
    # Optional: Batch mode
    print("\nWould you like to run batch inference mode? (y/n)")
    # For automation, skip this
    # batch_inference_mode(model)

if __name__ == "__main__":
    main()
