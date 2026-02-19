"""
YOLOv11 Testing Script - Test trained model on images or real-time AirSim feed
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
import airsim

# Configuration
MODEL_PATH = "runs/detect/runs/train/airsim_yolo_*/weights/best.pt"  # Will find latest model
CLASS_NAMES = ['Car', 'Hedge', 'Tree', 'House']
COLORS = [
    (255, 0, 0),      # Car - Red
    (34, 139, 34),    # Hedge - Forest Green
    (0, 200, 0),      # Tree - Bright Green
    (255, 0, 255),    # House - Magenta
]

def find_latest_model():
    """Find the latest trained model"""
    models = glob.glob(MODEL_PATH)
    if not models:
        print(f"ERROR: No trained model found at {MODEL_PATH}")
        print("Please train a model first using train_yolo.py")
        return None
    
    # Get the most recent model
    latest_model = max(models, key=os.path.getctime)
    print(f"Using model: {latest_model}")
    return latest_model

def test_on_images(model, image_folder="dataset/test/images", save_results=True):
    """Test model on a folder of images"""
    print("\n" + "=" * 60)
    print("TESTING ON IMAGES")
    print("=" * 60 + "\n")
    
    image_files = glob.glob(f"{image_folder}/*.png") + glob.glob(f"{image_folder}/*.jpg")
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to test\n")
    
    output_folder = "test_results"
    if save_results:
        os.makedirs(output_folder, exist_ok=True)
    
    for idx, img_path in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # Run inference
        results = model(img_path, conf=0.25, iou=0.45)
        
        # Get the annotated image
        annotated = results[0].plot()
        
        # Display
        cv2.imshow('YOLOv11 Detection', annotated)
        key = cv2.waitKey(1000)  # Show for 1 second
        
        if save_results:
            output_path = os.path.join(output_folder, f"result_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, annotated)
        
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if save_results:
        print(f"\n✓ Results saved to: {output_folder}")

def test_airsim_realtime(model, confidence=0.25):
    """Test model on real-time AirSim camera feed"""
    print("\n" + "=" * 60)
    print("REAL-TIME AIRSIM DETECTION")
    print("=" * 60 + "\n")
    
    print("Connecting to AirSim...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ Connected to AirSim")
    except Exception as e:
        print(f"ERROR: Could not connect to AirSim: {e}")
        return
    
    print("\nStarting real-time detection...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    try:
        while True:
            # Get image from AirSim (camera "0" is default)
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if responses and len(responses[0].image_data_uint8) > 0:
                # Convert to numpy array
                img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img.reshape(responses[0].height, responses[0].width, 3)
                
                # Run detection
                results = model(img, conf=confidence, verbose=False)
            
            # Get annotated image
            annotated = results[0].plot()
            
            # Add FPS
            frame_count += 1
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('YOLOv11 Real-time Detection - AirSim', annotated)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Warning: No image data received")
    
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print(f"\n✓ Processed {frame_count} frames")

def benchmark_model(model, test_images=100):
    """Benchmark model performance"""
    print("\n" + "=" * 60)
    print("MODEL BENCHMARK")
    print("=" * 60 + "\n")
    
    image_files = glob.glob("dataset/test/images/*.png")[:test_images]
    
    if not image_files:
        print("No test images found!")
        return
    
    import time
    
    print(f"Benchmarking on {len(image_files)} images...\n")
    
    times = []
    for img_path in image_files:
        start = time.time()
        results = model(img_path, conf=0.25, verbose=False)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    fps = 1 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Min time: {min(times)*1000:.2f} ms")
    print(f"Max time: {max(times)*1000:.2f} ms")

def main():
    print("\n" + "=" * 60)
    print("YOLOV11 MODEL TESTING")
    print("=" * 60)
    
    # Find and load model
    model_path = find_latest_model()
    if not model_path:
        return
    
    print("\nLoading model...")
    model = YOLO(model_path)
    print("✓ Model loaded successfully\n")
    
    # Menu
    print("=" * 60)
    print("SELECT TEST MODE:")
    print("=" * 60)
    print("1. Test on images (from test dataset)")
    print("2. Test on custom image folder")
    print("3. Real-time detection on AirSim")
    print("4. Benchmark model performance")
    print("5. Exit")
    print("=" * 60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        test_on_images(model, "dataset/test/images")
    elif choice == '2':
        folder = input("Enter image folder path: ").strip()
        test_on_images(model, folder)
    elif choice == '3':
        test_airsim_realtime(model)
    elif choice == '4':
        benchmark_model(model)
    elif choice == '5':
        print("Exiting...")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
