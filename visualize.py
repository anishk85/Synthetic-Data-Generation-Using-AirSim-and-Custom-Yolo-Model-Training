import cv2
import os
import glob

# Class names and colors
CLASS_NAMES = {
    0: "Car",
    1: "Hedge",
    2: "Tree",
    3: "House"
}

CLASS_COLORS = {
    0: (0, 0, 255),      # Red for cars
    1: (0, 255, 0),      # Green for hedges
    2: (0, 200, 0),      # Bright green for trees
    3: (255, 0, 255),    # Magenta for houses
}

# Paths - automatically find the latest session
BASE_DIR = r"C:\Users\anish\Documents\AirSim\flightLogs"
sessions = sorted(glob.glob(os.path.join(BASE_DIR, "session_*")))
if sessions:
    latest_session = sessions[-1]
    print(f"Using latest session: {os.path.basename(latest_session)}")
    
    # Find all images
    images = sorted(glob.glob(os.path.join(latest_session, "images", "*.png")))
    if not images:
        print("No images found!")
        exit()
    
    print(f"Found {len(images)} images. Close window to see next image.")
    
    def show_image(idx):
        image_path = images[idx]
        label_path = image_path.replace("\\images\\", "\\labels\\").replace(".png", ".txt")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return
        
        h, w, _ = img.shape
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Read label file if exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                data = line.strip().split()
                if len(data) < 5:
                    continue
                
                class_id = int(data[0])
                if class_id < 4:  # Only count valid classes
                    class_counts[class_id] += 1
                x_center = float(data[1]) * w
                y_center = float(data[2]) * h
                box_width = float(data[3]) * w
                box_height = float(data[4]) * h
                
                # Convert YOLO → top-left corner format
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)
                
                # Get color and name
                color = CLASS_COLORS.get(class_id, (0, 255, 0))
                name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                
                # Draw rectangle with thicker line
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add label with background
                label = name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness)
        
        # Add info overlay
        info_y = 30
        cv2.rectangle(img, (10, 5), (400, 125), (0, 0, 0), -1)
        cv2.putText(img, f"Image: {os.path.basename(image_path)} ({idx+1}/{len(images)})", 
                    (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 25
        for cid, name in CLASS_NAMES.items():
            if class_counts[cid] > 0:
                cv2.putText(img, f"{name}: {class_counts[cid]}", 
                            (15, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cid], 2)
                info_y += 20
        
        cv2.putText(img, "Close window to see next image | Press Q to quit", 
                    (15, info_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Labeled Image", img)
    
    # Loop through all images starting from first
    for idx in range(len(images)):
        print(f"Showing image {idx+1}/{len(images)}: {os.path.basename(images[idx])}")
        show_image(idx)
        
        # Wait for key press or window close
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC to quit
            print("Quitting visualization.")
            break
    
    cv2.destroyAllWindows()
    print("Visualization complete!")
else:
    print("No sessions found!")
