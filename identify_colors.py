"""
Visual color identifier - helps map RGB colors to object classes
Opens the debug segmentation image and lets you click to identify colors
"""
import cv2
import numpy as np

# Load the debug segmentation image
seg = cv2.imread('debug_segmentation.png')
if seg is None:
    print("Error: debug_segmentation.png not found! Run find_seg_colors.py first.")
    exit()

seg_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

# Create a copy for display
display = seg.copy()

print("\n" + "="*60)
print("SEGMENTATION COLOR IDENTIFIER")
print("="*60)
print("Click on objects in the image to see their RGB colors:")
print("  - Click on a CAR to find Car RGB")
print("  - Click on a HEDGE to find Hedge RGB")  
print("  - Click on a TREE to find Tree RGB")
print("  - Click on a HOUSE to find House RGB")
print("\nPress 'q' to quit")
print("="*60 + "\n")

clicked_colors = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get RGB color at clicked position
        b, g, r = seg[y, x]
        rgb = (r, g, b)
        
        print(f"\nClicked at ({x}, {y})")
        print(f"  RGB: ({r:3d}, {g:3d}, {b:3d})")
        
        # Count pixels of this color
        mask = np.all(seg_rgb == rgb, axis=2)
        count = np.sum(mask)
        percentage = (count / (seg.shape[0] * seg.shape[1])) * 100
        print(f"  Pixels: {count:7d} ({percentage:5.2f}%)")
        
        # Add to list
        if rgb not in clicked_colors:
            clicked_colors.append(rgb)
            print(f"  Added to list (#{len(clicked_colors)})")

# Create window and set mouse callback
cv2.namedWindow('Segmentation - Click on objects')
cv2.setMouseCallback('Segmentation - Click on objects', mouse_callback)

# Display loop
while True:
    cv2.imshow('Segmentation - Click on objects', display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Print summary
if clicked_colors:
    print("\n" + "="*60)
    print("IDENTIFIED COLORS:")
    print("="*60)
    print("\nCopy these to oddDatasetmanualcontrol.py RGB_TO_AIRSIM_ID:")
    print("\nRGB_TO_AIRSIM_ID = {")
    for i, color in enumerate(clicked_colors[:4], 1):
        class_names = ["Car", "Hedge", "Tree", "House"]
        class_name = class_names[i-1] if i <= 4 else f"Class{i}"
        print(f"    {color}: {i},  # {class_name}")
    print("}")
    
    print("\nAIRSIM_TO_YOLO = {")
    for i in range(1, min(len(clicked_colors) + 1, 5)):
        print(f"    {i}: {i-1},")
    print("}")
else:
    print("\nNo colors clicked!")
