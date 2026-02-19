# 🚀 Complete YOLOv11 Training Guide for AirSim Data

## 📋 Overview
This guide will walk you through collecting data from AirSim and training a custom YOLOv11 object detection model.

## 🎯 Dataset Classes
Your model will detect 4 classes:
- **Class 0: House** (Buildings, walls, doors, windows)
- **Class 1: Tree** (Vegetation, trees)
- **Class 2: Car** (Vehicles)
- **Class 3: Road** (Roads, paths)

---

## 📝 Step-by-Step Instructions

### **STEP 1: Collect Data from AirSim**

1. **Start AirSim Environment**
   - Launch your Unreal Engine environment with AirSim
   - Make sure the environment is running and ready

2. **Run Data Collection Script**
   ```powershell
   python oddDatasetmanualcontrol.py
   ```

3. **Fly and Collect Data**
   - **Y** - Takeoff
   - **W/S** - Forward/Backward
   - **A/D** - Left/Right
   - **Z/C** - Up/Down
   - **Q/E** - Rotate Left/Right
   - **R** - Start/Stop Recording (Toggle)
   - **L** - Land
   - **ESC** - Exit

4. **Tips for Good Data Collection**
   - Fly at different altitudes
   - Capture objects from different angles
   - Include varied lighting conditions
   - Ensure objects are at different distances
   - Collect at least 500-1000 images minimum
   - More data = better model!

5. **Data Location**
   - Data is saved in: `flightLogs/session_[timestamp]/`
   - Images: `images/*.png`
   - Labels: `labels/*.txt` (YOLO format)

---

### **STEP 2: Install Required Packages**

```powershell
pip install -r requirements_yolo.txt
```

**Verify GPU (if available):**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### **STEP 3: Prepare Dataset**

```powershell
python prepare_dataset.py
```

This script will:
- ✅ Collect all images and labels from session folders
- ✅ Split data into train/val/test (70%/20%/10%)
- ✅ Organize in YOLO format
- ✅ Verify data integrity
- ✅ Analyze class distribution

**Expected Output:**
```
dataset/
  train/
    images/     (70% of data)
    labels/
  val/
    images/     (20% of data)
    labels/
  test/
    images/     (10% of data)
    labels/
```

---

### **STEP 4: Train the Model**

```powershell
python train_yolo.py
```

**Training Configuration:**
- Model: YOLOv11n (nano - fastest)
- Epochs: 100 (with early stopping)
- Batch: 16 (adjust based on GPU memory)
- Image size: 640x640

**Training Time Estimates:**
- GPU (RTX 3060): ~1-2 hours for 100 epochs
- GPU (RTX 4090): ~30-45 minutes
- CPU: Very slow, not recommended

**What to Monitor:**
- Loss should decrease steadily
- mAP should increase
- Training typically converges around 50-100 epochs

**Output Location:**
```
runs/train/airsim_yolo_[timestamp]/
  weights/
    best.pt      ← Use this for inference
    last.pt
  results.csv    ← Training metrics
  results.png    ← Training curves
  confusion_matrix.png
```

---

### **STEP 5: Test the Model**

```powershell
python test_yolo.py
```

**Test Options:**
1. **Test on images** - Evaluate on test dataset
2. **Custom folder** - Test on your own images
3. **Real-time AirSim** - Live detection in simulator
4. **Benchmark** - Check inference speed

---

## 📊 Expected Results

### Good Model Performance:
- **mAP50**: > 0.80 (80%)
- **mAP50-95**: > 0.50 (50%)
- **Precision**: > 0.75 (75%)
- **Recall**: > 0.70 (70%)

### If Results Are Poor:
1. **Collect more data** (aim for 1000+ images)
2. **Balance classes** (similar number of each object)
3. **Increase epochs** (try 150-200)
4. **Use larger model** (yolo11s or yolo11m)
5. **Check data quality** (verify bounding boxes)

---

## 🎛️ Training Configuration Options

Edit `train_yolo.py` to adjust:

### Model Size (Speed vs Accuracy):
```python
'model': 'yolo11n.pt'  # Nano - Fastest
'model': 'yolo11s.pt'  # Small - Balanced
'model': 'yolo11m.pt'  # Medium - Better accuracy
'model': 'yolo11l.pt'  # Large - High accuracy
'model': 'yolo11x.pt'  # Extra Large - Best accuracy
```

### Training Duration:
```python
'epochs': 100        # Number of training cycles
'patience': 20       # Early stopping after N epochs without improvement
```

### Memory/Speed:
```python
'batch': 16          # Reduce if GPU runs out of memory (try 8, 4)
'imgsz': 640         # Image size (640, 1280)
'workers': 8         # CPU workers for data loading
```

---

## 🔧 Troubleshooting

### Issue: Out of GPU Memory
**Solution:** Reduce batch size in `train_yolo.py`
```python
'batch': 8  # or 4
```

### Issue: Training Too Slow
**Solution:** 
- Use GPU instead of CPU
- Reduce image size: `'imgsz': 480`
- Use smaller model: `'model': 'yolo11n.pt'`

### Issue: Poor Detection Results
**Solution:**
- Collect more diverse data
- Increase training epochs
- Check if labels are correct
- Use data augmentation (already enabled)

### Issue: Class Imbalance
**Solution:** Collect more data for underrepresented classes

---

## 📈 Monitoring Training

### Real-time:
Watch the terminal output for:
- Loss values (should decrease)
- mAP values (should increase)
- Learning rate changes

### Post-training:
Check these files in `runs/train/[experiment]/`:
- `results.png` - Training curves
- `confusion_matrix.png` - Per-class performance
- `results.csv` - Detailed metrics

---

## 🚁 Using the Trained Model in AirSim

After training, integrate into your AirSim script:

```python
from ultralytics import YOLO
import airsim
import numpy as np
import cv2

# Load trained model
model = YOLO('runs/train/airsim_yolo_[timestamp]/weights/best.pt')

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Real-time detection loop
while True:
    # Get image
    responses = client.simGetImages([
        airsim.ImageRequest("FrontCam", airsim.ImageType.Scene, False, False)
    ])
    
    img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = img.reshape(responses[0].height, responses[0].width, 3)
    
    # Run detection
    results = model(img, conf=0.25)
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: Class {cls}, Confidence: {conf:.2f}")
    
    # Display
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

---

## 📚 Additional Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv11 Paper**: Check Ultralytics GitHub
- **AirSim Docs**: https://microsoft.github.io/AirSim/

---

## 🎯 Quick Command Reference

```powershell
# 1. Collect data
python oddDatasetmanualcontrol.py

# 2. Prepare dataset
python prepare_dataset.py

# 3. Train model
python train_yolo.py

# 4. Test model
python test_yolo.py
```

---

## ✅ Success Checklist

- [ ] Collected at least 500 images
- [ ] Data split into train/val/test
- [ ] Training completed without errors
- [ ] mAP50 > 0.70
- [ ] Model saved in runs/train/
- [ ] Tested on real-time AirSim feed
- [ ] Detection working smoothly

---

## 🎊 You're Ready!

Follow these steps in order, and you'll have a working object detection model for your AirSim environment. Good luck with your training! 🚀
