"""
YOLOv11 Training Script for AirSim Object Detection
Trains a custom YOLOv11 model on collected AirSim data
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime

# Training Configuration
CONFIG = {
    # Model
    'model': 'yolo11n.pt',  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x (n=nano, fastest)
    
    # Dataset
    'data': 'dataset_config.yaml',
    
    # Training Parameters
    'epochs': 50,           # Number of training epochs
    'imgsz': 640,           # Image size (640 is standard)
    'batch': 8,            # Batch size (adjust based on GPU memory)
    'patience': 15,         # Early stopping patience
    
    # Optimization
    'optimizer': 'auto',    # Adam, SGD, auto
    'lr0': 0.01,           # Initial learning rate
    'lrf': 0.01,           # Final learning rate factor
    'momentum': 0.937,      # SGD momentum
    'weight_decay': 0.0005, # Optimizer weight decay
    
    # Augmentation
    'hsv_h': 0.015,        # HSV-Hue augmentation
    'hsv_s': 0.7,          # HSV-Saturation augmentation
    'hsv_v': 0.4,          # HSV-Value augmentation
    'degrees': 10.0,        # Rotation (+/- deg)
    'translate': 0.1,       # Translation (+/- fraction)
    'scale': 0.5,          # Image scale (+/- gain)
    'flipud': 0.0,         # Flip up-down probability
    'fliplr': 0.5,         # Flip left-right probability
    'mosaic': 1.0,         # Mosaic augmentation probability
    
    # Training Settings
    'device': 0,       # GPU device (0 for first GPU, 'cpu' for CPU)
    'workers': 4,          # Number of data loading workers
    'project': 'runs/train', # Project directory
    'name': f'airsim_yolo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    'exist_ok': False,     # Overwrite existing project/name
    'pretrained': True,    # Use pretrained weights
    'verbose': True,       # Verbose output
    'save': True,          # Save checkpoints
    'save_period': 10,     # Save checkpoint every x epochs
    'cache': False,        # Cache images for faster training (requires RAM)
    'rect': False,         # Rectangular training
    'cos_lr': True,        # Cosine learning rate scheduler
    'close_mosaic': 10,    # Disable mosaic augmentation for last N epochs
    'resume': False,       # Resume from last checkpoint
    'amp': True,           # Automatic Mixed Precision training
    'fraction': 1.0,       # Dataset fraction to use
    'profile': False,      # Profile ONNX and TensorRT speeds
    'freeze': None,        # Freeze layers (e.g., [0, 1, 2])
}

def check_environment():
    """Check if the environment is ready for training"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, training will be slow on CPU!")
    
    # Check dataset
    if not os.path.exists('dataset'):
        print("\nERROR: Dataset folder not found!")
        print("Please run prepare_dataset.py first to prepare your data.")
        return False
    
    if not os.path.exists(CONFIG['data']):
        print(f"\nERROR: Config file '{CONFIG['data']}' not found!")
        return False
    
    print("\n✓ Environment check passed!")
    return True

def train_model():
    """Train the YOLOv11 model"""
    print("\n" + "=" * 60)
    print("TRAINING YOLOV11 ON AIRSIM DATA")
    print("=" * 60 + "\n")
    
    # Load model
    print(f"Loading model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    
    print(f"\nModel architecture: {CONFIG['model']}")
    print(f"Dataset: {CONFIG['data']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch']}")
    print(f"Image size: {CONFIG['imgsz']}")
    print(f"Device: {CONFIG['device']}")
    print("\nStarting training...\n")
    
    # Train
    results = model.train(
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        patience=CONFIG['patience'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        flipud=CONFIG['flipud'],
        fliplr=CONFIG['fliplr'],
        mosaic=CONFIG['mosaic'],
        device=CONFIG['device'],
        workers=CONFIG['workers'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=CONFIG['exist_ok'],
        pretrained=CONFIG['pretrained'],
        verbose=CONFIG['verbose'],
        save=CONFIG['save'],
        save_period=CONFIG['save_period'],
        cache=CONFIG['cache'],
        rect=CONFIG['rect'],
        cos_lr=CONFIG['cos_lr'],
        close_mosaic=CONFIG['close_mosaic'],
        resume=CONFIG['resume'],
        amp=CONFIG['amp'],
        fraction=CONFIG['fraction'],
        profile=CONFIG['profile'],
        freeze=CONFIG['freeze'],
    )
    
    return model, results

def evaluate_model(model):
    """Evaluate the trained model"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60 + "\n")
    
    # Validate on test set
    print("Validating on test set...")
    metrics = model.val(split='test')
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 60)

def save_training_info(model):
    """Save training information"""
    model_path = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved to: {model_path}")
    print(f"Training results folder: {CONFIG['project']}/{CONFIG['name']}")
    print("\nFiles saved:")
    print("  - best.pt: Best model weights")
    print("  - last.pt: Last epoch weights")
    print("  - results.csv: Training metrics")
    print("  - confusion_matrix.png: Confusion matrix")
    print("  - results.png: Training curves")
    print("\nNext steps:")
    print("  1. Review training results in the runs/train folder")
    print("  2. Run test_yolo.py to test the model on new data")
    print("  3. Use the model for real-time detection in AirSim")

def main():
    # Check environment
    if not check_environment():
        return
    
    # Train model
    model, results = train_model()
    
    # Evaluate model
    evaluate_model(model)
    
    # Save info
    save_training_info(model)

if __name__ == "__main__":
    main()
