python oddDatasetmanualcontrol.py"""
Quick test script to verify logging and directory setup
Run this before data collection to ensure everything works
"""

import os
from datetime import datetime
import csv

print("\n" + "=" * 60)
print("LOGGING SYSTEM TEST")
print("=" * 60 + "\n")

# Test 1: Create directories
print("[TEST 1] Creating directory structure...")
try:
    session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    BASE_DIR = "flightLogs"
    SESSION_DIR = os.path.join(BASE_DIR, f"test_session_{session_time}")
    IMG_DIR = os.path.join(SESSION_DIR, "images")
    LBL_DIR = os.path.join(SESSION_DIR, "labels")
    
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)
    
    print(f"✓ Created: {BASE_DIR}")
    print(f"✓ Created: {SESSION_DIR}")
    print(f"✓ Created: {IMG_DIR}")
    print(f"✓ Created: {LBL_DIR}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 2: Create log file
print("\n[TEST 2] Creating log file...")
try:
    log_path = os.path.join(SESSION_DIR, "test_flight_log.csv")
    logFile = open(log_path, "w", newline="")
    writer = csv.writer(logFile)
    writer.writerow(["time", "command", "vx", "vy", "vz", "yawRate"])
    logFile.flush()
    print(f"✓ Log file created: {log_path}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 3: Write test data
print("\n[TEST 3] Writing test data...")
try:
    import time
    for i in range(5):
        writer.writerow([time.time(), f"test_cmd_{i}", 1.0, 2.0, 3.0, 0.5])
        logFile.flush()
    print(f"✓ Wrote 5 test entries")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 4: Close and verify
print("\n[TEST 4] Closing log file...")
try:
    logFile.close()
    print(f"✓ Log file closed")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test 5: Verify file exists and is readable
print("\n[TEST 5] Verifying log file...")
try:
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        print(f"✓ File exists: {log_path}")
        print(f"✓ File size: {size} bytes")
        
        # Read it back
        with open(log_path, 'r') as f:
            lines = f.readlines()
            print(f"✓ File contains {len(lines)} lines (including header)")
    else:
        print(f"✗ File does not exist: {log_path}")
        exit(1)
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Cleanup
print("\n[CLEANUP] Removing test files...")
try:
    import shutil
    shutil.rmtree(SESSION_DIR)
    print(f"✓ Removed test directory: {SESSION_DIR}")
except Exception as e:
    print(f"⚠ Warning: Could not remove test directory: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nLogging system is working correctly!")
print("You can now run oddDatasetmanualcontrol.py to collect data.")
print("=" * 60 + "\n")
