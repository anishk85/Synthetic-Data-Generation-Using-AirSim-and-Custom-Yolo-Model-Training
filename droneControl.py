import airsim
import keyboard
import time 
import csv
import math 
import os
from datetime import datetime

client = airsim.MultirotorClient()
client.confirmConnection()

# first looping setup 
os.makedirs("flightLogs", exist_ok=True)
log_path = f"flightLogs/drone_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
logfile = open(log_path, "w", newline="")
writer = csv.writer(logfile)

writer.writerow([
    "time", "command", "vx", "vy", "vz", "yawRate",
    "x", "y", "z", "roll", "pitch", "yaw",
    "gpsLat", "gpsLon", "gpsAlt"
])

Vel = 2.5
Yaw_Rate = 25
vx = vy = vz = 0.0
YawRate = 0.0
currentCmd = "idle"
flying = False
running = True

while running:
    vx = vy = vz = 0.0
    YawRate = 0.0
    currentCmd = "idle"
    time.sleep(0.05)

    if keyboard.is_pressed("y") and not flying:
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(-10, 2).join()
        flying = True
        currentCmd = "takeoff"
    
    if keyboard.is_pressed("l") and flying:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        flying = False
        currentCmd = "land"

    if not flying:
        time.sleep(0.05)
        continue

    if keyboard.is_pressed("w"):
        vx = Vel
        currentCmd = "forward"
    elif keyboard.is_pressed("s"):
        vx = -Vel
        currentCmd = "backward"
    
    if keyboard.is_pressed("a"):
        vy = -Vel
        currentCmd = "left"
    elif keyboard.is_pressed("d"):
        vy = Vel
        currentCmd = "right"
    
    if keyboard.is_pressed("z"):
        vz = -Vel
        currentCmd = "up"
    elif keyboard.is_pressed("c"):
        vz = Vel
        currentCmd = "down"
    
    if keyboard.is_pressed("q"):
        YawRate = Yaw_Rate  # YawMode expects degrees/sec when is_rate=True
        currentCmd = "yawLeft"

    elif keyboard.is_pressed("e"):
        YawRate = -Yaw_Rate  # YawMode expects degrees/sec when is_rate=True
        currentCmd = "yawRight"

    if keyboard.is_pressed("esc"):
        running = False
        break

    # Handle movement and yaw separately for better control
    if YawRate != 0:
        # Rotate by yaw rate
        client.rotateByYawRateAsync(YawRate, 0.1)
        
    # Handle movement
    if vx != 0 or vy != 0 or vz != 0:
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1)
    elif YawRate == 0:
        # Keep position if no input
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.1)
    
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation
    gps_data = client.getGpsData()
    
    roll, pitch, yaw = airsim.to_eularian_angles(orientation)
    
    # Log data
    current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    writer.writerow([
        current_time, currentCmd, vx, vy, vz, YawRate,
        position.x_val, position.y_val, position.z_val,
        math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
        gps_data.gnss.geo_point.latitude,
        gps_data.gnss.geo_point.longitude,
        gps_data.gnss.geo_point.altitude
    ])
    
    logfile.flush()


logfile.close()
print("Flight log saved to:", log_path)

# Y - Takeoff and enable control
# L - Land and disable control
# W/S - Forward/Backward
# A/D - Left/Right
# Z/C - Up/Down
# Q/E - Rotate left/right
# ESC - Exit program