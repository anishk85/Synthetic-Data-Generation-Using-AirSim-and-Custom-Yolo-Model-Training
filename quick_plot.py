import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
import os

# Get the most recent flight log file
log_files = glob.glob("flightLogs/drone_path_*.csv")
if not log_files:
    print("No flight log files found!")
    exit()

latest_log = max(log_files, key=os.path.getctime)
print(f"Plotting trajectory from: {latest_log}")

# Read the data
df = pd.read_csv(latest_log)

# Extract coordinates
x = df['x'].values
y = df['y'].values
z = df['z'].values

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(x, y, z, 'b-', linewidth=2, label='Flight Path')

# Mark start and end
if len(x) > 0:
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')

# Labels and title
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Drone 3D Flight Trajectory')
ax.legend()
ax.grid(True)

print(f"Flight data points: {len(x)}")
print(f"X range: {x.min():.2f} to {x.max():.2f} m")
print(f"Y range: {y.min():.2f} to {y.max():.2f} m") 
print(f"Z range: {z.min():.2f} to {z.max():.2f} m")

plt.show()