import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import glob
from datetime import datetime

def get_latest_log_file():
    """Get the most recent flight log file"""
    log_files = glob.glob("flightLogs/drone_path_*.csv")
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def plot_drone_trajectory(log_file=None):
    """Plot 3D trajectory of drone flight"""
    
    if log_file is None:
        log_file = get_latest_log_file()
    
    if log_file is None:
        print("No flight log files found in flightLogs directory!")
        return
    
    print(f"Reading flight data from: {log_file}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Extract position data
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(x, y, z, 'b-', linewidth=2, label='Flight Path')
    
    # Mark start and end points
    if len(x) > 0:
        ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End')
    
    # Color the trajectory based on time (gradient effect)
    if len(x) > 1:
        # Create a color gradient along the path
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        for i in range(len(x)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
                   color=colors[i], linewidth=1.5, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Drone 3D Trajectory')
    
    # Add legend
    ax.legend()
    
    # Make the plot look better
    ax.grid(True)
    
    # Set equal aspect ratio if possible
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), np.abs(z.max()-z.min())]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nFlight Statistics:")
    print(f"Total data points: {len(x)}")
    print(f"Flight duration: {len(x) * 0.05:.1f} seconds (approx)")
    print(f"X range: {x.min():.2f} to {x.max():.2f} m")
    print(f"Y range: {y.min():.2f} to {y.max():.2f} m")
    print(f"Z range: {z.min():.2f} to {z.max():.2f} m")
    if len(x) > 0:
        total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
        print(f"Approximate total distance traveled: {total_distance:.2f} m")

def plot_2d_trajectories(log_file=None):
    """Plot 2D projections of the trajectory"""
    
    if log_file is None:
        log_file = get_latest_log_file()
    
    if log_file is None:
        print("No flight log files found!")
        return
    
    # Read the CSV file
    df = pd.read_csv(log_file)
    
    # Extract position data
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    
    # Create 2D plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # XY plane (top view)
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.scatter(x[0], y[0], color='green', s=50, label='Start')
    ax1.scatter(x[-1], y[-1], color='red', s=50, label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('XY Plane (Top View)')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # XZ plane (side view)
    ax2.plot(x, z, 'b-', linewidth=2)
    ax2.scatter(x[0], z[0], color='green', s=50, label='Start')
    ax2.scatter(x[-1], z[-1], color='red', s=50, label='End')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('XZ Plane (Side View)')
    ax2.grid(True)
    ax2.legend()
    ax2.invert_yaxis()  # Invert Z axis to match AirSim coordinate system
    
    # YZ plane (front view)
    ax3.plot(y, z, 'b-', linewidth=2)
    ax3.scatter(y[0], z[0], color='green', s=50, label='Start')
    ax3.scatter(y[-1], z[-1], color='red', s=50, label='End')
    ax3.set_xlabel('Y Position (m)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('YZ Plane (Front View)')
    ax3.grid(True)
    ax3.legend()
    ax3.invert_yaxis()  # Invert Z axis to match AirSim coordinate system
    
    # Altitude over time
    time_points = np.arange(len(z)) * 0.05  # Assuming 0.05s intervals
    ax4.plot(time_points, z, 'r-', linewidth=2)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Altitude Over Time')
    ax4.grid(True)
    ax4.invert_yaxis()  # Invert Z axis to match AirSim coordinate system
    
    plt.tight_layout()
    plt.show()

def plotGPS():











def yawRate():
    log_files = glob.glob("flightLogs/drone_path_*.csv")
    if not log_files:
        print("No flight log files found in flightLogs directory!")
        return []
    



def list_available_logs():
    """List all available flight log files"""
    log_files = glob.glob("flightLogs/drone_path_*.csv")
    if not log_files:
        print("No flight log files found in flightLogs directory!")
        return []
    
    print("Available flight logs:")
    for i, file in enumerate(log_files):
        file_time = os.path.getctime(file)
        readable_time = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i+1}. {file} (Created: {readable_time})")
    
    return log_files

if __name__ == "__main__":
    print("Drone Trajectory Visualization")
    print("=" * 40)
    
    # List available logs
    log_files = list_available_logs()
    
    if not log_files:
        exit()
    
    print("\nOptions:")
    print("1. Plot latest flight (3D)")
    print("2. Plot latest flight (2D projections)")
    print("3. Choose specific log file")
    print("4. Plot all views of latest flight")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    selected_file = None
    
    if choice == "3":
        try:
            file_index = int(input(f"Choose file (1-{len(log_files)}): ")) - 1
            if 0 <= file_index < len(log_files):
                selected_file = log_files[file_index]
            else:
                print("Invalid selection, using latest file")
        except:
            print("Invalid input, using latest file")
    
    if choice == "1":
        plot_drone_trajectory(selected_file)
    elif choice == "2":
        plot_2d_trajectories(selected_file)
    elif choice == "3":
        plot_drone_trajectory(selected_file)
    elif choice == "4":
        print("Showing 3D trajectory...")
        plot_drone_trajectory(selected_file)
        print("Showing 2D projections...")
        plot_2d_trajectories(selected_file)
    else:
        print("Invalid choice, showing 3D trajectory of latest flight")
        plot_drone_trajectory(selected_file)