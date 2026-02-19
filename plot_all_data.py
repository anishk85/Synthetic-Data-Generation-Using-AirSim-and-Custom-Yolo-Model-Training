import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from datetime import datetime

def get_latest_log_file():
    """Get the most recent flight log file"""
    log_files = glob.glob("flightLogs/drone_path_*.csv")
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def plot_all_flight_data(log_file=None):
    """Plot 14 individual graphs from flight data - one at a time"""
    
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
    
    # Apply drone sign convention: Down is positive for Z
    # In AirSim, negative Z is up, but for drone convention we flip it
    df['z_drone'] = -df['z']  # Now positive is down (altitude loss)
    df['vz_drone'] = -df['vz']  # Positive velocity is downward
    
    # Create time array (in seconds)
    time_seconds = np.arange(len(df)) * 0.05  # Assuming 0.05s intervals
    
    print(f"\n{'='*60}")
    print(f"Flight Statistics:")
    print(f"{'='*60}")
    print(f"Total data points: {len(df)}")
    print(f"Flight duration: {time_seconds[-1]:.1f} seconds")
    print(f"Data columns: {list(df.columns)}")
    print(f"{'='*60}\n")
    print("Close each graph to see the next one...")
    print(f"NOTE: Z-axis uses drone convention (positive = down)")
    print(f"{'='*60}\n")
    
    # ============= GRAPH 1: X Position =============
    print("Showing Graph 1/14: X Position Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['x'], 'r-', linewidth=2.5, label='X Position')
    plt.fill_between(time_seconds, df['x'], alpha=0.3, color='red')
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('X Position (meters)', fontsize=12, fontweight='bold')
    plt.title('Graph 1/14: X Position Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 2: Y Position =============
    print("Showing Graph 2/14: Y Position Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['y'], 'g-', linewidth=2.5, label='Y Position')
    plt.fill_between(time_seconds, df['y'], alpha=0.3, color='green')
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Y Position (meters)', fontsize=12, fontweight='bold')
    plt.title('Graph 2/14: Y Position Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 3: Z Position (Drone Convention) =============
    print("Showing Graph 3/14: Z Position Over Time (Drone Convention: +Down)")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['z_drone'], 'b-', linewidth=2.5, label='Z Position (+ = Down)')
    plt.fill_between(time_seconds, df['z_drone'], alpha=0.3, color='blue')
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Z Position - Drone Convention (meters, + = Down)', fontsize=12, fontweight='bold')
    plt.title('Graph 3/14: Z Position Over Time (Positive = Down)', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 4: Vx (Velocity X) =============
    print("Showing Graph 4/14: Velocity X Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['vx'], color='#FF6B6B', linewidth=2.5, label='Vx (Velocity X)')
    plt.fill_between(time_seconds, df['vx'], alpha=0.3, color='#FF6B6B')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity X (m/s)', fontsize=12, fontweight='bold')
    plt.title('Graph 4/14: Velocity X Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 5: Vy (Velocity Y) =============
    print("Showing Graph 5/14: Velocity Y Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['vy'], color='#4ECDC4', linewidth=2.5, label='Vy (Velocity Y)')
    plt.fill_between(time_seconds, df['vy'], alpha=0.3, color='#4ECDC4')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity Y (m/s)', fontsize=12, fontweight='bold')
    plt.title('Graph 5/14: Velocity Y Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 6: Vz (Velocity Z) =============
    print("Showing Graph 6/14: Velocity Z Over Time (Drone Convention: +Down)")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['vz_drone'], color='#95E1D3', linewidth=2.5, label='Vz (+ = Downward)')
    plt.fill_between(time_seconds, df['vz_drone'], alpha=0.3, color='#95E1D3')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity Z (m/s, + = Downward)', fontsize=12, fontweight='bold')
    plt.title('Graph 6/14: Velocity Z Over Time (Positive = Downward)', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 7: Yaw Rate =============
    print("Showing Graph 7/14: Yaw Rate Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['yawRate'], color='#9B59B6', linewidth=2.5, label='Yaw Rate')
    plt.fill_between(time_seconds, df['yawRate'], alpha=0.3, color='#9B59B6')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Yaw Rate (rad/s)', fontsize=12, fontweight='bold')
    plt.title('Graph 7/14: Yaw Rate Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 8: Roll =============
    print("Showing Graph 8/14: Roll Angle Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['roll'], color='#E74C3C', linewidth=2.5, label='Roll Angle')
    plt.fill_between(time_seconds, df['roll'], alpha=0.3, color='#E74C3C')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Roll Angle (degrees)', fontsize=12, fontweight='bold')
    plt.title('Graph 8/14: Roll Angle Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 9: Pitch =============
    print("Showing Graph 9/14: Pitch Angle Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['pitch'], color='#27AE60', linewidth=2.5, label='Pitch Angle')
    plt.fill_between(time_seconds, df['pitch'], alpha=0.3, color='#27AE60')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Pitch Angle (degrees)', fontsize=12, fontweight='bold')
    plt.title('Graph 9/14: Pitch Angle Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 10: Yaw =============
    print("Showing Graph 10/14: Yaw Angle Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['yaw'], color='#2980B9', linewidth=2.5, label='Yaw Angle')
    plt.fill_between(time_seconds, df['yaw'], alpha=0.3, color='#2980B9')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Yaw Angle (degrees)', fontsize=12, fontweight='bold')
    plt.title('Graph 10/14: Yaw Angle Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 11: GPS Latitude =============
    print("Showing Graph 11/14: GPS Latitude Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['gpsLat'], color='#F39C12', linewidth=2.5, label='GPS Latitude')
    plt.fill_between(time_seconds, df['gpsLat'], alpha=0.3, color='#F39C12')
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Latitude (degrees)', fontsize=12, fontweight='bold')
    plt.title('Graph 11/14: GPS Latitude Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 12: GPS Longitude =============
    print("Showing Graph 12/14: GPS Longitude Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['gpsLon'], color='#16A085', linewidth=2.5, label='GPS Longitude')
    plt.fill_between(time_seconds, df['gpsLon'], alpha=0.3, color='#16A085')
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Longitude (degrees)', fontsize=12, fontweight='bold')
    plt.title('Graph 12/14: GPS Longitude Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 13: GPS Altitude =============
    print("Showing Graph 13/14: GPS Altitude Over Time")
    plt.figure(figsize=(12, 7))
    plt.plot(time_seconds, df['gpsAlt'], color='#8E44AD', linewidth=2.5, label='GPS Altitude')
    plt.fill_between(time_seconds, df['gpsAlt'], alpha=0.3, color='#8E44AD')
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('GPS Altitude (meters)', fontsize=12, fontweight='bold')
    plt.title('Graph 13/14: GPS Altitude Over Time', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # ============= GRAPH 14: XY Trajectory (Top View) =============
    print("Showing Graph 14/14: Flight Trajectory (Top View)")
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with color gradient based on time
    scatter = plt.scatter(df['x'], df['y'], c=time_seconds, cmap='viridis', 
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot the trajectory path
    plt.plot(df['x'], df['y'], 'b-', alpha=0.3, linewidth=1.5, label='Flight Path')
    
    # Mark start point
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=300, 
               marker='o', edgecolors='black', linewidth=2, label='Start', zorder=5)
    
    # Mark end point
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', s=300, 
               marker='X', edgecolors='black', linewidth=2, label='End', zorder=5)
    
    # Add direction arrows at intervals
    arrow_interval = len(df) // 10  # Show ~10 arrows
    for i in range(0, len(df) - arrow_interval, arrow_interval):
        dx = df['x'].iloc[i + arrow_interval] - df['x'].iloc[i]
        dy = df['y'].iloc[i + arrow_interval] - df['y'].iloc[i]
        if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only show if movement is significant
            plt.arrow(df['x'].iloc[i], df['y'].iloc[i], dx, dy, 
                     head_width=0.5, head_length=0.3, fc='blue', ec='blue', 
                     alpha=0.4, length_includes_head=True)
    
    plt.xlabel('X Position (meters)', fontsize=12, fontweight='bold')
    plt.ylabel('Y Position (meters)', fontsize=12, fontweight='bold')
    plt.title('Graph 14/14: Flight Trajectory - Top View (XY Plane)', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.axis('equal')  # Keep aspect ratio equal
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter, label='Time (seconds)', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print statistics
    speed = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    dx = np.diff(df['x'], prepend=df['x'].iloc[0])
    dy = np.diff(df['y'], prepend=df['y'].iloc[0])
    dz = np.diff(df['z'], prepend=df['z'].iloc[0])
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    total_distance = np.sum(distances)
    
    print(f"\n{'='*60}")
    print("Flight Summary Statistics:")
    print(f"{'='*60}")
    print(f"Max speed: {speed.max():.2f} m/s")
    print(f"Average speed: {speed.mean():.2f} m/s")
    print(f"Total distance traveled: {total_distance:.2f} m")
    print(f"Max altitude change (drone convention): {df['z_drone'].max() - df['z_drone'].min():.2f} m")
    print(f"Initial Z position (drone): {df['z_drone'].iloc[0]:.2f} m")
    print(f"Final Z position (drone): {df['z_drone'].iloc[-1]:.2f} m")
    print(f"{'='*60}")

def list_available_logs():
    """List all available flight log files"""
    log_files = glob.glob("flightLogs/drone_path_*.csv")
    if not log_files:
        print("No flight log files found in flightLogs directory!")
        return []
    
    print("\nAvailable flight logs:")
    print(f"{'='*60}")
    for i, file in enumerate(log_files):
        file_time = os.path.getctime(file)
        readable_time = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i+1}. {file} (Created: {readable_time})")
    print(f"{'='*60}")
    
    return log_files

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 14 INDIVIDUAL DRONE FLIGHT DATA GRAPHS")
    print("=" * 60)
    
    # List available logs
    log_files = list_available_logs()
    
    if not log_files:
        exit()
    
    print("\nOptions:")
    print("1. Plot 14 graphs from latest flight (includes trajectory)")
    print("2. Choose specific log file")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    selected_file = None
    
    if choice == "2":
        try:
            file_index = int(input(f"Choose file (1-{len(log_files)}): ")) - 1
            if 0 <= file_index < len(log_files):
                selected_file = log_files[file_index]
            else:
                print("Invalid selection, using latest file")
        except:
            print("Invalid input, using latest file")
    
    print("\n" + "=" * 60)
    print("Starting to display 14 individual graphs...")
    print("CLOSE EACH GRAPH WINDOW TO SEE THE NEXT ONE")
    print("=" * 60 + "\n")
    
    plot_all_flight_data(selected_file)
    
    print("\n" + "=" * 60)
    print("All 14 graphs displayed successfully!")
    print("=" * 60)
