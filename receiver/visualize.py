#!/usr/bin/env python3
"""
Create visualizations for error log data.
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
ERROR_LOGS_DIR = Path(__file__).parent / "ERROR_LOGS"
VISUALIZATIONS_DIR = Path(__file__).parent / "visualizations"

def ensure_visualizations_dir():
    """Create visualizations directory if it doesn't exist."""
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)

def ensure_folder(folder_name):
    """Create a folder in visualizations directory."""
    folder_path = VISUALIZATIONS_DIR / folder_name
    folder_path.mkdir(exist_ok=True)
    return folder_path

def extract_speed_from_filename(filename):
    """
    Extract speed value from filename (e.g., 'error_log_0.1.csv' -> 0.1).
    
    Args:
        filename: Name of the CSV file
        
    Returns:
        float: Speed value, or None if not found
    """
    try:
        # Remove extension and split by underscore
        name_without_ext = filename.replace('.csv', '')
        parts = name_without_ext.split('_')
        # Find the part that looks like a number
        for part in reversed(parts):
            try:
                return float(part)
            except ValueError:
                continue
        return None
    except Exception:
        return None

def read_csv_data(csv_file_path):
    """
    Read CSV file and return timestamps, errors, and lap numbers.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        tuple: (timestamps, errors, lap_numbers) as lists
    """
    timestamps = []
    errors = []
    lap_numbers = []
    
    try:
        with open(csv_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 3:
                    try:
                        timestamp = float(row[1])
                        error_str = row[2].strip()
                        lap_num = int(row[3]) if len(row) > 3 and row[3].strip() else 1
                        
                        if error_str and error_str != '':
                            error = float(error_str)
                            timestamps.append(timestamp)
                            errors.append(error)
                            lap_numbers.append(lap_num)
                    except (ValueError, IndexError):
                        continue
        
        return timestamps, errors, lap_numbers
    except Exception as e:
        print(f"Error reading {csv_file_path.name}: {e}")
        return [], [], []

def get_all_csv_files():
    """Get all CSV files with their speeds."""
    csv_files = list(ERROR_LOGS_DIR.glob("*.csv"))
    file_data = []
    
    for csv_file in sorted(csv_files):
        speed = extract_speed_from_filename(csv_file.name)
        if speed is not None:
            file_data.append((speed, csv_file))
    
    return sorted(file_data, key=lambda x: x[0])

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_speed_avgError():
    """1. Average Error vs Speed"""
    folder = ensure_folder("speed_avgError")
    file_data = get_all_csv_files()
    
    speeds = []
    avg_errors = []
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if errors:
            avg_error = np.mean(errors)
            speeds.append(speed)
            avg_errors.append(avg_error)
    
    if not speeds:
        print("No data for speed_avgError plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, avg_errors, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Average Error', fontsize=12, fontweight='bold')
    plt.title('Average Error vs Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(speeds)
    
    for speed, error in zip(speeds, avg_errors):
        plt.annotate(f'{error:.2f}', (speed, error), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "speed_avgError.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_error_overTime():
    """2. Error over Time (separate plot for each speed) - with smoothing"""
    folder = ensure_folder("error_overTime")
    file_data = get_all_csv_files()
    
    # Window size for moving average (adjust based on data density)
    # Using a window that represents ~2-3 seconds of data
    window_size = 50
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if not errors or len(errors) < window_size:
            continue
        
        # Convert to numpy arrays for easier processing
        timestamps = np.array(timestamps)
        errors = np.array(errors)
        
        # Calculate moving average (rolling mean) using convolution
        # Create a uniform window for averaging
        window = np.ones(window_size) / window_size
        smoothed_errors = np.convolve(errors, window, mode='valid')
        # Align timestamps with smoothed data (use end of window)
        smoothed_timestamps = timestamps[window_size-1:]
        
        if len(smoothed_errors) == 0:
            continue
        
        plt.figure(figsize=(12, 6))
        
        # Plot raw data (very light, for reference)
        plt.plot(timestamps, errors, linewidth=0.5, alpha=0.2, color='gray', label='Raw Data')
        
        # Plot smoothed data (prominent)
        plt.plot(smoothed_timestamps, smoothed_errors, linewidth=2, alpha=0.8, 
                color='blue', label=f'Smoothed (Window={window_size})')
        
        plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Error', fontsize=12, fontweight='bold')
        plt.title(f'Error over Time (Speed = {speed}) - Smoothed', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_path = folder / f"error_overTime_{speed}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

def plot_errorCumulative_overTime():
    """Cumulative Error over Time (error buildup) - separate plot for each speed"""
    folder = ensure_folder("errorCumulative_overTime")
    file_data = get_all_csv_files()
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if not errors:
            continue
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        errors = np.array(errors)
        
        # Calculate cumulative sum of errors
        cumulative_errors = np.cumsum(errors)
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, cumulative_errors, linewidth=2, alpha=0.8, color='darkred')
        plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Error', fontsize=12, fontweight='bold')
        plt.title(f'Cumulative Error Buildup over Time (Speed = {speed})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = folder / f"errorCumulative_overTime_{speed}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

def plot_speed_lap1Time():
    """Time to Complete Lap 1 vs Speed"""
    folder = ensure_folder("speed_lap1Time")
    file_data = get_all_csv_files()
    
    speeds = []
    lap1_times = []
    
    for speed, csv_file in file_data:
        timestamps, errors, lap_numbers = read_csv_data(csv_file)
        if not lap_numbers:
            continue
        
        # Find the first occurrence of lap_number = 2
        # This indicates when lap 1 was completed
        lap1_completion_time = None
        for i, lap_num in enumerate(lap_numbers):
            if lap_num == 2:
                lap1_completion_time = timestamps[i]
                break
        
        if lap1_completion_time is not None:
            speeds.append(speed)
            lap1_times.append(lap1_completion_time)
        else:
            # Drone crashed before completing lap 1
            # Optionally include with a marker or skip
            speeds.append(speed)
            lap1_times.append(np.nan)  # Mark as incomplete
    
    if not speeds:
        print("No data for speed_lap1Time plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot completed laps
    valid_mask = ~np.isnan(lap1_times)
    if np.any(valid_mask):
        plt.plot(np.array(speeds)[valid_mask], np.array(lap1_times)[valid_mask], 
                marker='o', linestyle='-', linewidth=2, markersize=8, color='green', label='Completed')
    
    # Plot incomplete laps (crashed before lap 1)
    incomplete_mask = np.isnan(lap1_times)
    if np.any(incomplete_mask):
        incomplete_speeds = np.array(speeds)[incomplete_mask]
        if np.any(valid_mask):
            y_pos = max(np.array(lap1_times)[valid_mask]) * 1.1
        else:
            y_pos = 300
        incomplete_y = [y_pos] * len(incomplete_speeds)
        plt.scatter(incomplete_speeds, incomplete_y,
                   marker='x', s=200, color='red', linewidths=3, label='Crashed before Lap 1', zorder=5)
    
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Time to Complete Lap 1 (seconds)', fontsize=12, fontweight='bold')
    plt.title('Time to Complete First Lap vs Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(speeds)
    plt.legend()
    
    # Add value labels on points
    for speed, time in zip(speeds, lap1_times):
        if not np.isnan(time):
            plt.annotate(f'{time:.1f}s', (speed, time), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "speed_lap1Time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_error_distribution():
    """3. Error Distribution (box plot for each speed)"""
    folder = ensure_folder("error_distribution")
    file_data = get_all_csv_files()
    
    error_data = []
    speed_labels = []
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if errors:
            error_data.append(errors)
            speed_labels.append(f'{speed}')
    
    if not error_data:
        print("No data for error_distribution plot")
        return
    
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(error_data, tick_labels=speed_labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Error', fontsize=12, fontweight='bold')
    plt.title('Error Distribution by Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = folder / "error_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_speed_maxError():
    """4. Max Error vs Speed"""
    folder = ensure_folder("speed_maxError")
    file_data = get_all_csv_files()
    
    speeds = []
    max_errors = []
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if errors:
            max_error = np.max(errors)
            speeds.append(speed)
            max_errors.append(max_error)
    
    if not speeds:
        print("No data for speed_maxError plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, max_errors, marker='o', linestyle='-', linewidth=2, markersize=8, color='red')
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Max Error', fontsize=12, fontweight='bold')
    plt.title('Maximum Error vs Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(speeds)
    
    for speed, error in zip(speeds, max_errors):
        plt.annotate(f'{error:.2f}', (speed, error), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "speed_maxError.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_speed_errorStd():
    """5. Error Standard Deviation vs Speed"""
    folder = ensure_folder("speed_errorStd")
    file_data = get_all_csv_files()
    
    speeds = []
    error_stds = []
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if errors:
            error_std = np.std(errors)
            speeds.append(speed)
            error_stds.append(error_std)
    
    if not speeds:
        print("No data for speed_errorStd plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, error_stds, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Error Standard Deviation', fontsize=12, fontweight='bold')
    plt.title('Error Variability vs Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(speeds)
    
    for speed, std in zip(speeds, error_stds):
        plt.annotate(f'{std:.2f}', (speed, std), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "speed_errorStd.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_speed_lapCompletion():
    """6. Lap Completion vs Speed"""
    folder = ensure_folder("speed_lapCompletion")
    file_data = get_all_csv_files()
    
    speeds = []
    lap_completions = []
    
    for speed, csv_file in file_data:
        timestamps, errors, lap_numbers = read_csv_data(csv_file)
        if lap_numbers:
            max_lap = max(lap_numbers)
            speeds.append(speed)
            lap_completions.append(max_lap)
    
    if not speeds:
        print("No data for speed_lapCompletion plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(speeds, lap_completions, width=0.05, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Laps Completed', fontsize=12, fontweight='bold')
    plt.title('Lap Completion vs Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(speeds)
    
    for speed, laps in zip(speeds, lap_completions):
        plt.annotate(f'{laps}', (speed, laps), 
                    textcoords="offset points", xytext=(0,5), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = folder / "speed_lapCompletion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_speed_crashTime():
    """7. Crash Time vs Speed"""
    folder = ensure_folder("speed_crashTime")
    file_data = get_all_csv_files()
    
    speeds = []
    crash_times = []
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if timestamps:
            # Last timestamp is when it crashed (or reached 10 minutes)
            crash_time = max(timestamps)
            speeds.append(speed)
            crash_times.append(crash_time)
    
    if not speeds:
        print("No data for speed_crashTime plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, crash_times, marker='o', linestyle='-', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Time Until Crash/End (seconds)', fontsize=12, fontweight='bold')
    plt.title('Crash Time vs Speed', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(speeds)
    plt.axhline(y=600, color='r', linestyle='--', alpha=0.5, label='10 minute limit')
    plt.legend()
    
    for speed, time in zip(speeds, crash_times):
        plt.annotate(f'{time:.1f}s', (speed, time), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "speed_crashTime.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_errorVolatility_overTime():
    """8. Error Volatility over Time (rolling std for each speed)"""
    folder = ensure_folder("errorVolatility_overTime")
    file_data = get_all_csv_files()
    
    window_size = 50  # Rolling window size
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if len(errors) < window_size:
            continue
        
        # Calculate rolling standard deviation
        rolling_std = []
        rolling_times = []
        
        for i in range(window_size, len(errors)):
            window_errors = errors[i-window_size:i]
            rolling_std.append(np.std(window_errors))
            rolling_times.append(timestamps[i])
        
        if not rolling_std:
            continue
        
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_times, rolling_std, linewidth=1.5, alpha=0.7, color='darkred')
        plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Rolling Std Dev (Error)', fontsize=12, fontweight='bold')
        plt.title(f'Error Volatility over Time (Speed = {speed}, Window = {window_size})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = folder / f"errorVolatility_overTime_{speed}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

def plot_lapNumber_avgError():
    """9. Average Error by Lap Number"""
    folder = ensure_folder("lapNumber_avgError")
    file_data = get_all_csv_files()
    
    # Aggregate errors by lap number across all speeds
    lap_errors = defaultdict(list)
    
    for speed, csv_file in file_data:
        timestamps, errors, lap_numbers = read_csv_data(csv_file)
        for error, lap_num in zip(errors, lap_numbers):
            lap_errors[lap_num].append(error)
    
    if not lap_errors:
        print("No data for lapNumber_avgError plot")
        return
    
    # Calculate average error per lap
    lap_numbers = sorted(lap_errors.keys())
    avg_errors = [np.mean(lap_errors[lap]) for lap in lap_numbers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lap_numbers, avg_errors, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Lap Number', fontsize=12, fontweight='bold')
    plt.ylabel('Average Error', fontsize=12, fontweight='bold')
    plt.title('Average Error by Lap Number (All Speeds)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(lap_numbers)
    
    for lap, error in zip(lap_numbers, avg_errors):
        plt.annotate(f'{error:.2f}', (lap, error), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "lapNumber_avgError.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_error_heatmap():
    """10. Error Heatmap (Speed × Time)"""
    folder = ensure_folder("error_heatmap")
    file_data = get_all_csv_files()
    
    # Create time bins (every 60 seconds)
    time_bins = np.arange(0, 601, 60)  # 0 to 600 seconds in 60s bins
    speeds = []
    heatmap_data = []
    
    for speed, csv_file in file_data:
        timestamps, errors, _ = read_csv_data(csv_file)
        if not errors:
            continue
        
        speeds.append(speed)
        bin_errors = []
        
        for i in range(len(time_bins) - 1):
            # Find errors in this time bin
            bin_mask = (np.array(timestamps) >= time_bins[i]) & (np.array(timestamps) < time_bins[i+1])
            bin_error_values = np.array(errors)[bin_mask]
            
            if len(bin_error_values) > 0:
                bin_errors.append(np.mean(bin_error_values))
            else:
                bin_errors.append(np.nan)  # No data in this bin
        
        heatmap_data.append(bin_errors)
    
    if not heatmap_data:
        print("No data for error_heatmap plot")
        return
    
    # Create heatmap
    heatmap_array = np.array(heatmap_data)
    
    plt.figure(figsize=(14, 8))
    im = plt.imshow(heatmap_array, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, label='Average Error')
    
    # Set ticks and labels
    plt.yticks(range(len(speeds)), [f'{s}' for s in speeds])
    plt.xticks(range(len(time_bins) - 1), 
              [f'{int(time_bins[i])}-{int(time_bins[i+1])}s' for i in range(len(time_bins) - 1)])
    
    plt.xlabel('Time Range (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Speed', fontsize=12, fontweight='bold')
    plt.title('Error Heatmap: Speed × Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = folder / "error_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_speed_performanceScore():
    """11. Performance Score vs Speed (composite metric)"""
    folder = ensure_folder("speed_performanceScore")
    file_data = get_all_csv_files()
    
    speeds = []
    performance_scores = []
    
    for speed, csv_file in file_data:
        timestamps, errors, lap_numbers = read_csv_data(csv_file)
        if not errors:
            continue
        
        # Calculate components
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        crash_time = max(timestamps) if timestamps else 0
        max_lap = max(lap_numbers) if lap_numbers else 0
        
        # Normalize and combine (lower error = better, higher time/laps = better)
        # Invert errors so higher is better
        error_score = 1 / (1 + avg_error)  # Normalized inverse error
        stability_score = 1 / (1 + np.std(errors))  # Lower std = better
        time_score = min(crash_time / 600, 1.0)  # Normalize to 0-1 (600s = 10 min)
        lap_score = max_lap / 10.0  # Normalize assuming max 10 laps
        
        # Composite score (weighted average)
        performance_score = (error_score * 0.3 + stability_score * 0.2 + 
                           time_score * 0.3 + lap_score * 0.2) * 100
        
        speeds.append(speed)
        performance_scores.append(performance_score)
    
    if not speeds:
        print("No data for speed_performanceScore plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(speeds, performance_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='darkblue')
    plt.xlabel('Speed', fontsize=12, fontweight='bold')
    plt.ylabel('Performance Score', fontsize=12, fontweight='bold')
    plt.title('Performance Score vs Speed\n(Lower Error + Higher Stability + Longer Time + More Laps)', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(speeds)
    
    for speed, score in zip(speeds, performance_scores):
        plt.annotate(f'{score:.1f}', (speed, score), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = folder / "speed_performanceScore.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to generate all visualizations."""
    ensure_visualizations_dir()
    
    print("Generating visualizations...")
    print("=" * 60)
    
    visualizations = [
        ("1. Average Error vs Speed", plot_speed_avgError),
        ("2. Error over Time", plot_error_overTime),
        ("2b. Cumulative Error Buildup over Time", plot_errorCumulative_overTime),
        ("3. Error Distribution", plot_error_distribution),
        ("4. Max Error vs Speed", plot_speed_maxError),
        ("5. Error Standard Deviation vs Speed", plot_speed_errorStd),
        ("6. Lap Completion vs Speed", plot_speed_lapCompletion),
        ("6b. Time to Complete Lap 1 vs Speed", plot_speed_lap1Time),
        ("7. Crash Time vs Speed", plot_speed_crashTime),
        ("8. Error Volatility over Time", plot_errorVolatility_overTime),
        ("9. Average Error by Lap Number", plot_lapNumber_avgError),
        ("10. Error Heatmap", plot_error_heatmap),
        ("11. Performance Score vs Speed", plot_speed_performanceScore),
    ]
    
    for name, func in visualizations:
        print(f"\n{name}...")
        try:
            func()
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Visualization generation complete!")

if __name__ == "__main__":
    main()
