#!/usr/bin/env python3
"""
Fill missing error values in error log CSV files.
For each missing error value, uses the previous timestamp's error value (forward fill).
"""

import csv
import os
from pathlib import Path

# Path to the ERROR_LOGS folder
ERROR_LOGS_DIR = Path(__file__).parent / "ERROR_LOGS"

def fill_missing_errors(csv_file_path):
    """
    Read a CSV file, fill missing error values with the previous error value,
    and write the updated data back to the file.
    
    Args:
        csv_file_path: Path to the CSV file to process
    """
    rows = []
    last_error = None
    filled_count = 0
    
    # Read the CSV file
    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        rows.append(header)
        
        for row in reader:
            if len(row) < 3:
                # Skip malformed rows
                rows.append(row)
                continue
            
            timestep = row[0]
            timestamp = row[1]
            error = row[2]
            lap_number = row[3] if len(row) > 3 else ''
            
            # Check if error is empty or None
            if error is None or error.strip() == '':
                # Use the last known error value
                if last_error is not None:
                    error = last_error
                    filled_count += 1
                else:
                    # If no previous error exists, keep it empty (shouldn't happen for first row)
                    error = ''
            else:
                # Update last_error with the current error value
                last_error = error
            
            # Create updated row
            updated_row = [timestep, timestamp, error]
            if lap_number:
                updated_row.append(lap_number)
            rows.append(updated_row)
    
    # Write the updated data back to the file
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    return filled_count

def main():
    """
    Process all CSV files in the ERROR_LOGS directory.
    """
    if not ERROR_LOGS_DIR.exists():
        print(f"ERROR_LOGS directory not found: {ERROR_LOGS_DIR}")
        return
    
    # Get all CSV files in the directory
    csv_files = list(ERROR_LOGS_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {ERROR_LOGS_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    print("-" * 60)
    
    total_filled = 0
    for csv_file in sorted(csv_files):
        print(f"Processing: {csv_file.name}...", end=' ')
        try:
            filled_count = fill_missing_errors(csv_file)
            total_filled += filled_count
            print(f"✓ Filled {filled_count} missing error value(s)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("-" * 60)
    print(f"Total missing values filled: {total_filled}")
    print("Done!")

if __name__ == "__main__":
    main()

