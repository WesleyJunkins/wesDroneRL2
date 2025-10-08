# Receiver for the Webots drone controller
# This receiver simulates a discrete-command interface
#   Meaning it receives an array, processes it, and sends a single set of commands (increase speed, turn left, etc...) as opposed to continuously adjusting values based on the current state of the array.
# By Wesley Junkins

import socket
import time
import numpy as np
import json

# Function to take the array, and make sure the drone is still following the curve
def process_array(array_2d, client_socket):
    """
    Check if the curve is properly centered in the array and send appropriate commands.
    
    Args:
        array_2d: 2D numpy array (64x64) containing the curve data
        client_socket: Socket connection to send commands to the controller
        
    Returns:
        bool: True if curve is properly centered, False otherwise
    """
    # Initialize instruction list
    instructions = []
    
    # DEBUG: Print array statistics
    print(f"=== ARRAY ANALYSIS DEBUG ===")
    print(f"Array shape: {array_2d.shape}")
    print(f"Min value: {array_2d.min()}, Max value: {array_2d.max()}")
    print(f"Mean value: {np.mean(array_2d):.3f}")
    print(f"Non-zero pixels: {np.count_nonzero(array_2d)} / {array_2d.size}")
    
    # Check columns 0-19 (should be all 0's)
    left_side_issue = False
    for col in range(20):  # 0 to 19 inclusive
        if not np.all(array_2d[:, col] == 0):
            print(f"Check 1 failed: Column {col} contains non-zero values")
            left_side_issue = True
            break
    
    # Check columns 43-63 (should be all 0's)
    right_side_issue = False
    for col in range(43, 64):  # 43 to 63 inclusive
        if not np.all(array_2d[:, col] == 0):
            print(f"Check 1 failed: Column {col} contains non-zero values")
            right_side_issue = True
            break
    
    # Add only one movement command based on which side has issues
    if left_side_issue and not right_side_issue:
        instructions.append("backward")  # Move backward to get back on track
    elif right_side_issue and not left_side_issue:
        instructions.append("forward")  # Move forward to get back on track
    elif left_side_issue and right_side_issue:
        # Both sides have issues - need to center first, don't add movement command
        print("Both sides have issues - skipping forward/backward movement")
    
    if not instructions:
        print("Passed Check 1")
    
    # Check 2: Analyze columns 20-24 and 38-43 for turn direction
    yaw_correction_needed = False
    yaw_direction = None
    
    # Check columns 20-24 (left side)
    for col in range(20, 25):  # 20 to 25 inclusive
        # Check upper half (rows 0-31)
        upper_half = array_2d[0:32, col]
        if not np.all(upper_half == 0):
            print("Needs to turn left")
            yaw_correction_needed = True
            yaw_direction = "yaw_decrease"
            break
        
        # Check lower half (rows 32-63)
        lower_half = array_2d[32:64, col]
        if not np.all(lower_half == 0):
            print("Needs to turn right")
            yaw_correction_needed = True
            yaw_direction = "yaw_increase"
            break
    
    # Only check right side if no yaw correction was found on left side
    if not yaw_correction_needed:
        # Check columns 38-43 (right side)
        for col in range(38, 44):  # 38 to 43 inclusive
            # Check upper half (rows 0-31)
            upper_half = array_2d[0:32, col]
            if not np.all(upper_half == 0):
                print("Needs to turn right")
                yaw_correction_needed = True
                yaw_direction = "yaw_increase"
                break
            
            # Check lower half (rows 32-63)
            lower_half = array_2d[32:64, col]
            if not np.all(lower_half == 0):
                print("Needs to turn left")
                yaw_correction_needed = True
                yaw_direction = "yaw_decrease"
                break
    
    # Add yaw correction if needed
    if yaw_correction_needed and yaw_direction:
        instructions.append(yaw_direction)
    
    if not any("yaw" in inst for inst in instructions):
        print("Passed Check 2")
    
    # Check 3: Analyze average values in center columns 31 and 32
    # Calculate average of all values in columns 31 and 32
    center_columns = np.concatenate([array_2d[:, 31], array_2d[:, 32]])
    average_value = np.mean(center_columns)
    print(f"Center average: {average_value:.3f}")
    
    # DEBUG: Print detailed center analysis
    col_31_avg = np.mean(array_2d[:, 31])
    col_32_avg = np.mean(array_2d[:, 32])
    print(f"DEBUG: Column 31 avg: {col_31_avg:.3f}, Column 32 avg: {col_32_avg:.3f}")
    
    if average_value < 2.0 or average_value > 4.0:
        print("Too off-centered")
        
        # Additional check: analyze column 29 for directional feedback
        column_29_average = np.mean(array_2d[:, 29])
        print(f"DEBUG: Column 29 average: {column_29_average:.3f}")
        
        if column_29_average < 0.5:
            print("Needs to shift left")
            instructions.append("left")
        elif column_29_average > 1.0:
            print("Needs to shift right")
            instructions.append("right")
    else:
        print("Passed Check 3")
    
    # If all checks passed (no corrections needed), send forward command
    if len(instructions) == 0:
        print("All checks passed - sending forward command")
        instructions.append("forward")

    # TESTING INSTRUCTIONS
    instructions = ["forward", "right", "yaw_increase"]
    
    # Send all instructions (no priority filtering)
    send_instructions(client_socket, instructions)
    
    # Return True if no corrections needed, False if corrections were made
    return len(instructions) == 1 and instructions[0] == "forward"

def send_instructions(client_socket, instructions):
    """
    Send compiled instructions as a single JSON command to the controller.
    
    Args:
        client_socket: The socket connection to the controller
        instructions: List of instruction strings (e.g., ["forward", "left", "yaw_increase"])
    """
    # Initialize command values array [forward, backward, left, right, yaw_increase, 
    # yaw_decrease, height_diff_increase, height_diff_decrease, reset_simulation]
    command_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Map instructions to command values
    for instruction in instructions:
        if instruction == "forward":
            command_values[0] = 1
        elif instruction == "backward":
            command_values[1] = 1
        elif instruction == "left":
            command_values[2] = 1
        elif instruction == "right":
            command_values[3] = 1
        elif instruction == "yaw_increase":
            command_values[4] = 1
        elif instruction == "yaw_decrease":
            command_values[5] = 1
        elif instruction == "height_diff_increase":
            command_values[6] = 1
        elif instruction == "height_diff_decrease":
            command_values[7] = 1
        elif instruction == "reset_simulation":
            command_values[8] = 1
    
    # Send the compiled command
    send_command(client_socket, command_values)
    
    # Print summary of instructions sent
    if instructions:
        print(f"Sent instructions: {', '.join(instructions)}")
    else:
        print("No corrections needed - drone is on track")

def send_command(client_socket, command_values):
    """
    Send a JSON command to the controller.
    
    Args:
        client_socket: The socket connection to the controller
        command_values: Array of 9 values [forward, backward, left, right, yaw_increase, 
                        yaw_decrease, height_diff_increase, height_diff_decrease, 
                        reset_simulation]
    """
    # Create JSON command structure
    command_json = {
        "forward": command_values[0],
        "backward": command_values[1], 
        "left": command_values[2],
        "right": command_values[3],
        "yaw_increase": command_values[4],
        "yaw_decrease": command_values[5],
        "height_diff_increase": command_values[6],
        "height_diff_decrease": command_values[7],
        "reset_simulation": command_values[8]
    }
    
    # Convert to JSON string and send
    json_string = json.dumps(command_json)
    client_socket.sendall(json_string.encode('utf-8'))
    print(f"Sent command: {json_string}")

# Define HOSTNAME and PORT
hostname = 'localhost'
port = 8080

server_address = (hostname, port)  # Define the server's address and port

# Create server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(server_address)
server_socket.listen(1)
server_socket.setblocking(False)  # Non-blocking

print(f"Server listening on port {port}")

while True:
    try:
        # Accept client connection
        try:
            client_socket, client_address = server_socket.accept()
            print(f"Controller connected from {client_address}")
            client_socket.setblocking(False)  # Non-blocking
            
            while True:
                # Receive binary array data from the controller
                # Expecting 4096 bytes (64x64 int8_t array)
                try:
                    response = client_socket.recv(4096)
                    
                    if len(response) == 4096:
                        # Convert binary data to numpy array
                        array_2d = np.frombuffer(response, dtype=np.int8).reshape((64, 64))
                        
                        print(f"\n=== RECEIVER RECEIVED 2D ARRAY (64x64) ===")
                        print("0=black, 1=dark_gray, 2=medium_gray, 3=light_gray, 4=white")
                        print("-" * 130)
                        
                        # Print the entire array
                        for y in range(64):
                            row_str = ""
                            for x in range(64):
                                row_str += f"{array_2d[y, x]} "
                            print(row_str)
                        
                        print("-" * 130)
                        print(f"Array shape: {array_2d.shape}")
                        print(f"Data type: {array_2d.dtype}")
                        print(f"Min value: {array_2d.min()}, Max value: {array_2d.max()}")
                        print(f"Non-zero pixels: {np.count_nonzero(array_2d)} / {array_2d.size}")

                        # Process the array to check curve centering and send appropriate commands
                        process_array(array_2d, client_socket)

                        # No delay - send commands immediately after processing
                    elif len(response) > 0:
                        print(f"Received {len(response)} bytes (expected 4096)")
                        print(f"First few bytes: {response[:20]}")
                    elif len(response) == 0:
                        # Client disconnected
                        print("Controller disconnected")
                        client_socket.close()
                        break
                        
                except socket.error as e:
                    # No data available (non-blocking)
                    pass
                    
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
        except socket.error:
            # No client connected yet
            pass
            
        time.sleep(1)  # Wait before checking for connections again

    except KeyboardInterrupt:
        print("\nShutting down server...")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)

server_socket.close()