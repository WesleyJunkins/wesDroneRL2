# Receiver for the Webots drone controller
# This receiver simulates a discrete-command interface
#   Meaning it receives an array, processes it, and sends a single set of commands (increase speed, turn left, etc...) as opposed to continuously adjusting values based on the current state of the array.
# By Wesley Junkins

import socket
import time
import numpy as np
import json

# Function to take the array, and make sure the drone is still following the curve
def process_array(array_2d):
    """
    Check if the curve is properly centered in the array.
    
    Args:
        array_2d: 2D numpy array (64x64) containing the curve data
        
    Returns:
        bool: True if curve is properly centered, False otherwise
    """
    # Check columns 0-19 (should be all 0's)
    for col in range(20):  # 0 to 19 inclusive
        if not np.all(array_2d[:, col] == 0):
            print(f"Check 1 failed: Column {col} contains non-zero values")
            return False
    
    # Check columns 43-63 (should be all 0's)
    for col in range(43, 64):  # 43 to 63 inclusive
        if not np.all(array_2d[:, col] == 0):
            print(f"Check 1 failed: Column {col} contains non-zero values")
            return False
    
    print("Passed Check 1")
    
    # Check 2: Analyze columns 20-24 and 38-43 for turn direction
    # Check columns 20-24 (left side)
    for col in range(20, 25):  # 20 to 25 inclusive
        # Check upper half (rows 0-31)
        upper_half = array_2d[0:32, col]
        if not np.all(upper_half == 0):
            print("Needs to turn left")
            return True
        
        # Check lower half (rows 32-63)
        lower_half = array_2d[32:64, col]
        if not np.all(lower_half == 0):
            print("Needs to turn right")
            return True
    
    # Check columns 38-43 (right side)
    for col in range(38, 44):  # 38 to 43 inclusive
        # Check upper half (rows 0-31)
        upper_half = array_2d[0:32, col]
        if not np.all(upper_half == 0):
            print("Needs to turn right")
            return True
        
        # Check lower half (rows 32-63)
        lower_half = array_2d[32:64, col]
        if not np.all(lower_half == 0):
            print("Needs to turn left")
            return True
    
    print("Passed Check 2")
    
    # Check 3: Analyze average values in center columns 31 and 32
    # Calculate average of all values in columns 31 and 32
    center_columns = np.concatenate([array_2d[:, 31], array_2d[:, 32]])
    average_value = np.mean(center_columns)
    print(average_value)
    
    if average_value < 2.0 or average_value > 4.0:
        print("Too off-centered")
        
        # Additional check: analyze column 29 for directional feedback
        column_29_average = np.mean(array_2d[:, 29])
        
        if column_29_average < 0.5:
            print("Needs to shift left")
        elif column_29_average > 1.0:
            print("Needs to shift right")
        
        return True
    else:
        print("Passed Check 3")
        return True

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
                        print(f"Received 2D array (64x64):")
                        # Set numpy to print full array without truncation
                        np.set_printoptions(threshold=np.inf, linewidth=130)
                        print(array_2d)
                        print(f"Array shape: {array_2d.shape}")
                        print(f"Data type: {array_2d.dtype}")
                        print(f"Min value: {array_2d.min()}, Max value: {array_2d.max()}")
                        print("-" * 50)

                        # Process the array to check curve centering
                        process_array(array_2d)

                        # Send command to the controller
                        send_command(client_socket, [0, 0, 0, 0, 0, 0, 0, 0, 1])

                        # Wait for 0.5 seconds
                        time.sleep(0.5)
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