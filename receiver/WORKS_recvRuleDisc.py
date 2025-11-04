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
    Advanced line following using computer vision techniques.
    
    Args:
        array_2d: 2D numpy array (64x64) containing the curve data
        client_socket: Socket connection to send commands to the controller
        
    Returns:
        bool: True if curve is properly centered, False otherwise
    """
    # Initialize command array [forward, backward, left, right, yaw_increase, 
    # yaw_decrease, height_diff_increase, height_diff_decrease, reset_simulation]
    command_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Convert to float for better processing
    img = array_2d.astype(np.float32)
    
    # 1. LINE DETECTION: Find the brightest path (line)
    # Use adaptive thresholding - find pixels brighter than mean + std
    # This essentially just finds the brightest line in the image
    mean_val = np.mean(img)
    std_val = np.std(img)
    threshold = mean_val + 0.5 * std_val
    
    # Create line mask
    # This is a binary mask of the pixels that are brighter than the threshold
    # This will make it easier to find the line in the image
    line_mask = (img >= threshold).astype(np.uint8)
    
    # Check if we have enough line pixels
    # Makes sure the line is still in view. If not, the drone will just break from the path
    # I may change this later to just reset the simulation
    line_pixels = np.sum(line_mask) # Counts all the 1s
    if line_pixels < 50:  # Not enough line detected
        print("No clear line detected - continuing forward")
        command_values[0] = 1  # forward
        send_command(client_socket, command_values)
        return False # End early because we can't see the line
    
    # 2. HORIZONTAL CENTERING: Find line center in each row
    # Basically gets the center line of the curve, no matter the thickness of the curve
    rows_with_line = [] # List of rows that have contact with the line
    line_centers = [] # List of the center of the line in each row
    for row in range(64): # Scan each row
        row_pixels = line_mask[row, :] # Get all the pixels in the row
        if np.sum(row_pixels) > 0: # If row has some pixels of the line, then find the center
            # Find center of line in this row
            col_indices = np.where(row_pixels)[0]
            if len(col_indices) > 0:
                center = np.mean(col_indices)
                rows_with_line.append(row)
                line_centers.append(center)
    if len(line_centers) < 5:  # Not enough line segments
        print("Insufficient line segments - continuing forward")
        command_values[0] = 1  # forward
        send_command(client_socket, command_values)
        return False # End early because we can't see the line
    
    # 3. LINE FITTING: Fit a line to the detected points
    rows_with_line = np.array(rows_with_line)
    line_centers = np.array(line_centers)
    
    # Fit line: center = slope * row + intercept
    # This finds the best straight line that fits through all the detected line center points
    # Use robust fitting (least squares) (linear regression)
    A = np.vstack([rows_with_line, np.ones(len(rows_with_line))]).T # A becomes a 2-column matrix with the left column being the rows that have 1s and the right column being all 1s
    # This solves the least squares equation to find the slope and intercept of the line
    # A is the design matrix
    # line_centers is the vector of y-values that we want to fit to the line
    # np.linalg.lstsq is a function that solves the least squares equation, it finds the line that minimizes the sum of squared errors
    # [0] is the first element of the tuple returned by np.linalg.lstsq, which is the solution vector
    slope, intercept = np.linalg.lstsq(A, line_centers, rcond=None)[0]
    
    # 4. CONTROL DECISIONS
    image_center = 31.5  # Center of 64-pixel wide image
    
    # Use the fitted line to determine lateral position at different heights
    # Check where line should be at top (row 10) and middle (row 32) of image
    line_at_top = slope * 10 + intercept # Where the line is heading in the future
    line_at_middle = slope * 32 + intercept# Where the line is currently positioned
    
    # Use the line position at the middle for more stable centering
    # The middle is less affected by camera tilt than the top
    error = line_at_middle - image_center # How far the line is from the center of the image
    
    # Much larger deadband to prevent oscillation from camera movement
    # Only correct when significantly off-center to allow faster forward progress
    if abs(error) > 15.0:  # Very large deadband for smooth following
        if error > 0:
            command_values[2] = 1  # left (line is to the right, move left to center)
        else:
            command_values[3] = 1  # right (line is to the left, move right to center)
    
    # Check line slope for yaw correction
    # Only apply yaw correction for significant curves, not small tilts
    if abs(slope) > 0.2:  # Increased threshold to ignore camera tilt
        if slope > 0:
            command_values[4] = 1  # yaw_increase (turn left to follow right-curving line)
        else:
            command_values[5] = 1  # yaw_decrease (turn right to follow left-curving line)
    
    # Check if line is too far to the sides (forward/backward)
    current_center = slope * 32 + intercept  # Line center at middle of image
    if current_center < 10:  # Line too far left
        command_values[1] = 1  # backward
    elif current_center > 54:  # Line too far right
        command_values[0] = 1  # forward
    
    # Prioritize forward movement - only apply lateral corrections if line is very far off
    # This allows the drone to make faster progress around the track
    if sum(command_values) == 0:
        command_values[0] = 1  # forward
    elif command_values[0] == 0 and (command_values[2] == 1 or command_values[3] == 1):
        # If we're making lateral corrections, also go forward to maintain progress
        command_values[0] = 1  # forward
    
    # Print results with debug info
    command_names = ["forward", "backward", "left", "right", "yaw_increase", 
                    "yaw_decrease", "height_diff_increase", "height_diff_decrease", "reset_simulation"]
    active_commands = [name for i, name in enumerate(command_names) if command_values[i] == 1]
    
    print(f"Line pixels: {line_pixels}, Slope: {slope:.3f}, Intercept: {intercept:.1f}")
    print(f"Commands sent: {', '.join(active_commands)}")
    print(f"Command array: {command_values}")
    
    # Send the command array
    send_command(client_socket, command_values)
    
    # Return True if only forward command is set, False if corrections were made
    return command_values == [1, 0, 0, 0, 0, 0, 0, 0, 0]


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
                        
                        # print(f"\n=== RECEIVER RECEIVED 2D ARRAY (64x64) ===")
                        # print("0=black, 1=dark_gray, 2=medium_gray, 3=light_gray, 4=white")
                        # print("-" * 130)
                        
                        # Print the entire array
                        # for y in range(64):
                        #     row_str = ""
                        #     for x in range(64):
                        #         row_str += f"{array_2d[y, x]} "
                        #     print(row_str)
                        
                        # print("-" * 130)
                        # print(f"Array shape: {array_2d.shape}")
                        # print(f"Data type: {array_2d.dtype}")
                        # print(f"Min value: {array_2d.min()}, Max value: {array_2d.max()}")
                        # print(f"Non-zero pixels: {np.count_nonzero(array_2d)} / {array_2d.size}")

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