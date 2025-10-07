# Receiver for the Webots drone controller
# This receiver simulates a discrete-command interface
#   Meaning it receives an array, processes it, and sends a single set of commands (increase speed, turn left, etc...) as opposed to continuously adjusting values based on the current state of the array.
# By Wesley Junkins

import socket
import time
import numpy as np
import json

class PIDController:
    """
    PID Controller for drone navigation based on array analysis.
    """
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, setpoint=0.0, output_limits=(-1.0, 1.0)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            setpoint: Target value (0.0 for centered)
            output_limits: Tuple of (min, max) output values
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        # Initialize PID variables
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def update(self, current_value, dt=None):
        """
        Update PID controller with current value.
        
        Args:
            current_value: Current measured value
            dt: Time step (if None, will calculate automatically)
            
        Returns:
            float: PID output value
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
            if dt <= 0:
                dt = 0.01  # Prevent division by zero
        
        # Calculate error
        error = self.setpoint - current_value
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Update for next iteration
        self.previous_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

# Initialize PID controllers for different control axes
# These are global variables to maintain state between function calls
forward_pid = PIDController(kp=0.5, ki=0.05, kd=0.02, setpoint=0.0, output_limits=(-1.0, 1.0))
lateral_pid = PIDController(kp=0.8, ki=0.1, kd=0.03, setpoint=0.0, output_limits=(-1.0, 1.0))
yaw_pid = PIDController(kp=1.2, ki=0.15, kd=0.05, setpoint=0.0, output_limits=(-1.0, 1.0))

def process_array(array_2d, client_socket):
    """
    Process the 2D array using PID controllers to maintain proper curve following.
    
    Args:
        array_2d: 2D numpy array (64x64) containing the curve data
        client_socket: Socket connection to send commands to the controller
        
    Returns:
        bool: True if curve is properly centered, False otherwise
    """
    global forward_pid, lateral_pid, yaw_pid
    
    # Calculate control errors for each axis
    
    # 1. Forward/Backward Control (based on curve position in array)
    # Calculate how far the curve extends into forbidden zones
    left_penalty = 0.0
    right_penalty = 0.0
    
    # Check columns 0-19 (should be all 0's) - left forbidden zone
    for col in range(20):
        non_zero_count = np.count_nonzero(array_2d[:, col])
        left_penalty += non_zero_count / (64 * 20)  # Normalize by total possible
    
    # Check columns 43-63 (should be all 0's) - right forbidden zone  
    for col in range(43, 64):
        non_zero_count = np.count_nonzero(array_2d[:, col])
        right_penalty += non_zero_count / (64 * 21)  # Normalize by total possible
    
    # Forward/backward error: positive means too far forward (right penalty), negative means too far back (left penalty)
    forward_error = right_penalty - left_penalty
    
    # 2. Lateral Control (left/right positioning)
    # Calculate center of mass of the curve
    center_columns = np.concatenate([array_2d[:, 31], array_2d[:, 32]])
    center_average = np.mean(center_columns)
    
    # Target center average should be around 3.0 (middle of 0-6 range)
    lateral_error = 3.0 - center_average
    
    # 3. Yaw Control (rotation/heading)
    # Analyze asymmetry in left and right detection zones
    left_upper_detection = 0.0
    left_lower_detection = 0.0
    right_upper_detection = 0.0
    right_lower_detection = 0.0
    
    # Check columns 20-24 (left side)
    for col in range(20, 25):
        upper_half = np.mean(array_2d[0:32, col])
        lower_half = np.mean(array_2d[32:64, col])
        left_upper_detection += upper_half
        left_lower_detection += lower_half
    
    # Check columns 38-43 (right side)
    for col in range(38, 44):
        upper_half = np.mean(array_2d[0:32, col])
        lower_half = np.mean(array_2d[32:64, col])
        right_upper_detection += upper_half
        right_lower_detection += lower_half
    
    # Calculate yaw error based on asymmetry
    # Positive error means need to turn right (clockwise), negative means turn left
    left_asymmetry = left_upper_detection - left_lower_detection
    right_asymmetry = right_upper_detection - right_lower_detection
    yaw_error = (left_asymmetry - right_asymmetry) / 10.0  # Normalize
    
    # Update PID controllers
    forward_output = forward_pid.update(forward_error)
    lateral_output = lateral_pid.update(lateral_error)
    yaw_output = yaw_pid.update(yaw_error)
    
    # Print debug information
    print(f"Forward error: {forward_error:.3f}, Output: {forward_output:.3f}")
    print(f"Lateral error: {lateral_error:.3f}, Output: {lateral_output:.3f}")
    print(f"Yaw error: {yaw_error:.3f}, Output: {yaw_output:.3f}")
    
    # Convert PID outputs to command values
    command_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Map PID outputs to command values (threshold at 0.3 to avoid noise)
    if forward_output > 0.3:
        command_values[0] = min(1, forward_output)  # forward
    elif forward_output < -0.3:
        command_values[1] = min(1, abs(forward_output))  # backward
        
    if lateral_output > 0.3:
        command_values[3] = min(1, lateral_output)  # right
    elif lateral_output < -0.3:
        command_values[2] = min(1, abs(lateral_output))  # left
        
    if yaw_output > 0.3:
        command_values[4] = min(1, yaw_output)  # yaw_increase
    elif yaw_output < -0.3:
        command_values[5] = min(1, abs(yaw_output))  # yaw_decrease
    
    # Send the command
    send_command(client_socket, command_values)
    
    # Return True if all errors are small (drone is on track)
    return abs(forward_error) < 0.1 and abs(lateral_error) < 0.5 and abs(yaw_error) < 0.2

def send_instructions(client_socket, instructions):
    """
    Send compiled instructions as a single JSON command to the controller.
    This function is kept for backward compatibility but is no longer used
    in the PID-controlled version.
    
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
                        print(f"Received 2D array (64x64):")
                        # Set numpy to print full array without truncation
                        np.set_printoptions(threshold=np.inf, linewidth=130)
                        print(array_2d)
                        print(f"Array shape: {array_2d.shape}")
                        print(f"Data type: {array_2d.dtype}")
                        print(f"Min value: {array_2d.min()}, Max value: {array_2d.max()}")
                        print("-" * 50)

                        # Process the array to check curve centering and send appropriate commands
                        process_array(array_2d, client_socket)

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