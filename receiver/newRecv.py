import socket
import time
import numpy as np
import json

def send_command(client_socket, command_values):
    """
    Send a JSON command to the controller.
    
    Args:
        client_socket: The socket connection to the controller
        command_values: Array of 10 values [forward, backward, left, right, yaw_increase, 
                        yaw_decrease, height_diff_increase, height_diff_decrease, 
                        reset_simulation, reset_values]
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
        "reset_simulation": command_values[8],
        "reset_values": command_values[9] # Also to stop the drones flight temporarily
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

                        # Send command to the controller
                        send_command(client_socket, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

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