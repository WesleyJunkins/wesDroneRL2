#!/usr/bin/env python3
"""
Drone Control System - Receiver Module
=====================================

This module provides a keyboard-based control interface for a Crazyflie drone
running in Webots simulation. It communicates with the drone controller via
socket connection and receives 2D arrays.

Features:
- Socket communication with Webots controller
- 2D array reception and processing
- Automatic reconnection on connection loss
- Random command generation based on received arrays

Author: Wesley Junkins
"""

import socket
import json
import time
import sys
import numpy as np
import random
from pynput import keyboard

# Configuration constants
SOCKET_HOST = 'localhost'
SOCKET_PORT = 8080

# Keyboard to command mapping
CONTROL_KEYS = {
    'w': 'forward',      # Move forward
    's': 'backward',     # Move backward
    'a': 'left',         # Move left
    'd': 'right',        # Move right
    'q': 'yaw_decrease', # Turn left (yaw decrease)
    'e': 'yaw_increase', # Turn right (yaw increase)
    'r': 'height_diff_increase',   # Increase altitude
    'f': 'height_diff_decrease',   # Decrease altitude
    'space': 'reset_simulation'    # Reset simulation
}

class DroneController:
    """
    Main controller class for drone communication and control.
    
    Handles:
    - Socket communication with Webots controller
    - 2D array reception and processing
    - Random command generation
    - Automatic reconnection
    """
    
    def __init__(self):
        """Initialize the drone controller with default state."""
        self.socket = None
        # Control state - all commands start as 0 (inactive)
        self.control_state = {
            'forward': 0,
            'backward': 0,
            'left': 0,
            'right': 0,
            'yaw_increase': 0,
            'yaw_decrease': 0,
            'height_diff_increase': 0,
            'height_diff_decrease': 0,
            'reset_simulation': 0,
            'request_image': 0  # Special command to request arrays
        }
        self.running = True
        self.last_sent_state = None
        
        # Array processing state
        self.array_buffer = b''           # Accumulates incoming array data
        self.expecting_array = False      # True when receiving array data
        self.array_header = None          # (width, height) tuple
        
        # Random command generation
        self.commands_available = [
            'forward', 'backward', 'left', 'right',
            'yaw_increase', 'yaw_decrease',
            'height_diff_increase', 'height_diff_decrease',
            'reset_simulation'  # This will restart the simulation
        ]
        self.array_received = False       # Flag to indicate when to send a command
        
    def connect_socket(self):
        """
        Establish socket connection to the Webots controller.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((SOCKET_HOST, SOCKET_PORT))
            print(f"Connected to Webots controller on {SOCKET_HOST}:{SOCKET_PORT}")
            return True
        except Exception as e:
            print(f"Failed to connect to Webots controller: {e}")
            print("Make sure the Webots simulation is running and the controller is active")
            return False
    
    def send_random_command(self):
        """
        Send a random command to the controller.
        
        This method is called continuously and sends random commands
        regardless of whether arrays are received.
        """
        if self.socket:
            try:
                # Reset all commands to 0
                for command in self.control_state:
                    self.control_state[command] = 0
                
                # Select one random command (excluding request_image)
                random_command = random.choice(self.commands_available)
                self.control_state[random_command] = 1
                
                # Send the command
                json_data = json.dumps(self.control_state)
                self.socket.send(json_data.encode('utf-8'))
                print(f"Sent random command: {random_command}")
                
                # Reset the flag if array was received
                if self.array_received:
                    self.array_received = False
                
            except Exception as e:
                print(f"Failed to send random command: {e}")
                self.reconnect_socket()
        elif not self.socket:
            # No socket connection, try to connect but don't send commands
            if not self.connect_socket():
                # Don't print anything to avoid spam - just silently try to connect
                pass
    
    def receive_2d_array(self):
        """
        Receive and process 2D arrays from the Webots controller.
        
        This method handles the 2D array protocol:
        1. Receives array header in format: "ARRAY:WIDTHxHEIGHT:"
        2. Receives binary array data
        3. Prints the 2D array to console
        4. Sends a random command after processing
        """
        if not self.socket:
            return
        
        data = None
        try:
            # Set socket to non-blocking to avoid blocking the main loop
            self.socket.setblocking(False)
            
            # Try to receive data from the controller
            data = self.socket.recv(4096)
            if data:
                print(f"Received data: {len(data)} bytes")
                print(f"Data starts with: {data[:20]}")
                # Check if this packet contains an array header
                if data.startswith(b'ARRAY:'):
                    print("Found ARRAY header!")
                else:
                    print("Data does not start with ARRAY:")
            
        except BlockingIOError:
            # This is expected with non-blocking sockets when no data is available
            pass
        except Exception as e:
            print(f"Error receiving data: {e}")
            pass
            
        # If we received data, process it
        if data:
            # Check if this packet contains an array header
            if data.startswith(b'ARRAY:'):
                print("Found ARRAY header!")
                # Parse the array header to get dimensions
                header_end = data.find(b':', 6)  # Skip "ARRAY:"
                if header_end != -1:
                    # Extract only the header part (avoiding binary data)
                    header_bytes = data[:header_end + 1]
                    header_str = header_bytes.decode('utf-8')
                    
                    # Parse header format: "ARRAY:WIDTHxHEIGHT:"
                    header_parts = header_str.split(':')
                    if len(header_parts) >= 2:
                        dims = header_parts[1].split('x')
                        if len(dims) == 2:
                            # Extract array dimensions
                            width = int(dims[0])
                            height = int(dims[1])
                            
                            # Set up array reception
                            self.array_header = (width, height)
                            self.expecting_array = True
                            
                            # Extract initial array data that came with the header
                            array_data_part = data[header_end + 1:]
                            self.array_buffer = array_data_part
                            
                            print(f"Receiving 2D array: {width}x{height}")
                            return
            
            # If we're expecting array data, accumulate it
            if self.expecting_array and self.array_header:
                self.array_buffer += data
                width, height = self.array_header
                expected_size = width * height  # int8_t = 1 byte per element
                
                # Check if we have received the complete array
                if len(self.array_buffer) >= expected_size:
                    # Process the complete array
                    array_data = np.frombuffer(self.array_buffer[:expected_size], dtype=np.int8)
                    array_2d = array_data.reshape((height, width))
                    
                    # Print the 2D array to console
                    print(f"\nReceived 2D array ({height}x{width}):")
                    print("0=black, 1=dark_gray, 2=medium_gray, 3=light_gray, 4=white")
                    print("-" * (width * 2 + 10))
                    for row in array_2d:
                        print(" ".join(map(str, row)))
                    print("-" * (width * 2 + 10))
                    
                    # Set flag to send a random command after array processing
                    self.array_received = True
                    
                    # Request another array after successful reception
                    self.control_state['request_image'] = 1
                    time.sleep(0.1)  # Small delay to ensure the request is sent
                    self.control_state['request_image'] = 0
                    
                    # Reset for next array
                    self.expecting_array = False
                    self.array_header = None
                    self.array_buffer = b''
    
    def reconnect_socket(self):
        """
        Attempt to reconnect to the Webots controller.
        
        This method is called when the connection is lost
        and allows the system to automatically recover.
        """
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        print("Attempting to reconnect...")
        if self.connect_socket():
            print("Reconnected successfully")
        else:
            print("Failed to reconnect, but continuing to listen...")
    
    def on_key_press(self, key):
        """
        Handle keyboard key press events.
        
        Currently disabled - system uses random commands instead of keyboard input.
        """
        # Keyboard input is disabled in favor of random commands
        pass
    
    def on_key_release(self, key):
        """
        Handle keyboard key release events.
        
        Currently disabled - system uses random commands instead of keyboard input.
        """
        # Keyboard input is disabled in favor of random commands
        pass
    
    def on_escape(self, key):
        """
        Handle escape key to exit the program gracefully.
        
        Returns:
            bool: False to stop the keyboard listener
        """
        if key == keyboard.Key.esc:
            print("Escape pressed. Exiting...")
            self.running = False
            return False
    
    def print_controls(self):
        """Display the control interface instructions to the user."""
        print("\n" + "="*50)
        print("DRONE CONTROL INTERFACE")
        print("="*50)
        print("System Mode: RANDOM COMMAND GENERATION")
        print("="*50)
        print("Available Commands:")
        print("  forward - Move forward")
        print("  backward - Move backward") 
        print("  left - Move left")
        print("  right - Move right")
        print("  yaw_increase - Turn right")
        print("  yaw_decrease - Turn left")
        print("  height_diff_increase - Increase altitude")
        print("  height_diff_decrease - Decrease altitude")
        print("  reset_simulation - Restart simulation")
        print("="*50)
        print("System will send random commands continuously")
        print("No keyboard input required - ESC to exit")
        print("Will automatically reconnect if connection is lost")
        print("="*50 + "\n")
    
    def run(self):
        """
        Main run loop for the drone controller.
        
        This method:
        1. Establishes initial connection
        2. Sets up keyboard listeners
        3. Runs the main control loop
        4. Handles cleanup on exit
        """
        # Try to connect initially, but don't exit if it fails
        self.connect_socket()
        
        self.print_controls()
        
        # Set up keyboard listener for control commands
        with keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        ) as listener:
            
            # Set up separate listener for escape key
            escape_listener = keyboard.Listener(on_press=self.on_escape)
            escape_listener.start()
            
            try:
                last_status_time = time.time()
                while self.running:
                    current_time = time.time()
                    
                    # Send random commands continuously
                    self.send_random_command()
                    
                    # Receive 2D arrays
                    if self.socket:
                        self.receive_2d_array()
                    else:
                        # Try to connect if not connected
                        self.connect_socket()
                    
                    # Print status every 5 seconds
                    if current_time - last_status_time >= 5.0:
                        # Show connection status
                        if self.socket:
                            print("✓ Connected - Sending random commands continuously")
                            print(f"Socket info: {self.socket}")
                            try:
                                print(f"Remote address: {self.socket.getpeername()}")
                            except OSError:
                                print("Remote address: Unknown (socket may be closed)")
                        else:
                            print("✗ Not connected - waiting for Webots controller...")
                        
                        last_status_time = current_time
                    
                    time.sleep(0.01)  # Small delay to prevent high CPU usage
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            
            finally:
                # Cleanup
                escape_listener.stop()
                if self.socket:
                    self.socket.close()
                print("Controller disconnected")

def main():
    """Main entry point for the drone control system."""
    print("Starting Drone Controller...")
    controller = DroneController()
    controller.run()

if __name__ == "__main__":
    main() 