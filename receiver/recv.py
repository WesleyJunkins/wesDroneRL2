#!/usr/bin/env python3
"""
Drone Control System - Receiver Module
=====================================

This module provides a keyboard-based control interface for a Crazyflie drone
running in Webots simulation. It communicates with the drone controller via
socket connection and receives camera images.

Features:
- Real-time keyboard control (WASD, QE, RF, SPACE, ESC)
- Socket communication with Webots controller
- Camera image reception and saving
- Automatic reconnection on connection loss
- Command-based image transmission

Author: Wesley Junkins
"""

import socket
import json
import time
import sys
import numpy as np
import cv2
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
    - Keyboard input processing
    - Camera image reception and processing
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
            'request_image': 0  # Special command to request camera images
        }
        self.running = True
        self.last_sent_state = None
        
        # Image processing state
        self.image_buffer = b''           # Accumulates incoming image data
        self.expecting_image = False      # True when receiving image data
        self.image_header = None          # (width, height, channels) tuple
        
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
    
    def send_continuous_commands(self):
        """
        Send control commands continuously to the controller.
        
        This method is called in the main loop to maintain
        constant communication with the Webots controller.
        """
        if self.socket:
            try:
                json_data = json.dumps(self.control_state)
                self.socket.send(json_data.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send continuous command: {e}")
                self.reconnect_socket()
        else:
            # No socket connection, try to connect but don't send commands
            if not self.connect_socket():
                # Don't print anything to avoid spam - just silently try to connect
                pass
    
    def receive_camera_image(self):
        """
        Receive and process camera images from the Webots controller.
        
        This method handles the binary image protocol:
        1. Receives image header in format: "IMAGE:WIDTHxHEIGHTxCHANNELS:"
        2. Accumulates binary image data
        3. Saves complete images as JPEG files
        4. Requests additional images after successful reception
        """
        if not self.socket:
            return
        
        try:
            # Set socket to non-blocking to avoid blocking the main loop
            self.socket.setblocking(False)
            
            # Try to receive data from the controller
            data = self.socket.recv(4096)
            if data:
                # Check if this packet contains an image header
                if data.startswith(b'IMAGE:'):
                    # Parse the image header to get dimensions
                    header_end = data.find(b':', 6)  # Skip "IMAGE:"
                    if header_end != -1:
                        # Extract only the header part (avoiding binary data)
                        header_bytes = data[:header_end + 1]
                        header_str = header_bytes.decode('utf-8')
                        
                        # Parse header format: "IMAGE:WIDTHxHEIGHTxCHANNELS:"
                        header_parts = header_str.split(':')
                        if len(header_parts) >= 2:
                            dims = header_parts[1].split('x')
                            if len(dims) == 3:
                                # Extract image dimensions
                                width = int(dims[0])
                                height = int(dims[1])
                                channels = int(dims[2])
                                
                                # Set up image reception
                                self.image_header = (width, height, channels)
                                self.expecting_image = True
                                
                                # Extract initial image data that came with the header
                                image_data_part = data[header_end + 1:]
                                self.image_buffer = image_data_part
                                
                                print(f"Receiving camera image: {width}x{height}, {channels} channels")
                                return
                
                # If we're expecting image data, accumulate it
                if self.expecting_image and self.image_header:
                    self.image_buffer += data
                    width, height, channels = self.image_header
                    expected_size = width * height * channels
                    
                    # Check if we have received the complete image
                    if len(self.image_buffer) >= expected_size:
                        # Process the complete image
                        image_data = np.frombuffer(self.image_buffer[:expected_size], dtype=np.uint8)
                        image = image_data.reshape((height, width, channels))
                        
                        # Convert RGBA to RGB for saving (Webots uses RGBA format)
                        if channels == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                        elif channels == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Save the image as JPEG
                        cv2.imwrite('drone_camera.jpg', image)
                        print(f"Saved camera image: {width}x{height}, {channels} channels")
                        
                        # Request another image after successful reception
                        self.control_state['request_image'] = 1
                        time.sleep(0.1)  # Small delay to ensure the request is sent
                        self.control_state['request_image'] = 0
                        
                        # Reset for next image
                        self.expecting_image = False
                        self.image_header = None
                        self.image_buffer = b''
                        
        except Exception as e:
            # Non-blocking socket will throw exception when no data available
            # This is normal and expected behavior
            pass
    
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
        
        Maps keyboard input to drone control commands
        and updates the control state accordingly.
        """
        try:
            # Convert key to string for easier handling
            if hasattr(key, 'char') and key.char:
                key_str = key.char.lower()
            elif key == keyboard.Key.space:
                key_str = 'space'
            else:
                return
            
            # Check if it's a control key and update state
            if key_str in CONTROL_KEYS:
                control_name = CONTROL_KEYS[key_str]
                self.control_state[control_name] = 1
                print(f"Key pressed: {key_str} -> {control_name}")
                
        except AttributeError:
            pass
    
    def on_key_release(self, key):
        """
        Handle keyboard key release events.
        
        Resets control commands when keys are released
        to ensure precise control.
        """
        try:
            # Convert key to string for easier handling
            if hasattr(key, 'char') and key.char:
                key_str = key.char.lower()
            elif key == keyboard.Key.space:
                key_str = 'space'
            else:
                return
            
            # Check if it's a control key and reset state
            if key_str in CONTROL_KEYS:
                control_name = CONTROL_KEYS[key_str]
                self.control_state[control_name] = 0
                print(f"Key released: {key_str} -> {control_name}")
                
        except AttributeError:
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
        print("Controls:")
        print("  W - Forward")
        print("  S - Backward") 
        print("  A - Left")
        print("  D - Right")
        print("  Q - Yaw Decrease (Turn Left)")
        print("  E - Yaw Increase (Turn Right)")
        print("  R - Height Increase")
        print("  F - Height Decrease")
        print("  SPACE - Reset Simulation")
        print("  ESC - Exit")
        print("="*50)
        print("Press keys to control the drone...")
        print("System will continuously send commands as fast as possible")
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
                    
                    # Send commands continuously as fast as possible
                    self.send_continuous_commands()
                    
                    # Receive camera images
                    self.receive_camera_image()
                    
                    # Print status every 5 seconds
                    if current_time - last_status_time >= 5.0:
                        active_controls = [k for k, v in self.control_state.items() if v == 1]
                        
                        # Show connection status and active controls
                        if self.socket:
                            if active_controls:
                                print(f"✓ Connected - Active controls: {active_controls}")
                            else:
                                print("✓ Connected - Sending idle state (all controls: 0)")
                        else:
                            if active_controls:
                                print(f"✗ Not connected - Active controls: {active_controls} (not sending)")
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