#!/usr/bin/env python3
"""
Keyboard to Socket Controller for Webots Crazyflie
Detects keyboard input and sends JSON control commands to the Webots controller
"""

import socket
import json
import time
import sys
import numpy as np
import cv2
from pynput import keyboard

# Configuration
SOCKET_HOST = 'localhost'
SOCKET_PORT = 8080
CONTROL_KEYS = {
    'w': 'forward',
    's': 'backward', 
    'a': 'left',
    'd': 'right',
    'q': 'yaw_decrease',
    'e': 'yaw_increase',
    'r': 'height_diff_increase',
    'f': 'height_diff_decrease',
    'space': 'reset_simulation'
}

class DroneController:
    def __init__(self):
        self.socket = None
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
            'request_image': 0
        }
        self.running = True
        self.last_sent_state = None
        self.image_buffer = b''
        self.expecting_image = False
        self.image_header = None
        
    def connect_socket(self):
        """Connect to the Webots controller socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((SOCKET_HOST, SOCKET_PORT))
            print(f"Connected to Webots controller on {SOCKET_HOST}:{SOCKET_PORT}")
            return True
        except Exception as e:
            print(f"Failed to connect to Webots controller: {e}")
            print("Make sure the Webots simulation is running and the controller is active")
            return False
    
    def send_control_command(self):
        """Send current control state to the controller"""
        if self.socket:
            try:
                json_data = json.dumps(self.control_state)
                self.socket.send(json_data.encode('utf-8'))
                self.last_sent_state = self.control_state.copy()
                # Only print when there's actual movement (not all zeros)
                if any(self.control_state.values()):
                    print(f"Sent: {json_data}")
            except Exception as e:
                print(f"Failed to send command: {e}")
                # Try to reconnect
                self.reconnect_socket()
    
    def send_continuous_commands(self):
        """Send control commands continuously"""
        if self.socket:
            try:
                json_data = json.dumps(self.control_state)
                self.socket.send(json_data.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send continuous command: {e}")
                # Try to reconnect
                self.reconnect_socket()
        else:
            # No socket connection, try to connect but don't send commands
            if not self.connect_socket():
                # Don't print anything to avoid spam - just silently try to connect
                pass
    
    def receive_camera_image(self):
        """Receive and process camera image from controller"""
        if not self.socket:
            return
        
        try:
            # Set socket to non-blocking
            self.socket.setblocking(False)
            
            # Try to receive data
            data = self.socket.recv(4096)
            if data:
                print(f"Received data: {len(data)} bytes")
                print(f"First 20 bytes: {data[:20]}")
                print(f"Data starts with IMAGE: {data.startswith(b'IMAGE:')}")
                # Check if this is an image header
                if data.startswith(b'IMAGE:'):
                    print("Found image header!")
                    print("DEBUG: Starting header parsing...")
                    try:
                        # Find the end of the header (after the second ':')
                        header_end = data.find(b':', 6)  # Skip "IMAGE:"
                        if header_end != -1:
                            # Extract only the header part (including the second ':')
                            header_bytes = data[:header_end + 1]
                            header_str = header_bytes.decode('utf-8')
                            print(f"Header string: {header_str}")
                            print(f"Header string length: {len(header_str)}")
                            print("About to parse header...")
                            
                            # Parse the header directly
                            # Format is: IMAGE:64x64x4: followed by image data
                            header_parts = header_str.split(':')
                            print(f"Header parts: {header_parts}")
                            if len(header_parts) >= 2:
                                print("Header has enough parts, parsing dimensions...")
                                dims = header_parts[1].split('x')
                                print(f"Dimensions: {dims}")
                                if len(dims) == 3:
                                    print("Dimensions parsed successfully!")
                                    width = int(dims[0])
                                    height = int(dims[1])
                                    channels = int(dims[2])
                                    self.image_header = (width, height, channels)
                                    self.expecting_image = True
                                    
                                    # Extract image data that came with the header
                                    image_data_part = data[header_end + 1:]
                                    self.image_buffer = image_data_part
                                    print(f"Receiving image: {width}x{height}, {channels} channels")
                                    print(f"Initial image data: {len(image_data_part)} bytes")
                                    return
                                else:
                                    print(f"Wrong number of dimensions: {len(dims)}")
                            else:
                                print(f"Not enough header parts: {len(header_parts)}")
                        else:
                            print("Could not find header end position")
                    except Exception as e:
                        print(f"ERROR in header parsing: {e}")
                        import traceback
                        traceback.print_exc()
                
                # If expecting image data, accumulate it
                if self.expecting_image and self.image_header:
                    print(f"Accumulating image data: {len(data)} bytes, buffer size: {len(self.image_buffer)}")
                    self.image_buffer += data
                    width, height, channels = self.image_header
                    expected_size = width * height * channels
                    print(f"Expected size: {expected_size}, current buffer: {len(self.image_buffer)}")
                    
                    if len(self.image_buffer) >= expected_size:
                        # Complete image received
                        image_data = np.frombuffer(self.image_buffer[:expected_size], dtype=np.uint8)
                        image = image_data.reshape((height, width, channels))
                        
                        # Convert RGBA to RGB for saving
                        if channels == 4:
                            # Remove alpha channel and convert to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                        elif channels == 3:
                            # Convert BGR to RGB if needed
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Save or display the image
                        cv2.imwrite('drone_camera.jpg', image)
                        print(f"Saved camera image: {width}x{height}, {channels} channels")
                        
                        # Send a request for another image after receiving one
                        self.control_state['request_image'] = 1
                        # Reset it after sending to prevent continuous requests
                        time.sleep(0.1)  # Small delay to ensure the request is sent
                        self.control_state['request_image'] = 0
                        
                        # Reset for next image
                        self.expecting_image = False
                        self.image_header = None
                        self.image_buffer = b''
                        
        except Exception as e:
            # Non-blocking socket will throw exception when no data
            pass
    
    def reconnect_socket(self):
        """Attempt to reconnect to the controller"""
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
            # Don't stop running, just keep trying
    
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            # Convert key to string for easier handling
            if hasattr(key, 'char') and key.char:
                key_str = key.char.lower()
            elif key == keyboard.Key.space:
                key_str = 'space'
            else:
                return
            
            # Check if it's a control key
            if key_str in CONTROL_KEYS:
                control_name = CONTROL_KEYS[key_str]
                self.control_state[control_name] = 1
                print(f"Key pressed: {key_str} -> {control_name}")
                # The continuous loop will handle sending
                
        except AttributeError:
            pass
    
    def on_key_release(self, key):
        """Handle key release events"""
        try:
            # Convert key to string for easier handling
            if hasattr(key, 'char') and key.char:
                key_str = key.char.lower()
            elif key == keyboard.Key.space:
                key_str = 'space'
            else:
                return
            
            # Check if it's a control key
            if key_str in CONTROL_KEYS:
                control_name = CONTROL_KEYS[key_str]
                self.control_state[control_name] = 0
                print(f"Key released: {key_str} -> {control_name}")
                # The continuous loop will handle sending
                
        except AttributeError:
            pass
    
    def on_escape(self, key):
        """Handle escape key to exit"""
        if key == keyboard.Key.esc:
            print("Escape pressed. Exiting...")
            self.running = False
            return False
    
    def print_controls(self):
        """Print control instructions"""
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
        """Main run loop"""
        # Try to connect initially, but don't exit if it fails
        self.connect_socket()
        
        self.print_controls()
        
        # Set up keyboard listener
        with keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        ) as listener:
            
            # Also listen for escape key
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
                        
                        # Also show connection status
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
                escape_listener.stop()
                if self.socket:
                    self.socket.close()
                print("Controller disconnected")

def main():
    """Main function"""
    print("Starting Drone Controller...")
    controller = DroneController()
    controller.run()

if __name__ == "__main__":
    main() 