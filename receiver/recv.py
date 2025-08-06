#!/usr/bin/env python3
"""
Keyboard to Socket Controller for Webots Crazyflie
Detects keyboard input and sends JSON control commands to the Webots controller
"""

import socket
import json
import time
import sys
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
            'reset_simulation': 0
        }
        self.running = True
        self.last_sent_state = None
        
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
            print("Failed to reconnect")
            self.running = False
    
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
        print("="*50 + "\n")
    
    def run(self):
        """Main run loop"""
        if not self.connect_socket():
            return
        
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
                    
                    # Print status every 5 seconds
                    if current_time - last_status_time >= 5.0:
                        active_controls = [k for k, v in self.control_state.items() if v == 1]
                        if active_controls:
                            print(f"Active controls: {active_controls}")
                        else:
                            print("Sending idle state (all controls: 0)")
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