# Drone Control System

This project provides a socket-based control system for a Crazyflie drone in Webots simulation.

## Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Compile the Webots Controller
```bash
cd controllers/wes_flight_controller_crazyflie
make clean
make
```

## Usage

### 1. Start Webots Simulation
1. Open Webots
2. Load the world file: `worlds/wesDroneRL2.wbt`
3. Start the simulation
4. The controller will automatically start listening on port 8080

### 2. Run the Keyboard Controller
```bash
python recv.py
```

### 3. Control the Drone
Use the following keys to control the drone:

- **W** - Move Forward
- **S** - Move Backward
- **A** - Move Left
- **D** - Move Right
- **Q** - Yaw Decrease (Turn Left)
- **E** - Yaw Increase (Turn Right)
- **R** - Increase Height
- **F** - Decrease Height
- **SPACE** - Reset Simulation
- **ESC** - Exit the controller

## Architecture

### Controller (`wes_flight_controller_crazyflie.c`)
- Runs inside Webots simulation
- Listens for JSON commands on port 8080
- Maintains persistent socket connection
- Parses control commands and applies them to the drone
- Handles PID control for stable flight

### Keyboard Interface (`recv.py`)
- Detects keyboard input using pynput
- Sends JSON commands to the controller via socket
- Maintains persistent connection with automatic reconnection
- Provides real-time control feedback

### Communication Protocol
The system uses JSON messages with the following format:
```json
{
  "forward": 1,
  "backward": 0,
  "left": 0,
  "right": 0,
  "yaw_increase": 0,
  "yaw_decrease": 0,
  "height_diff_increase": 0,
  "height_diff_decrease": 0,
  "reset_simulation": 0
}
```

## Troubleshooting

### Connection Issues
- Make sure Webots simulation is running before starting `recv.py`
- Check that port 8080 is not being used by another application
- Verify the controller compiled successfully
- The system will automatically attempt to reconnect if the connection is lost

### Keyboard Not Responding
- Ensure you have the necessary permissions for keyboard input
- On macOS, you may need to grant accessibility permissions to Terminal/IDE
- Try running with sudo if permission issues persist

### Drone Not Moving
- Check that the controller is properly initialized in Webots
- Verify the JSON messages are being received (check console output)
- Ensure the drone has proper physics and motor setup in the world file

### macOS Accessibility Permissions
If you see "This process is not trusted!" messages on macOS:
1. Go to System Preferences > Security & Privacy > Privacy
2. Select "Accessibility" from the left sidebar
3. Click the lock icon to make changes
4. Add your Terminal application or IDE to the list
5. Restart the application and try again 