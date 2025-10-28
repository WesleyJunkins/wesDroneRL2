# Receiver for the Webots drone controller using Reinforcement Learning
# This receiver uses Stable Baselines 3 to learn optimal drone control policies
# By Wesley Junkins

import socket
import time
import numpy as np
import json
import argparse
import os
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch

class DroneControlEnv(gym.Env):
    """
    Custom Gymnasium environment for drone line following control.
    
    State: 64x64 grayscale image from drone camera
    Action: Discrete actions for drone movement commands
    Reward: Based on line following performance
    """
    
    def __init__(self, client_socket=None, training_mode=False):
        super().__init__()
        
        # Action space: 9 discrete actions corresponding to drone commands
        # [forward, backward, left, right, yaw_increase, yaw_decrease, 
        #  height_diff_increase, height_diff_decrease, reset_simulation]
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 64x64 grayscale image
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(64, 64), dtype=np.int8
        )
        
        # Environment state
        self.current_observation = None
        self.client_socket = client_socket
        self.training_mode = training_mode
        self.episode_count = 0
        self.step_count = 0
        self.max_steps_per_episode = 1000
        
        # Performance tracking for reward calculation
        self.previous_line_center = 32.0  # Center of image
        self.consecutive_good_steps = 0
        self.total_reward = 0
        
        # Command mapping
        self.command_names = [
            "forward", "backward", "left", "right", "yaw_increase", 
            "yaw_decrease", "height_diff_increase", "height_diff_decrease", "reset_simulation"
        ]
        
        # For training mode - data collection
        self.data_buffer = []
        self.waiting_for_data = False
        self.reset_requested = False
        self.waiting_for_reset_completion = False
        self.waiting_for_ready_signal = False
        self.is_ready = False
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.consecutive_good_steps = 0
        self.total_reward = 0
        self.previous_line_center = 32.0
        
        # Reset reset-related flags
        self.reset_requested = False
        self.waiting_for_reset_completion = False
        self.waiting_for_ready_signal = False
        self.is_ready = False
        
        if self.training_mode and self.client_socket:
            # In training mode with Webots connection, wait for initial camera data
            self.current_observation = self.wait_for_camera_data()
            self.is_ready = True  # Mark as ready after receiving initial data
        else:
            # Fallback to synthetic data
            self.current_observation = np.zeros((64, 64), dtype=np.int8)
            self.is_ready = True  # Mark as ready for non-training mode
        
        info = {
            "episode_count": self.episode_count,
            "step_count": self.step_count
        }
        
        return self.current_observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Check connection health before proceeding
        if self.training_mode and not self.is_connection_healthy():
            raise ConnectionError("Webots connection is not healthy")
        
        # Check if we're waiting for ready signal - don't send commands yet
        if self.waiting_for_ready_signal:
            print("Waiting for ready signal - not sending commands yet")
            # Return current observation without processing
            reward = 0
            terminated = False
            truncated = False
            
            info = {
                "step_count": self.step_count,
                "total_reward": self.total_reward,
                "consecutive_good_steps": self.consecutive_good_steps,
                "action_taken": "waiting_for_ready",
                "waiting_for_ready": True
            }
            
            return self.current_observation, reward, terminated, truncated, info
        
        # Convert action to command array
        command_values = [0] * 9
        command_values[action] = 1
        
        # Check if this is a reset simulation action
        if action == 8:  # reset_simulation action
            self.reset_requested = True
            self.waiting_for_reset_completion = True
            self.waiting_for_ready_signal = True
            self.is_ready = False
            print("Reset simulation requested - waiting for drone to restart...")
        
        # Send command to drone controller (only if ready)
        if self.client_socket and self.is_ready:
            try:
                self.send_command(command_values)
            except ConnectionError:
                # Connection lost during command sending
                raise ConnectionError("Lost connection while sending command")
        
        # Handle reset simulation waiting
        if self.waiting_for_reset_completion:
            try:
                # Wait for reset completion and ready signal
                self.current_observation = self.wait_for_reset_completion()
                self.waiting_for_reset_completion = False
                self.reset_requested = False
                self.waiting_for_ready_signal = False
                self.is_ready = True
                print("Reset completed - drone is ready for new episode")
                
                # Terminate current episode after reset
                reward = 0  # No reward for reset action
                terminated = True
                truncated = False
                
                info = {
                    "step_count": self.step_count,
                    "total_reward": self.total_reward,
                    "consecutive_good_steps": self.consecutive_good_steps,
                    "action_taken": self.command_names[action],
                    "episode_terminated": "reset_simulation"
                }
                
                return self.current_observation, reward, terminated, truncated, info
                
            except ConnectionError as e:
                # Connection lost during reset - re-raise to trigger recovery
                raise e
        
        # Get new observation after action (for non-reset actions)
        if self.training_mode and self.client_socket and self.is_ready:
            # In training mode, wait for new camera data after sending command
            try:
                self.current_observation = self.wait_for_camera_data()
            except ConnectionError as e:
                # Connection lost during data reception - re-raise to trigger recovery
                raise e
        
        # Calculate reward based on current observation
        reward = self.calculate_reward(self.current_observation, action)
        self.total_reward += reward
        
        # Check if episode is done
        terminated = self.step_count >= self.max_steps_per_episode
        truncated = False  # We don't truncate episodes
        
        # Update consecutive good steps counter
        if reward > 0:
            self.consecutive_good_steps += 1
        else:
            self.consecutive_good_steps = 0
        
        info = {
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "consecutive_good_steps": self.consecutive_good_steps,
            "action_taken": self.command_names[action]
        }
        
        return self.current_observation, reward, terminated, truncated, info
    
    def calculate_reward(self, observation, action):
        """
        Calculate reward based on line following performance.
        
        Args:
            observation: Current camera image (64x64)
            action: Action taken by the agent
            
        Returns:
            float: Reward value
        """
        # Handle reset simulation action - no reward calculation needed
        if action == 8:  # reset_simulation action
            return 0.0
        
        try:
            # Convert to float for processing
            img = observation.astype(np.float32)
            
            # Find line using adaptive thresholding
            mean_val = np.mean(img)
            std_val = np.std(img)
            threshold = mean_val + 0.5 * std_val
            line_mask = (img >= threshold).astype(np.uint8)
            
            # Count line pixels
            line_pixels = np.sum(line_mask)
            
            # Base reward for detecting line
            if line_pixels < 50:
                return -10.0  # Heavy penalty for losing the line
            
            # Find line center
            line_centers = []
            for row in range(64):
                row_pixels = line_mask[row, :]
                if np.sum(row_pixels) > 0:
                    col_indices = np.where(row_pixels)[0]
                    if len(col_indices) > 0:
                        center = np.mean(col_indices)
                        line_centers.append(center)
            
            if len(line_centers) < 5:
                return -5.0  # Penalty for insufficient line segments
            
            # Calculate current line center (average of all detected centers)
            current_line_center = np.mean(line_centers)
            image_center = 31.5
            
            # Reward for centering the line
            centering_error = abs(current_line_center - image_center)
            centering_reward = max(0, 10 - centering_error)  # Higher reward for better centering
            
            # Reward for forward progress (action 0 = forward)
            progress_reward = 1.0 if action == 0 else 0.0
            
            # Penalty for excessive corrections
            correction_penalty = 0
            if action in [1, 2, 3, 4, 5]:  # backward, left, right, yaw actions
                if centering_error < 10:  # Don't penalize if correction is needed
                    correction_penalty = -0.5
            
            # Bonus for consecutive good steps
            stability_bonus = min(2.0, self.consecutive_good_steps * 0.1)
            
            total_reward = centering_reward + progress_reward + correction_penalty + stability_bonus
            
            # Update previous line center for next step
            self.previous_line_center = current_line_center
            
            return total_reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return -1.0
    
    def update_observation(self, observation):
        """Update the current observation with new camera data."""
        self.current_observation = observation.copy()
    
    def is_connection_healthy(self):
        """Check if the Webots connection is still healthy."""
        if not self.client_socket:
            return False
        
        try:
            # Try to send a small ping to check connection
            self.client_socket.send(b'ping')
            return True
        except socket.error:
            return False
        except Exception:
            return False
    
    def wait_for_camera_data(self, timeout=5.0):
        """Wait for camera data from Webots during training."""
        if not self.client_socket:
            return np.zeros((64, 64), dtype=np.int8)
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Keep socket in blocking mode for consistent behavior
                response = self.client_socket.recv(4096)
                
                if len(response) == 4096:
                    # Convert binary data to numpy array
                    array_2d = np.frombuffer(response, dtype=np.int8).reshape((64, 64))
                    return array_2d
                elif len(response) == 0:
                    # Connection lost - raise exception to trigger recovery
                    raise ConnectionError("Webots connection lost during training")
                    
            except socket.error as e:
                if e.errno == 32:  # Broken pipe
                    raise ConnectionError("Webots connection lost during training")
                elif e.errno == 11:  # Resource temporarily unavailable (non-blocking)
                    time.sleep(0.01)  # Small delay
                    continue
                else:
                    raise ConnectionError(f"Socket error: {e}")
        
        # Timeout - raise exception to trigger recovery
        raise ConnectionError(f"Timeout waiting for camera data after {timeout}s")
    
    def wait_for_reset_completion(self, timeout=10.0):
        """Wait for reset simulation to complete and drone to send ready signal."""
        if not self.client_socket:
            return np.zeros((64, 64), dtype=np.int8)
        
        import time
        start_time = time.time()
        print("Waiting for reset completion and ready signal...")
        
        while time.time() - start_time < timeout:
            try:
                # Keep socket in blocking mode for consistent behavior
                response = self.client_socket.recv(4096)
                
                if len(response) == 4096:
                    # Convert binary data to numpy array
                    array_2d = np.frombuffer(response, dtype=np.int8).reshape((64, 64))
                    print("Received ready signal after reset - drone is back in the air")
                    return array_2d
                elif len(response) == 0:
                    # Connection lost - raise exception to trigger recovery
                    raise ConnectionError("Webots connection lost during reset")
                    
            except socket.error as e:
                if e.errno == 32:  # Broken pipe
                    raise ConnectionError("Webots connection lost during reset")
                elif e.errno == 11:  # Resource temporarily unavailable (non-blocking)
                    time.sleep(0.01)  # Small delay
                    continue
                else:
                    raise ConnectionError(f"Socket error during reset: {e}")
        
        # Timeout - raise exception to trigger recovery
        raise ConnectionError(f"Timeout waiting for reset completion after {timeout}s")
    
    def send_command(self, command_values):
        """Send command to drone controller."""
        if not self.client_socket:
            return
            
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
        
        json_string = json.dumps(command_json)
        try:
            self.client_socket.sendall(json_string.encode('utf-8'))
            print(f"RL Agent sent command: {self.command_names[command_values.index(1)]}")
        except socket.error as e:
            if e.errno == 32:  # Broken pipe
                raise ConnectionError("Lost connection while sending command")
            else:
                raise ConnectionError(f"Error sending command: {e}")
        except Exception as e:
            raise ConnectionError(f"Error sending command: {e}")

class RLDroneController:
    """
    Main RL-based drone controller class.
    Handles training, inference, and model management.
    """
    
    def __init__(self, model_path=None, algorithm='PPO'):
        self.model_path = model_path
        self.algorithm = algorithm
        self.model = None
        self.env = None
        self.client_socket = None
        
        # Initialize model
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one."""
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            if self.algorithm == 'PPO':
                self.model = PPO.load(self.model_path)
            elif self.algorithm == 'DQN':
                self.model = DQN.load(self.model_path)
            elif self.algorithm == 'A2C':
                self.model = A2C.load(self.model_path)
        else:
            print(f"Creating new {self.algorithm} model")
            self.create_new_model()
    
    def create_new_model(self):
        """Create a new RL model."""
        # Create environment (will be replaced with Webots-connected env during training)
        self.env = DroneControlEnv(training_mode=False)
        
        # Create model based on algorithm
        if self.algorithm == 'PPO':
            self.model = PPO(
                "MlpPolicy",  # We'll use MLP for now, can switch to CNN later
                self.env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                tensorboard_log="./tensorboard_logs/"
            )
        elif self.algorithm == 'DQN':
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                tensorboard_log="./tensorboard_logs/"
            )
        elif self.algorithm == 'A2C':
            self.model = A2C(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                tensorboard_log="./tensorboard_logs/"
            )
    
    def train(self, total_timesteps=100000, save_freq=10000):
        """Train the RL model with Webots integration."""
        print(f"Starting training for {total_timesteps} timesteps...")
        print("Waiting for Webots simulation to connect...")
        
        # Start socket server for Webots connection
        hostname = 'localhost'
        port = 8080
        server_address = (hostname, port)
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(server_address)
        server_socket.listen(1)
        server_socket.setblocking(True)  # Blocking for training
        
        print(f"Training server listening on port {port}")
        print("Please start your Webots simulation and connect the drone controller...")
        
        try:
            # Wait for Webots controller to connect
            client_socket, client_address = server_socket.accept()
            print(f"Webots controller connected from {client_address}")
            client_socket.setblocking(True)  # Blocking for training
            
            # Create training environment with Webots connection
            train_env = Monitor(DroneControlEnv(client_socket=client_socket, training_mode=True))
            
            # Create evaluation environment (without Webots connection for faster evaluation)
            eval_env = Monitor(DroneControlEnv(training_mode=False))
            
            # Set up callbacks
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./models/',
                log_path='./logs/',
                eval_freq=save_freq,
                deterministic=True,
                render=False
            )
            
            # Train the model with connection recovery
            self.model.set_env(train_env)
            print("Starting RL training with real Webots data...")
            
            # Training loop with connection recovery
            remaining_timesteps = total_timesteps
            trained_timesteps = 0
            
            while remaining_timesteps > 0:
                try:
                    # Train for a chunk of timesteps
                    chunk_size = min(remaining_timesteps, save_freq)
                    self.model.learn(
                        total_timesteps=chunk_size,
                        callback=eval_callback,
                        progress_bar=True,
                        reset_num_timesteps=False
                    )
                    
                    trained_timesteps += chunk_size
                    remaining_timesteps -= chunk_size
                    
                    print(f"Completed {trained_timesteps}/{total_timesteps} timesteps")
                    
                except ConnectionError as e:
                    print(f"Connection lost: {e}")
                    print("Attempting to reconnect to Webots...")
                    
                    # Close current connection
                    try:
                        client_socket.close()
                    except:
                        pass
                    
                    # Wait for reconnection
                    try:
                        client_socket, client_address = server_socket.accept()
                        print(f"Reconnected to Webots controller from {client_address}")
                        client_socket.setblocking(True)
                        
                        # Update environment with new connection
                        train_env.env.client_socket = client_socket
                        train_env.env.is_ready = False  # Reset ready state
                        train_env.env.waiting_for_ready_signal = True
                        
                        print("Resuming training...")
                        
                    except Exception as reconnect_error:
                        print(f"Failed to reconnect: {reconnect_error}")
                        print("Please restart Webots simulation and press Enter to continue...")
                        input("Press Enter when Webots is ready...")
                        
                        # Try to reconnect again
                        try:
                            client_socket, client_address = server_socket.accept()
                            print(f"Reconnected to Webots controller from {client_address}")
                            client_socket.setblocking(True)
                            train_env.env.client_socket = client_socket
                            train_env.env.is_ready = False  # Reset ready state
                            train_env.env.waiting_for_ready_signal = True
                        except Exception as final_error:
                            print(f"Final reconnection failed: {final_error}")
                            raise final_error
            
            # Save the final model
            if self.model_path:
                self.model.save(self.model_path)
                print(f"Model saved to {self.model_path}")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            # Clean up
            try:
                client_socket.close()
            except:
                pass
            server_socket.close()
            print("Training server closed")
    
    def set_client_socket(self, client_socket):
        """Set the client socket for communication."""
        self.client_socket = client_socket
        if self.env:
            self.env.client_socket = client_socket
    
    def predict_action(self, observation):
        """Predict action for given observation."""
        if self.model is None:
            return 0  # Default to forward action
        
        action, _states = self.model.predict(observation, deterministic=True)
        return action
    
    def process_array(self, array_2d, client_socket):
        """
        Process camera array using RL agent.
        
        Args:
            array_2d: 2D numpy array (64x64) containing the curve data
            client_socket: Socket connection to send commands to the controller
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            # Update environment observation
            if self.env:
                self.env.update_observation(array_2d)
            
            # Set client socket if not already set
            if not self.client_socket:
                self.set_client_socket(client_socket)
            
            # Predict action using RL model
            action = self.predict_action(array_2d)
            
            # Convert action to command array
            command_values = [0] * 9
            command_values[action] = 1
            
            # Send command
            self.send_command(client_socket, command_values)
            
            print(f"RL Agent action: {self.env.command_names[action] if self.env else 'Unknown'}")
            
            return True
            
        except Exception as e:
            print(f"Error in RL processing: {e}")
            # Fallback to forward command
            command_values = [1, 0, 0, 0, 0, 0, 0, 0, 0]
            self.send_command(client_socket, command_values)
            return False
    
    def send_command(self, client_socket, command_values):
        """Send command to controller."""
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
        
        json_string = json.dumps(command_json)
        try:
            client_socket.sendall(json_string.encode('utf-8'))
        except Exception as e:
            print(f"Error sending command: {e}")

def main():
    # Argument parsing from the command line
    parser = argparse.ArgumentParser(description='RL-based Drone Controller')
    parser.add_argument('--mode', choices=['train', 'inference'], default='inference',
                       help='Mode: train the model or run inference')
    parser.add_argument('--model-path', type=str, default='./models/drone_rl_model',
                       help='Path to save/load the RL model')
    parser.add_argument('--algorithm', choices=['PPO', 'DQN', 'A2C'], default='PPO',
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of timesteps for training')
    args = parser.parse_args()
    
    # Create RL controller
    rl_controller = RLDroneController(
        model_path=args.model_path,
        algorithm=args.algorithm
    )
    
    if args.mode == 'train':
        print("Starting training mode...")
        rl_controller.train(total_timesteps=args.timesteps)
        print("Training completed!")
        return
    
    # Inference mode - start socket server
    print("Starting inference mode...")

    # Define HOSTNAME and PORT
    hostname = 'localhost'
    port = 8080
    server_address = (hostname, port)

    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(server_address)
    server_socket.listen(1)
    server_socket.setblocking(False)  # Non-blocking

    print(f"RL Server listening on port {port}")

    try:
        while True:
            try:
                # Accept client connection
                try:
                    client_socket, client_address = server_socket.accept()
                    print(f"Controller connected from {client_address}")
                    client_socket.setblocking(False)  # Non-blocking
                    
                    # Set client socket for RL controller
                    rl_controller.set_client_socket(client_socket)
                    
                    while True:
                        # Receive binary array data from the controller
                        try:
                            response = client_socket.recv(4096)
                            
                            if len(response) == 4096:
                                # Convert binary data to numpy array
                                array_2d = np.frombuffer(response, dtype=np.int8).reshape((64, 64))
                                
                                # Process the array using RL agent
                                rl_controller.process_array(array_2d, client_socket)
                                
                            elif len(response) > 0:
                                print(f"Received {len(response)} bytes (expected 4096)")
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
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)
                
            time.sleep(1)  # Wait before checking for connections again

    except KeyboardInterrupt:
        print("\nShutting down RL server...")
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)

    finally:
        server_socket.close()

if __name__ == "__main__":
    main()