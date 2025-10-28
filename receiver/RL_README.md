# Reinforcement Learning Drone Controller

This implementation replaces the rule-based line following with a reinforcement learning approach using Stable Baselines 3.

## Features

- **Custom Gymnasium Environment**: `DroneControlEnv` for drone line following
- **Multiple RL Algorithms**: Support for PPO, DQN, and A2C
- **Model Training & Saving**: Automatic model persistence and loading
- **Command Line Interface**: Easy training and inference modes
- **Tensorboard Integration**: Training progress visualization

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create necessary directories:
```bash
mkdir -p models logs tensorboard_logs
```

## Usage

### Training Mode

Train a new RL model:
```bash
python recvRLDisc.py --mode train --algorithm PPO --timesteps 100000 --model-path ./models/drone_ppo_model
```

Available algorithms:
- `PPO` (Proximal Policy Optimization) - Recommended
- `DQN` (Deep Q-Network)
- `A2C` (Advantage Actor-Critic)

### Inference Mode

Run the trained model:
```bash
python recvRLDisc.py --mode inference --model-path ./models/drone_ppo_model --algorithm PPO
```

### Default Usage

Run with default settings (inference mode with PPO):
```bash
python recvRLDisc.py
```

## Architecture

### DroneControlEnv
- **State Space**: 64x64 grayscale image from drone camera
- **Action Space**: 9 discrete actions (forward, backward, left, right, yaw_increase, yaw_decrease, height_diff_increase, height_diff_decrease, reset_simulation)
- **Reward Function**: Based on line centering, forward progress, and stability

### Reward Function
- **Line Detection**: +10 for detecting line, -10 for losing line
- **Centering**: Higher reward for better line centering (max +10)
- **Progress**: +1 for forward movement
- **Stability**: Bonus for consecutive good steps
- **Correction Penalty**: Small penalty for unnecessary corrections

### Model Management
- Models are automatically saved during training
- Best models are saved based on evaluation performance
- Tensorboard logs are created in `./tensorboard_logs/`

## Training Process

1. **Start Webots Simulation**: Load `worlds/wesDroneRL2.wbt`
2. **Start Training**: Run the training command
3. **Monitor Progress**: Use Tensorboard to view training metrics
4. **Evaluate Model**: The system automatically evaluates during training

## Integration with Webots

The RL controller maintains the same socket communication protocol as the original system:
- Receives 64x64 camera arrays from Webots controller
- Sends JSON commands back to the controller
- Uses port 8080 for communication

## File Structure

```
receiver/
├── recvRLDisc.py          # Main RL controller
├── requirements.txt       # Updated dependencies
├── models/               # Saved RL models
├── logs/                 # Training logs
└── tensorboard_logs/     # Tensorboard logs
```

## Performance Tips

1. **Training**: Start with smaller timesteps (10,000-50,000) for initial testing
2. **Algorithm Selection**: PPO generally works best for this continuous control task
3. **Reward Tuning**: Adjust reward function parameters in `calculate_reward()` method
4. **Model Architecture**: Currently uses MLP policy, can be upgraded to CNN for better image processing

## Troubleshooting

- **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **Model Loading**: Ensure model path exists and contains valid model files
- **Socket Issues**: Check that Webots controller is running and port 8080 is available
- **Training Issues**: Monitor Tensorboard logs for training progress and potential issues
