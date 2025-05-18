# PPO Implementation for Car Racing

This repository contains an implementation of the Proximal Policy Optimization (PPO) algorithm for solving the CarRacing-v3 environment from OpenAI Gymnasium.

## Table of Contents

- [Theory](#theory)
- [Implementation](#implementation)
  - [Network Architecture](#network-architecture)
  - [PPO Components](#ppo-components)
  - [Frame Preprocessing](#frame-preprocessing)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Installation](#installation)
- [Results](#results)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)
- [References](#references)

## Theory

Proximal Policy Optimization (PPO) is a policy gradient algorithm introduced by Schulman et al. in 2017. PPO addresses the issue of large policy updates in traditional policy gradient methods by limiting the step size to maintain proximity to the previous policy.

The algorithm works as follows:

1. Collect trajectories using the current policy
2. For each step in the trajectory, compute Generalized Advantage Estimation (GAE)
3. Update the policy multiple times using a clipped objective function that prevents too large updates

Mathematically, the PPO clipped objective function is:

$$L^{CLIP} (\theta) = \hat{\mathbb{E}}t [ \min(r_t(\theta) \hat{A}t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}t) ]$$

Where:
- $r_t(\theta) = \frac{\pi\theta(a_t | s_t)}{\pi{\theta{old}}(a_t | s_t)}$ is the probability ratio between the new and old policies
- $\hat{A}_t$ is the estimated advantage at time $t$
- $\epsilon$ is the clipping parameter (typically 0.1 or 0.2)

PPO also uses Generalized Advantage Estimation (GAE) to compute advantages:

$$\hat{A} _t = \delta_t + (\gamma \lambda) \delta_{t+1} + ... + (\gamma \lambda)^{T-t+1} \delta_{T-1}$$

Where:
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error
- $\gamma$ is the discount factor
- $\lambda$ is the GAE parameter that controls the bias-variance tradeoff

## Implementation

### Network Architecture

The implementation uses two convolutional neural networks:

1. **ActorCNN**: A network that outputs action probabilities given an image state
   - Uses a CNN feature extractor followed by fully connected layers
   - Outputs logits for action probabilities (discrete action space for CarRacing-v3)

2. **CriticCNN**: A network that estimates the state value function
   - Shares the same CNN architecture for feature extraction
   - Outputs a scalar value representing the state value

#### CNN Feature Extractor

```
CNNFeatureExtractor:
- Conv2d(input_channels=4, output_channels=32, kernel_size=8, stride=4)
- ReLU
- Conv2d(32, 64, kernel_size=4, stride=2)
- ReLU
- Conv2d(64, 64, kernel_size=3, stride=1)
- ReLU
- Flatten to 64 * 7 * 7 = 3136 features
```

#### Actor Network

```
ActorCNN:
- CNNFeatureExtractor
- Linear(3136, hidden_dim=256)
- ReLU
- Linear(256, action_dim=5)
```

#### Critic Network

```
CriticCNN:
- CNNFeatureExtractor
- Linear(3136, hidden_dim=256)
- ReLU
- Linear(256, 1)
```

### PPO Components

The PPO implementation includes several key components:

1. **Experience Collection**:
   - Samples trajectories using the current policy
   - Stacks multiple frames together to capture temporal information
   - Computes rewards and stores transitions

2. **Advantage Estimation**:
   - Uses Generalized Advantage Estimation (GAE)
   - Normalizes advantages to improve stability

3. **Policy Update**:
   - Updates the policy using mini-batch optimization
   - Uses a clipped surrogate objective to prevent too large policy updates
   - Adds an entropy bonus to encourage exploration
   - Uses early stopping based on KL divergence to prevent destructive updates

4. **Value Function Update**:
   - Updates the value function using mean squared error loss
   - Uses a separate value coefficient to balance policy and value losses

### Frame Preprocessing

The environment provides RGB images of size (96, 96, 3), which are preprocessed before being fed to the networks:

1. **Color Conversion**: Convert RGB to grayscale to reduce dimensionality
2. **Resizing**: Resize frames to 84x84 pixels
3. **Normalization**: Normalize pixel values to [0, 1]
4. **Frame Stacking**: Stack multiple consecutive frames (default: 4) to capture temporal information

Frame stacking is implemented using a queue data structure:

```python
class FrameStack:
    def __init__(self, num_frames=4, frame_size=(84, 84)):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frames = deque(maxlen=num_frames)
```

This allows the agent to perceive movement and velocity, which are crucial for solving the CarRacing environment.

## Usage

### Command Line Arguments

The main script supports various arguments:

```
python -m DLA_Labs.L2_DRL.car_racing.main [OPTIONS]
```

Options:

- `--project STR`: WandB project name (default: "DRL-CarRacing")
- `--seed INT`: Random seed (default: 42)
- `--device STR`: Device to run on (cpu/cuda/mps) (default: "cuda" if available, else "cpu")
- `--num_frames INT`: Number of frames to stack (default: 4)
- `--total_timesteps INT`: Total timesteps to train for (default: 1000000)
- `--rollout_length INT`: Steps per rollout (default: 2048)
- `--eval_interval INT`: Timesteps between evaluations (default: 10000)
- `--eval_episodes INT`: Number of episodes for evaluation (default: 5)
- `--gamma FLOAT`: Discount factor (default: 0.99)
- `--gae_lambda FLOAT`: GAE lambda parameter (default: 0.95)
- `--clip_ratio FLOAT`: PPO clip parameter (default: 0.2)
- `--ppo_epochs INT`: Number of PPO epochs per update (default: 4)
- `--mini_batch_size INT`: PPO mini-batch size (default: 64)
- `--value_coef FLOAT`: Value loss coefficient (default: 0.5)
- `--entropy_coef FLOAT`: Entropy bonus coefficient (default: 0.01)
- `--max_grad_norm FLOAT`: Maximum gradient norm (default: 0.5)
- `--target_kl FLOAT`: Target KL divergence for early stopping (default: 0.01)
- `--hidden_dim INT`: Hidden dimension for networks (default: 256)
- `--lr_actor FLOAT`: Learning rate for actor (default: 3e-4)
- `--lr_critic FLOAT`: Learning rate for critic (default: 1e-3)
- `--visualize`: Flag to visualize trained agent
- `--visualize_untrained`: Flag to visualize untrained agent
- `--load_model STR`: Path to load model from

### Example Commands

1. **Train from scratch:**

   ```
   python -m DLA_Labs.L2_DRL.car_racing.main --total_timesteps 1000000
   ```

2. **Train with custom hyperparameters:**

   ```
   python -m DLA_Labs.L2_DRL.car_racing.main --gamma 0.99 --gae_lambda 0.95 --clip_ratio 0.2 --entropy_coef 0.01
   ```

3. **Visualize trained agent:**

   ```
   python -m DLA_Labs.L2_DRL.car_racing.main --visualize --load_model path/to/checkpoint
   ```

4. **Visualize untrained agent:**

   ```
   python -m DLA_Labs.L2_DRL.car_racing.main --visualize_untrained
   ```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/username/DLA-Labs.git
   cd DLA-Labs
   ```

2. Install the dependencies with pip or conda:

   ```
   pip install -r requirements.txt
   ```
   
   or
   
   ```
   conda install --file requirements.txt
   ```

3. Run the setup to install the package in development mode:

   ```
   pip install -e .
   ```

## Results

The implementation was evaluated on the CarRacing-v3 environment, which provides a score based on distance traveled, staying on the track, and completion time.

Our implementation achieved a high median score of **884 ± 28** over 100 consecutive runs, which would be a world record score for CarRacing-v3 (especially using the discrete space) according to the OpenAI Gym Leaderboard (https://github.com/openai/gym/wiki/Leaderboard). However, the leaderboard has been closed, so this result was not officially submitted.

The performance was achieved with the following hyperparameters:
- Frame stacking: 4 frames
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Clip ratio: 0.2
- Hidden dimension: 256
- Actor learning rate: 3e-4
- Critic learning rate: 1e-3
- Entropy coefficient: 0.01
- Value coefficient: 0.5
- PPO epochs: 4
- Mini-batch size: 64

Training curves showed stable improvement over time, with the agent learning to:
1. Stay on the track consistently
2. Take optimal racing lines around curves
3. Maintain high speed while avoiding penalties
4. "Drift" in some curves

## Testing

The repository includes a testing script (`test_car_racing.py`) for evaluating the trained agent. This script allows you to:

1. Load a previously trained model checkpoint
2. Run the agent for a specified number of test episodes
3. Calculate average performance metrics (return and episode length)
4. Optionally visualize the agent's performance

### Testing Command Line Arguments

```
python -m DLA_Labs.L2_DRL.car_racing.test_car_racing [OPTIONS]
```

Options:
- `--model_path STR`: Path to the saved model checkpoint (required)
- `--num_episodes INT`: Number of episodes to test (default: 10)
- `--num_frames INT`: Number of frames to stack (default: 4)
- `--hidden_dim INT`: Hidden dimension for networks (default: 256)
- `--seed INT`: Random seed (default: 42)
- `--device STR`: Device to run on (default: "cuda" if available, else "cpu")
- `--render`: Flag to render the environment for visualization

### Testing Example

```
python -m DLA_Labs.L2_DRL.car_racing.test_car_racing --model_path path/to/checkpoint --num_episodes 20 --render
```

This will load the model from the specified checkpoint, run it for 20 episodes, and render the environment to visualize the agent's performance.

## Project Structure

```
DLA_Labs/L2_DRL/car_racing/
├── main.py         # Main script for running the application
├── networks.py     # Actor and Critic network implementations
├── trainer.py      # PPO trainer implementation
├── utils.py        # Helper functions for preprocessing, evaluation, etc.
└── test_car_racing.py  # Script for testing trained agents
```

## Common Issues

- **High variance in returns**: PPO can show high variance in returns during training. Increasing the number of frames stacked or adjusting the GAE lambda parameter can help stabilize training.

- **Unstable learning**: If learning is unstable, try reducing the learning rate, increasing the number of update epochs, or adjusting the clipping parameter.

- **Premature convergence**: If the agent converges to a suboptimal policy, try increasing the entropy coefficient to encourage more exploration.

- **Computational requirements**: Training on CarRacing-v3 can be computationally intensive due to image processing. Consider using a GPU for faster training.

- **Overtraining**: Be cautious of overtraining, which can lead to overfitting to specific tracks. Regular evaluation on different seeds can help detect this issue.

## References

1. Bagdanov, A. D., DLA course material (2025)

2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

4. OpenAI Gymnasium CarRacing environment: [https://gymnasium.farama.org/environments/box2d/car_racing/](https://gymnasium.farama.org/environments/box2d/car_racing/)

5. OpenAI Gym Leaderboard (archived): [https://github.com/openai/gym/wiki/Leaderboard](https://github.com/openai/gym/wiki/Leaderboard)
