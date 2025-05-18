# REINFORCE Implementation for CartPole

This repository contains an implementation of the REINFORCE algorithm (Monte Carlo Policy Gradient) for solving the CartPole environment from OpenAI Gymnasium.

## Table of Contents

- [Theory](#theory)
- [Implementation](#implementation)
  - [Network Architecture](#network-architecture)
  - [Baseline Variants](#baseline-variants)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Example Commands](#example-commands)
- [Installation](#installation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)
- [References](#references)

## Theory

REINFORCE is a policy gradient algorithm introduced by Williams in 1992. It uses Monte Carlo methods to estimate the gradient of the expected return with respect to the policy parameters. The key insight is that we can use the actual returns obtained during episodes to directly optimize the policy.

The algorithm works as follows:

1. Sample trajectories using the current policy
2. For each step in the trajectory, compute the return (discounted sum of future rewards)
3. Update the policy by increasing the probability of actions that led to higher returns and decreasing the probability of actions that led to lower returns

Mathematically, the REINFORCE update rule is:

$$\boldsymbol \theta_{t+1} \triangleq \boldsymbol \theta_t + \alpha G_t \frac {\nabla \pi (A_t | S_t, \boldsymbol \theta) }{\pi (A_t | S_t, \boldsymbol \theta)}$$

Where:
- $\boldsymbol \theta$ are the policy parameters
- $\alpha$ is the learning rate
- $\pi(A_t|S_t, \boldsymbol \theta)$ is the probability of taking action $A_t$ in state $S_t$ according to the policy
- $G_t$ is the return (discounted sum of rewards) following action $A_t$

The algorithm can be improved by using a baseline to reduce variance. A common baseline is to subtract the average return or use a value function approximator.

## Implementation

### Network Architecture

The implementation uses two neural network architectures:

1. **PolicyNet**: A network that outputs action probabilities given a state
   - Configurable number of hidden layers and width
   - ReLU activation functions
   - Outputs logits for action probabilities

2. **ValueNet**: A network that estimates the state value function (for value baseline)
   - Shares the same architecture as PolicyNet
   - Single output node for the state value

### Baseline Variants

The implementation supports three baseline variants:

1. **No Baseline ('none')**: Standard REINFORCE without any baseline
2. **Standardized Returns ('std')**: Subtracts the mean and divides by the standard deviation of returns
3. **Value Function ('value')**: Uses a separate value network to estimate and subtract the state value

## Usage

### Command Line Arguments

The main script supports various arguments:

```
python -m DLA_Labs.L2_DRL.cartpole.main [OPTIONS]
```

Options:

- `--project STR`: WandB project name (default: "DRL-CartPole")
- `--env STR`: Gym environment to use (default: "CartPole-v1")
- `--seed INT`: Random seed (default: 42)
- `--device STR`: Device to run on (cpu/cuda/mps) (default: "cuda" if available, else "cpu")
- `--episodes INT`: Number of training episodes (default: 1000)
- `--eval_interval INT`: Episodes between evaluations (default: 100)
- `--eval_episodes INT`: Number of episodes for evaluation (default: 10)
- `--baseline STR`: Baseline to use (none/std/value) (default: "none")
- `--gamma FLOAT`: Discount factor (default: 0.99)
- `--lr_policy FLOAT`: Learning rate for policy network (default: 1e-3)
- `--lr_value FLOAT`: Learning rate for value network (default: 1e-3)
- `--hidden_layers INT`: Number of hidden layers (default: 1)
- `--hidden_width INT`: Width of hidden layers (default: 128)
- `--visualize`: Flag to visualize trained agent

### Example Commands

1. **Train with no baseline:**

   ```
   python -m DLA_Labs.L2_DRL.cartpole.main --baseline none --episodes 1000
   ```

2. **Train with standardized returns baseline:**

   ```
   python -m DLA_Labs.L2_DRL.cartpole.main --baseline std --episodes 1000
   ```

3. **Train with value function baseline:**

   ```
   python -m DLA_Labs.L2_DRL.cartpole.main --baseline value --episodes 1000
   ```

4. **Visualize trained agent:**

   ```
   python -m DLA_Labs.L2_DRL.cartpole.main --baseline value --episodes 1000 --visualize
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

The implementation was evaluated on the CartPole-v1 environment, which has a maximum episode length of 500 steps and a maximum possible return of 500.

Below is a comparison of the three baseline variants:

| Baseline | Evaluation Return | Evaluation Length | Episode Return | Episode Length | Policy Loss |
|----------|-------------------|-------------------|----------------|----------------|-------------|
| Value    | 500               | 500               | 99.40          | 498            | 0.01        |
| Std      | 429               | 429               | 95.20          | 413            | -           |
| None     | 281               | 282               | 79.40          | 199            | 25          |

As shown in the results, the value baseline achieves the best performance, reaching the maximum possible return of 500. The standardized returns baseline also performs well but falls short of the maximum. The version without any baseline performs significantly worse.

The value baseline not only achieves better final performance but also demonstrates more stable learning with less variance in returns across episodes.

## Project Structure

```
DLA_Labs/L2_DRL/cartpole/
├── main.py         # Main script for running the application
├── networks.py     # Policy and Value network implementations
├── trainer.py      # REINFORCE trainer implementation
└── utils.py        # Helper functions for training and evaluation
```

## Common Issues

- **Training instability**: REINFORCE can be unstable, especially without a baseline. Using a value function baseline helps stabilize training.
- **High variance**: If experiencing high variance in returns, consider increasing the number of episodes or using a different baseline.
- **Overfitting**: If the policy performs well during training but poorly during evaluation, consider adding regularization or early stopping.
- **Slow convergence**: If convergence is slow, try adjusting the learning rate or discount factor.

## References

1. Bagdanov, A. D., DLA course material (2025)

2. Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. _Machine Learning, 8(3-4), 229-256_.

3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. _MIT Press_.

4. OpenAI Gymnasium CartPole environment: [https://gymnasium.farama.org/environments/classic_control/cart_pole/](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
