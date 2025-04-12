import numpy as np
import torch
from torch.distributions import Categorical
import os


def select_action(obs, policy):
    """
    Given an observation and policy, sample an action from pi(a | obs).

    Args:
        obs (torch.Tensor): The observation from the environment
        policy (nn.Module): The policy network

    Returns:
        tuple: The selected action and the log probability of that action
    """
    dist = Categorical(logits=policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))


def compute_returns(rewards, gamma):
    """
    Compute the discounted total reward for a sequence of rewards.

    Args:
        rewards (list or numpy array): List or array of rewards.
        gamma (float): Discount factor.

    Returns:
        numpy array: Discounted total returns.
    """
    # Calculate discounted returns in reverse order, then flip the array
    rewards = np.array(rewards)
    discounted_rewards = []
    R = 0

    # Calculate in reverse order
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    return np.array(discounted_rewards)


def run_episode(env, policy, value_net=None, max_steps=500, render=False):
    """
    Run an episode in the given environment using the specified policy.

    Args:
        env: The environment to run the episode in.
        policy (nn.Module): The policy used to select actions.
        value_net (nn.Module, optional): Value network for collecting state values.
        max_steps (int): Maximum number of steps in the episode.
        render (bool): Whether to render the environment.

    Returns:
        tuple: A tuple containing:
            - observations (list): List of observations.
            - actions (list): List of actions taken.
            - log_probs (torch.Tensor): Log probabilities of actions.
            - rewards (list): List of rewards received.
            - state_values (list, optional): Values estimated by value network if provided.
    """
    observations = []
    actions = []
    log_probs = []
    rewards = []
    state_values = []

    # Reset the environment and start the episode
    obs, info = env.reset()
    done = False

    for _ in range(max_steps):
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # Get action from policy
        action, log_prob = select_action(obs_tensor, policy)
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)

        # Collect value estimate if value network is provided
        if value_net is not None:
            state_value = value_net(obs_tensor)
            state_values.append(state_value)

        # Execute action in environment
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

        if done:
            break

    # Prepare return values
    result = [
        observations,
        actions,
        torch.cat(log_probs),
        rewards
    ]

    if value_net is not None:
        result.append(torch.cat(state_values))

    return tuple(result)


def evaluate_agent(env, policy, num_episodes=10):
    """
    Evaluate a policy over multiple episodes without training.

    Args:
        env: The environment to evaluate in
        policy (nn.Module): The policy to evaluate
        num_episodes (int): Number of episodes to run

    Returns:
        tuple: (average_return, average_episode_length)
    """
    total_returns = []
    episode_lengths = []

    for _ in range(num_episodes):
        _, _, _, rewards = run_episode(env, policy)
        total_returns.append(sum(rewards))
        episode_lengths.append(len(rewards))

    avg_return = sum(total_returns) / num_episodes
    avg_length = sum(episode_lengths) / num_episodes

    return avg_return, avg_length


def save_checkpoint(name, model, optimizer, directory):
    """
    Save model checkpoint.

    Args:
        name (str): Name identifier for the checkpoint
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        directory (str): Directory to save to
    """
    os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        os.path.join(directory, f'checkpoint-{name}.pt')
    )


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a checkpoint.

    Args:
        filepath (str): Path to the checkpoint file
        model (nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into

    Returns:
        nn.Module: The model with loaded state
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model
