import numpy as np
import torch
import gymnasium as gym
import cv2
from collections import deque
from torch.distributions import Categorical


def preprocess_frame(frame, target_size=(84, 84)):
    """
    Preprocess an RGB frame:
    1. Convert to grayscale
    2. Resize to target size
    3. Normalize values to [0,1]

    Args:
        frame (numpy.ndarray): RGB frame from the environment
        target_size (tuple): Target size after resize (width, height)

    Returns:
        numpy.ndarray: Preprocessed grayscale frame
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Resize
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Normalize
    normalized = resized / 255.0

    return normalized


class FrameStack:
    """
    Class for stacking multiple frames together.
    """

    def __init__(self, num_frames=4, frame_size=(84, 84)):
        """
        Initialize frame stacking.

        Args:
            num_frames (int): Number of frames to stack
            frame_size (tuple): Size of each frame (width, height)
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frames = deque(maxlen=num_frames)

    def reset(self, initial_frame):
        """
        Reset with an initial frame.

        Args:
            initial_frame (numpy.ndarray): Initial frame to fill the stack
        """
        processed_frame = preprocess_frame(initial_frame, self.frame_size)
        # Fill the stack with the same initial frame
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(processed_frame)

    def add_frame(self, frame):
        """
        Add a new frame to the stack.

        Args:
            frame (numpy.ndarray): New frame to add
        """
        processed_frame = preprocess_frame(frame, self.frame_size)
        self.frames.append(processed_frame)

    def get_stacked_frames(self):
        """
        Get stacked frames as a numpy array.

        Returns:
            numpy.ndarray: Stacked frames with shape (num_frames, height, width)
        """
        return np.array(self.frames)


def select_action(observations, actor, device, deterministic=False):
    """
    Select an action based on the observations using the actor network.

    Args:
        observations (numpy.ndarray): Stacked observations
        actor (nn.Module): Actor network
        device (str): Device to run on ('cpu' or 'cuda')
        deterministic (bool): If True, select the most probable action

    Returns:
        tuple: (selected action, log probability, action distribution entropy)
    """
    # Convert to tensor
    obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(device)

    # Get action logits
    with torch.no_grad():
        action_logits = actor(obs_tensor)

    # Create categorical distribution
    dist = Categorical(logits=action_logits)

    if deterministic:
        action = torch.argmax(action_logits, dim=1)
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    return action.item(), log_prob.item(), entropy.item()


def evaluate_agent(env, actor, critic, device, num_episodes=5, frame_stack_size=4, render=False):
    """
    Evaluate agent performance over multiple episodes.

    Args:
        env: The environment to evaluate in
        actor (nn.Module): The actor network
        critic (nn.Module): The critic network
        device (str): Device to run on ('cpu' or 'cuda')
        num_episodes (int): Number of episodes to run
        frame_stack_size (int): Number of frames to stack
        render (bool): Whether to render the environment

    Returns:
        tuple: (average_return, average_episode_length)
    """
    returns = []
    episode_lengths = []

    render_mode = "human" if render else None
    if render:
        # Create a separate environment for rendering
        eval_env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    else:
        eval_env = env

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()

        # Initialize frame stack
        frame_stack = FrameStack(num_frames=frame_stack_size)
        frame_stack.reset(obs)

        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Get stacked frames
            stacked_frames = frame_stack.get_stacked_frames()

            # Select action (deterministic for evaluation)
            action, _, _ = select_action(stacked_frames, actor, device, deterministic=True)

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            # Update frame stack
            frame_stack.add_frame(next_obs)

            episode_reward += reward
            episode_length += 1

        returns.append(episode_reward)
        episode_lengths.append(episode_length)

    avg_return = sum(returns) / num_episodes
    avg_length = sum(episode_lengths) / num_episodes

    if render and eval_env != env:
        eval_env.close()

    return avg_return, avg_length


def save_ppo_checkpoint(actor, critic, actor_optimizer, critic_optimizer, directory, name="best"):
    """
    Save actor and critic networks.

    Args:
        actor (nn.Module): Actor network
        critic (nn.Module): Critic network
        actor_optimizer (torch.optim.Optimizer): Actor optimizer
        critic_optimizer (torch.optim.Optimizer): Critic optimizer
        directory (str): Directory to save to
        name (str): Name identifier for the checkpoint
    """
    import os
    os.makedirs(directory, exist_ok=True)

    # Save actor
    torch.save(
        {
            'model_state_dict': actor.state_dict(),
            'optimizer_state_dict': actor_optimizer.state_dict(),
        },
        os.path.join(directory, f'checkpoint-{name}-actor.pt')
    )

    # Save critic
    torch.save(
        {
            'model_state_dict': critic.state_dict(),
            'optimizer_state_dict': critic_optimizer.state_dict(),
        },
        os.path.join(directory, f'checkpoint-{name}-critic.pt')
    )
