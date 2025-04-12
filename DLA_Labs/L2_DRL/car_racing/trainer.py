import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from rich.console import Console
from collections import deque
import random

from DLA_Labs.L2_DRL.cartpole.utils import save_checkpoint
from DLA_Labs.L2_DRL.car_racing.utils import FrameStack, preprocess_frame, select_action, evaluate_agent, save_ppo_checkpoint

console = Console()


class PPOTrainer:
    """
    Trainer class for the Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(self, env, actor, critic, actor_optimizer, critic_optimizer,
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5,
                 ppo_epochs=4, mini_batch_size=64, num_frames=4,
                 target_kl=0.01, device='cpu'):
        """
        Initialize the PPO trainer.

        Args:
            env: The environment to train in
            actor (nn.Module): The actor network
            critic (nn.Module): The critic network
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_ratio (float): PPO clip parameter
            value_coef (float): Value loss coefficient
            entropy_coef (float): Entropy bonus coefficient
            max_grad_norm (float): Maximum gradient norm for clipping
            ppo_epochs (int): Number of PPO epochs per update
            mini_batch_size (int): Mini batch size for updates
            num_frames (int): Number of frames to stack
            target_kl (float): Target KL divergence for early stopping
            device (str): Device to run on ('cpu' or 'cuda')
        """
        self.env = env
        self.actor = actor
        self.critic = critic
        self.device = device
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.num_frames = num_frames
        self.target_kl = target_kl
        self.device = device

        # Initialize frame stacker
        self.frame_stack = FrameStack(num_frames=num_frames)

        # Move networks to device
        self.actor.to(device)
        self.critic.to(device)

    def compute_gae(self, rewards, values, next_value, dones):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): List of rewards
            values (list): List of value estimates
            next_value (float): Value estimate for the next state
            dones (list): List of done flags

        Returns:
            tuple: (advantages, returns)
        """
        advantages = []
        gae = 0
        values = values + [next_value]

        # Calculate in reverse order
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        # Calculate returns as advantages + values
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return advantages, returns

    def collect_rollout(self, rollout_length):
        """
        Collect experience by running the policy in the environment.

        Args:
            rollout_length (int): Number of steps to collect

        Returns:
            tuple: Experience data (observations, actions, log_probs, rewards, values, dones)
        """
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        # Reset environment
        obs, _ = self.env.reset()
        self.frame_stack.reset(obs)
        done = False
        episode_rewards = []

        for _ in range(rollout_length):
            # Get stacked frames
            stacked_frames = self.frame_stack.get_stacked_frames()

            # Convert to tensor for network
            obs_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)

            # Get action and value
            with torch.no_grad():
                action_logits = self.actor(obs_tensor)
                value = self.critic(obs_tensor).squeeze()

            # Create distribution and sample action
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store data
            observations.append(stacked_frames)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            # Update frame stack
            self.frame_stack.add_frame(next_obs)

            # Store rewards and dones
            rewards.append(reward)
            dones.append(float(done))
            episode_rewards.append(reward)

            # Reset environment if done
            if done:
                obs, _ = self.env.reset()
                self.frame_stack.reset(obs)
                console.print(f"Episode finished with total reward: {sum(episode_rewards):.2f}")
                episode_rewards = []

        # Get value for last observation
        stacked_frames = self.frame_stack.get_stacked_frames()
        obs_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.critic(obs_tensor).squeeze().item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)

        return (
            observations,
            actions,
            log_probs,
            rewards,
            values,
            dones,
            advantages,
            returns
        )

    def update_policy(self, observations, actions, old_log_probs, returns, advantages):
        """
        Update policy using PPO.

        Args:
            observations (list): List of observations
            actions (list): List of actions
            old_log_probs (list): List of log probabilities from old policy
            returns (list): List of returns
            advantages (list): List of advantages

        Returns:
            tuple: (policy_loss, value_loss, entropy)
        """
        # Convert to tensors
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset indices
        dataset_size = len(observations)
        indices = np.arange(dataset_size)

        # Start PPO update
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []

        for _ in range(self.ppo_epochs):
            # Shuffle indices
            np.random.shuffle(indices)

            # Create mini-batches
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                # Get mini-batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass through actor
                action_logits = self.actor(batch_obs)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratio and clipped ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(ratio * batch_advantages, clip_adv).mean()

                # Get value predictions and compute value loss
                values_pred = self.critic(batch_obs).squeeze()
                value_loss = F.mse_loss(values_pred, batch_returns)

                # Compute entropy loss
                entropy_loss = -self.entropy_coef * entropy

                # Compute total loss
                total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

                # Update actor and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Calculate approximate KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                    kl_divs.append(kl_div)

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

            # Early stopping based on KL divergence
            mean_kl = np.mean(kl_divs[-dataset_size // self.mini_batch_size:])
            if mean_kl > 1.5 * self.target_kl:
                console.print(f"Early stopping at epoch {_ + 1}/{self.ppo_epochs} due to reaching max KL divergence.")
                break

        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses)

    def train(self, total_timesteps, rollout_length=2048, eval_interval=10000,
              eval_episodes=5, wandb_run=None):
        """
        Train using PPO algorithm.

        Args:
            total_timesteps (int): Total timesteps to train for
            rollout_length (int): Number of steps in each rollout
            eval_interval (int): Number of timesteps between evaluations
            eval_episodes (int): Number of episodes for evaluation
            wandb_run: WandB run object for logging

        Returns:
            dict: Training history
        """
        history = {
            'timesteps': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'eval_returns': [],
            'eval_lengths': [],
            'episode_returns': [],
            'episode_lengths': []
        }

        # For tracking episode stats
        episode_reward = 0
        episode_length = 0

        # For tracking the best model
        best_eval_return = float('-inf')

        # Training loop
        timesteps_done = 0

        console.print(f"[bold blue]Starting training for {total_timesteps} timesteps[/bold blue]")

        while timesteps_done < total_timesteps:
            # Collect rollout
            (observations, actions, old_log_probs, rewards, values, dones, advantages, returns) = self.collect_rollout(
                rollout_length)

            # Update policy
            policy_loss, value_loss, entropy_loss = self.update_policy(
                observations, actions, old_log_probs, returns, advantages
            )

            timesteps_done += rollout_length

            # Track progress
            if timesteps_done % 10000 == 0 or timesteps_done >= total_timesteps:
                console.print(f"[bold blue]Timestep {timesteps_done}/{total_timesteps}[/bold blue]")

            # Store training statistics
            history['timesteps'].append(timesteps_done)
            history['policy_losses'].append(policy_loss)
            history['value_losses'].append(value_loss)
            history['entropy_losses'].append(entropy_loss)

            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    'timesteps': timesteps_done,
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'entropy_loss': entropy_loss
                })

            # Evaluate at intervals
            if timesteps_done % eval_interval == 0 or timesteps_done >= total_timesteps:
                avg_return, avg_length = evaluate_agent(
                    self.env, self.actor, self.critic, self.device,
                    num_episodes=eval_episodes, frame_stack_size=self.num_frames
                )

                history['eval_returns'].append(avg_return)
                history['eval_lengths'].append(avg_length)

                console.print(f"[bold green]Evaluation at timestep {timesteps_done}: "
                              f"Avg Return = {avg_return:.2f}, Avg Length = {avg_length:.2f}[/bold green]")

                # Log evaluation metrics
                if wandb_run:
                    wandb_run.log({
                        'timesteps': timesteps_done,
                        'eval_avg_return': avg_return,
                        'eval_avg_length': avg_length
                    })

                # Save best model
                if avg_return > best_eval_return and wandb_run:
                    best_eval_return = avg_return
                    save_ppo_checkpoint(
                        self.actor, self.critic,
                        self.actor_optimizer, self.critic_optimizer,
                        wandb_run.dir, name="best"
                    )
                    console.print(f"[bold cyan]New best model saved! Return: {best_eval_return:.2f}[/bold cyan]")

        return history
