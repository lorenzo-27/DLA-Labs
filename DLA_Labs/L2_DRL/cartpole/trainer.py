import torch
from rich.console import Console
from DLA_Labs.L2_DRL.cartpole.utils import run_episode, compute_returns, evaluate_agent, save_checkpoint

console = Console()

class REINFORCETrainer:
    """
    Trainer class for the REINFORCE algorithm with various baseline options.
    """

    def __init__(self, env, policy, optimizer, gamma=0.99, baseline='none',
                 value_net=None, value_optimizer=None, value_coef=0.5,
                 device='cpu'):
        """
        Initialize the REINFORCE trainer.

        Args:
            env: The environment to train in.
            policy (nn.Module): The policy network.
            optimizer (torch.optim.Optimizer): Optimizer for the policy.
            gamma (float): Discount factor.
            baseline (str): Type of baseline to use ('none', 'std', 'value').
            value_net (nn.Module, optional): Value network for value baseline.
            value_optimizer (torch.optim.Optimizer, optional): Optimizer for value network.
            value_coef (float): Coefficient for value loss in value baseline.
            device (str): Device to run on ('cpu' or 'cuda').
        """
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.baseline = baseline
        self.value_net = value_net
        self.value_optimizer = value_optimizer
        self.value_coef = value_coef
        self.device = device

        # Check valid baseline
        if baseline not in ['none', 'std', 'value']:
            raise ValueError(f"Unknown baseline '{baseline}'. Choose from 'none', 'std', 'value'.")

        # Check if value network is provided when value baseline is selected
        if baseline == 'value' and (value_net is None or value_optimizer is None):
            raise ValueError("Value network and optimizer must be provided when using value baseline.")

    def train(self, num_episodes, eval_interval=100, eval_episodes=10, wandb_run=None):
        """
        Train the policy using REINFORCE.

        Args:
            num_episodes (int): Number of episodes to train for.
            eval_interval (int): Interval at which to evaluate the policy.
            eval_episodes (int): Number of episodes to use for evaluation.
            wandb_run: WandB run object for logging.

        Returns:
            dict: Dictionary containing training history.
        """
        history = {
            'episode_returns': [],
            'episode_lengths': [],
            'running_avg_return': [],
            'eval_returns': [],
            'eval_lengths': [],
            'policy_losses': []
        }

        if self.baseline == 'value':
            history['value_losses'] = []

        # For tracking best performance
        best_eval_return = float('-inf')
        running_return = 0.0

        for episode in range(1, num_episodes + 1):
            # Progress tracking
            if episode % 100 == 0:
                console.print(f"[bold blue]Episode {episode}/{num_episodes}[/bold blue]")

            # Run episode
            if self.baseline == 'value':
                observations, actions, log_probs, rewards, predicted_values = run_episode(
                    self.env, self.policy, self.value_net)
            else:
                observations, actions, log_probs, rewards = run_episode(
                    self.env, self.policy)

            # Calculate returns
            returns = torch.tensor(compute_returns(rewards, self.gamma), dtype=torch.float32)

            # Update running average
            episode_return = returns[0].item()
            running_return = 0.05 * episode_return + 0.95 * running_return if episode > 1 else episode_return

            # Store episode statistics
            history['episode_returns'].append(episode_return)
            history['episode_lengths'].append(len(rewards))
            history['running_avg_return'].append(running_return)

            # Apply baseline
            if self.baseline == 'none':
                base_returns = returns
            elif self.baseline == 'std':
                # Standardize returns if they have non-zero standard deviation
                if returns.std() > 0:
                    base_returns = (returns - returns.mean()) / returns.std()
                else:
                    base_returns = returns - returns.mean()
            elif self.baseline == 'value':
                # Value function baseline
                base_returns = returns - predicted_values.detach().squeeze()

                # Update value network
                self.value_optimizer.zero_grad()
                value_loss = torch.nn.functional.mse_loss(predicted_values.squeeze(), returns)
                value_loss.backward()
                self.value_optimizer.step()
                history['value_losses'].append(value_loss.item())

            # Update policy
            self.optimizer.zero_grad()
            policy_loss = (-log_probs * base_returns).mean()
            policy_loss.backward()
            self.optimizer.step()

            history['policy_losses'].append(policy_loss.item())

            # Log to wandb
            if wandb_run:
                log_data = {
                    'episode': episode,
                    'episode_return': episode_return,
                    'episode_length': len(rewards),
                    'running_avg_return': running_return,
                    'policy_loss': policy_loss.item()
                }
                if self.baseline == 'value':
                    log_data['value_loss'] = value_loss.item()
                wandb_run.log(log_data)

            # Evaluate at intervals
            if episode % eval_interval == 0:
                avg_return, avg_length = evaluate_agent(self.env, self.policy, eval_episodes)
                history['eval_returns'].append(avg_return)
                history['eval_lengths'].append(avg_length)

                console.print(f"[bold green]Evaluation after episode {episode}: "
                      f"Avg Return = {avg_return:.2f}, Avg Length = {avg_length:.2f}[/bold green]")

                # Log evaluation metrics
                if wandb_run:
                    wandb_run.log({
                        'episode': episode,
                        'eval_avg_return': avg_return,
                        'eval_avg_length': avg_length
                    })

                # Save best model
                if avg_return > best_eval_return and wandb_run:
                    best_eval_return = avg_return
                    save_checkpoint('best', self.policy, self.optimizer, wandb_run.dir)
                    if self.baseline == 'value':
                        save_checkpoint('best_value', self.value_net,
                                        self.value_optimizer, wandb_run.dir)

        return history