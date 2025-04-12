import argparse
import gymnasium as gym
import torch
import wandb
from rich.console import Console

from DLA_Labs.L2_DRL.cartpole.networks import PolicyNet, ValueNet
from DLA_Labs.L2_DRL.cartpole.trainer import REINFORCETrainer
from DLA_Labs.L2_DRL.cartpole.utils import evaluate_agent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='REINFORCE implementation for CartPole.')

    # General parameters
    parser.add_argument('--project', type=str, default='DRL-CartPole', help='WandB project name')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cpu/cuda/mps)')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval_interval', type=int, default=100, help='Episodes between evaluations')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')

    # Algorithm parameters
    parser.add_argument('--baseline', type=str, default='none',
                        choices=['none', 'std', 'value'],
                        help='Baseline to use (none/std/value)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr_policy', type=float, default=1e-3, help='Learning rate for policy network')
    parser.add_argument('--lr_value', type=float, default=1e-3, help='Learning rate for value network')

    # Network parameters
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--hidden_width', type=int, default=128, help='Width of hidden layers')

    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Visualize trained agent')

    return parser.parse_args()


def main():
    """Main function."""
    console = Console()

    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)

    # Initialize wandb
    wandb_run = wandb.init(
        project=args.project,
        config={
            'environment': args.env,
            'baseline': args.baseline,
            'gamma': args.gamma,
            'lr_policy': args.lr_policy,
            'lr_value': args.lr_value if args.baseline == 'value' else None,
            'hidden_layers': args.hidden_layers,
            'hidden_width': args.hidden_width,
            'episodes': args.episodes,
            'eval_interval': args.eval_interval,
            'eval_episodes': args.eval_episodes,
            'seed': args.seed
        }
    )

    # Create environments
    env = gym.make(args.env)
    env.reset(seed=args.seed)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create the policy network
    policy = PolicyNet(
        input_dim=state_dim,
        output_dim=action_dim,
        n_hidden=args.hidden_layers,
        width=args.hidden_width
    )

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)

    # Create value network if using value baseline
    value_net = None
    value_optimizer = None
    if args.baseline == 'value':
        value_net = ValueNet(
            input_dim=state_dim,
            n_hidden=args.hidden_layers,
            width=args.hidden_width
        )
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr_value)

    # Create trainer
    trainer = REINFORCETrainer(
        env=env,
        policy=policy,
        optimizer=policy_optimizer,
        gamma=args.gamma,
        baseline=args.baseline,
        value_net=value_net,
        value_optimizer=value_optimizer,
        device=args.device
    )

    # Train the agent
    history = trainer.train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        wandb_run=wandb_run
    )

    # Visualize if requested
    if args.visualize:
        console.print(f"[bold magenta]Visualizing trained agent...[/bold magenta]")
        env_render = gym.make(args.env, render_mode='human')
        for _ in range(5):  # Show 5 episodes
            evaluate_agent(env_render, policy, num_episodes=1)
        env_render.close()

    # Clean up
    env.close()
    wandb_run.finish()

    return history


if __name__ == "__main__":
    main()