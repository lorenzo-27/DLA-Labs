import argparse
import gymnasium as gym
import torch
import wandb
from rich.console import Console
import os
import numpy as np

from DLA_Labs.L2_DRL.car_racing.networks import ActorCNN, CriticCNN
from DLA_Labs.L2_DRL.car_racing.trainer import PPOTrainer
from DLA_Labs.L2_DRL.car_racing.utils import evaluate_agent, preprocess_frame, FrameStack


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PPO implementation for CarRacing.')

    # General parameters
    parser.add_argument('--project', type=str, default='DRL-CarRacing', help='WandB project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cpu/cuda/mps)')

    # Environment parameters
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames to stack')

    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps to train for')
    parser.add_argument('--rollout_length', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--eval_interval', type=int, default=10000, help='Timesteps between evaluations')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of episodes for evaluation')

    # PPO parameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip parameter')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO epochs per update')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='PPO mini-batch size')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy bonus coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--target_kl', type=float, default=0.01, help='Target KL divergence for early stopping')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for networks')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Learning rate for actor')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for critic')

    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Visualize trained agent')
    parser.add_argument('--visualize_untrained', action='store_true', help='Visualize untrained agent')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')

    return parser.parse_args()


def main():
    """Main function."""
    console = Console()

    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = gym.make("CarRacing-v3", continuous=False)

    # Get action dimension
    action_dim = env.action_space.n
    input_channels = args.num_frames  # Number of stacked frames

    console.print(f"[bold blue]Environment: CarRacing-v3[/bold blue]")
    console.print(f"[bold blue]Action Space: {env.action_space}[/bold blue]")
    console.print(f"[bold blue]Observation Space: {env.observation_space}[/bold blue]")
    console.print(f"[bold blue]Input Channels: {input_channels}[/bold blue]")
    console.print(f"[bold blue]Action Dimension: {action_dim}[/bold blue]")
    console.print(f"[bold blue]Device: {args.device}[/bold blue]")

    # Create networks
    actor = ActorCNN(input_channels=input_channels, action_dim=action_dim, hidden_dim=args.hidden_dim)
    critic = CriticCNN(input_channels=input_channels, hidden_dim=args.hidden_dim)

    # Move the networks to the correct device
    actor = actor.to(args.device)
    critic = critic.to(args.device)

    # Setup optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # Load model if specified
    if args.load_model:
        console.print(f"[bold yellow]Loading model from {args.load_model}[/bold yellow]")
        checkpoint_actor = torch.load(f"{args.load_model}-actor.pt")
        checkpoint_critic = torch.load(f"{args.load_model}-critic.pt")

        actor.load_state_dict(checkpoint_actor['model_state_dict'])
        critic.load_state_dict(checkpoint_critic['model_state_dict'])

        actor_optimizer.load_state_dict(checkpoint_actor['optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint_critic['optimizer_state_dict'])

    # Visualize untrained agent if requested
    if args.visualize_untrained:
        console.print("[bold magenta]Visualizing untrained agent...[/bold magenta]")
        env_render = gym.make("CarRacing-v3", continuous=False, render_mode='human')
        evaluate_agent(env_render, actor, critic, args.device, num_episodes=3,
                       frame_stack_size=args.num_frames, render=True)
        env_render.close()
        return

    # Visualize trained agent if requested
    if args.visualize:
        console.print("[bold magenta]Visualizing trained agent...[/bold magenta]")
        env_render = gym.make("CarRacing-v3", continuous=False, render_mode='human')
        avg_return, avg_length = evaluate_agent(env_render, actor, critic, args.device,
                                                num_episodes=5, frame_stack_size=args.num_frames, render=True)
        console.print(f"[bold green]Average Return: {avg_return:.2f}, Average Length: {avg_length:.2f}[/bold green]")
        env_render.close()
        return

    # Initialize wandb
    wandb_run = wandb.init(
        project=args.project,
        config={
            'environment': 'CarRacing-v3',
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_ratio': args.clip_ratio,
            'ppo_epochs': args.ppo_epochs,
            'mini_batch_size': args.mini_batch_size,
            'value_coef': args.value_coef,
            'entropy_coef': args.entropy_coef,
            'max_grad_norm': args.max_grad_norm,
            'target_kl': args.target_kl,
            'lr_actor': args.lr_actor,
            'lr_critic': args.lr_critic,
            'hidden_dim': args.hidden_dim,
            'num_frames': args.num_frames,
            'total_timesteps': args.total_timesteps,
            'rollout_length': args.rollout_length,
            'seed': args.seed
        }
    )

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        num_frames=args.num_frames,
        target_kl=args.target_kl,
        device=args.device
    )

    # Train the agent
    history = trainer.train(
        total_timesteps=args.total_timesteps,
        rollout_length=args.rollout_length,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        wandb_run=wandb_run
    )

    # Visualize trained agent after training
    console.print("[bold magenta]Training complete! Visualizing trained agent...[/bold magenta]")
    env_render = gym.make("CarRacing-v3", continuous=False, render_mode='human')
    evaluate_agent(env_render, actor, critic, args.device, num_episodes=3,
                   frame_stack_size=args.num_frames, render=True)
    env_render.close()

    # Clean up
    env.close()
    wandb_run.finish()

    return history


if __name__ == "__main__":
    main()