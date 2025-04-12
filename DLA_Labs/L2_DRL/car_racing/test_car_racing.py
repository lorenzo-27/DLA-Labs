import argparse
import gymnasium as gym
import torch
import numpy as np
from rich.console import Console

from DLA_Labs.L2_DRL.car_racing.networks import ActorCNN, CriticCNN
from DLA_Labs.L2_DRL.car_racing.utils import evaluate_agent, FrameStack


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test PPO agent for CarRacing.')

    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to test')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for networks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu/cuda)')
    parser.add_argument('--render', action='store_true', help='Render the environment')

    return parser.parse_args()


def main():
    """Main function for testing the trained PPO agent."""
    console = Console()
    args = parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make("CarRacing-v2", continuous=False, render_mode=render_mode)

    # Get action dimension
    action_dim = env.action_space.n
    input_channels = args.num_frames

    console.print(f"[bold blue]Testing PPO agent on CarRacing-v2[/bold blue]")
    console.print(f"[bold blue]Model path: {args.model_path}[/bold blue]")
    console.print(f"[bold blue]Number of episodes: {args.num_episodes}[/bold blue]")

    # Create networks
    actor = ActorCNN(input_channels=input_channels, action_dim=action_dim, hidden_dim=args.hidden_dim)
    critic = CriticCNN(input_channels=input_channels, hidden_dim=args.hidden_dim)

    # Load model
    try:
        checkpoint_actor = torch.load(f"{args.model_path}-actor.pt", map_location=args.device)
        checkpoint_critic = torch.load(f"{args.model_path}-critic.pt", map_location=args.device)

        actor.load_state_dict(checkpoint_actor['model_state_dict'])
        critic.load_state_dict(checkpoint_critic['model_state_dict'])

        console.print("[bold green]Successfully loaded model![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    # Move networks to device
    actor.to(args.device)
    critic.to(args.device)

    # Test the agent
    console.print("[bold yellow]Starting evaluation...[/bold yellow]")
    avg_return, avg_length = evaluate_agent(
        env, actor, critic, args.device,
        num_episodes=args.num_episodes,
        frame_stack_size=args.num_frames,
        render=args.render
    )

    console.print(f"[bold green]Evaluation results:[/bold green]")
    console.print(f"[bold green]Average Return: {avg_return:.2f}[/bold green]")
    console.print(f"[bold green]Average Episode Length: {avg_length:.2f}[/bold green]")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()