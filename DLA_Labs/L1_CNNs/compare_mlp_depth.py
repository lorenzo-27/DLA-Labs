import argparse
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import copy

from DLA_Labs.L1_CNNs.dataset import load_mnist
from DLA_Labs.L1_CNNs.models.m_mlp import MLP, ResidualMLP


def create_configs(depths, residual_options, epochs):
    """Create configs for different depths and residual options."""
    configs = []

    base_config = {
        "model": {
            "name": "mlp",
            "input_shape": [1, 28, 28],
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 128,
            "num_epochs": epochs,
            "weight_decay": 0.0001,
            "checkpoint_dir": "checkpoints/",
            "log_every": 10,
            "save_every": 5,
            "early_stopping_patience": 5,
            "optimizer": "adam",
            "scheduler": "reduce_lr",
        },
        "dataset": {
            "name": "mnist",
            "data_dir": "data/",
            "val_size": 5000,
        }
    }

    for depth in depths:
        # Generate layer sizes based on depth
        # Start with 128, then halve until 32, then keep adding 32s for deeper networks
        layer_sizes = [128]
        for i in range(1, depth):
            if i == 1:
                layer_sizes.append(64)
            elif i == 2:
                layer_sizes.append(32)
            else:
                layer_sizes.append(32)
        layer_sizes.append(10)  # Output layer

        for residual in residual_options:
            # Deep copy to ensure no overwriting of configs
            config = copy.deepcopy(base_config)
            config["model"]["layer_sizes"] = layer_sizes
            config["model"]["residual"] = residual
            configs.append(config)

    return configs


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Save gradients for the first batch for analysis (optional)
    first_batch = next(iter(dataloader))
    first_inputs, first_targets = first_batch[0].to(device), first_batch[1].to(device)
    optimizer.zero_grad()
    outputs = model(first_inputs)
    loss = criterion(outputs, first_targets)
    loss.backward()

    # Calculate gradient magnitudes for all parameters
    grad_magnitudes = []
    for param in model.parameters():
        if param.grad is not None:
            grad_magnitudes.append(torch.norm(param.grad).item())

    avg_grad_magnitude = sum(grad_magnitudes) / len(grad_magnitudes) if grad_magnitudes else 0

    return running_loss / len(dataloader), 100.0 * correct / total, avg_grad_magnitude


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on the given dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, test_loader, config):
    """Train a model and evaluate it on test set."""
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping_patience = config["training"]["early_stopping_patience"]

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # For tracking results
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    grad_magnitudes = []

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Train
        train_loss, train_acc, avg_grad_magnitude = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Grad Magnitude: {avg_grad_magnitude:.6f}")

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}")

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        grad_magnitudes.append(avg_grad_magnitude)

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"New best model with accuracy: {best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs")
            break

    # Load best model for testing
    model.load_state_dict(best_model_state)

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_acc": test_acc,
        "grad_magnitudes": grad_magnitudes,
        "epochs_completed": len(train_losses)
    }


def analyze_gradients(model, train_loader, device):
    """Analyze gradient magnitudes on a single batch."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Get a single batch
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward and backward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Analyze gradients
    grad_info = {}

    # For MLP and ResidualMLP, we need different approaches to extract layer info
    if hasattr(model, 'model'):  # Standard MLP
        for i, layer in enumerate(model.model):
            if hasattr(layer, 'layers') and hasattr(layer.layers[0], 'weight'):
                weight = layer.layers[0].weight
                if weight.grad is not None:
                    grad_norm = torch.norm(weight.grad).item()
                    grad_info[f"Layer {i}"] = grad_norm
            elif hasattr(layer, 'weight'):
                weight = layer.weight
                if weight.grad is not None:
                    grad_norm = torch.norm(weight.grad).item()
                    grad_info[f"Layer {i}"] = grad_norm

    elif hasattr(model, 'first_layer') and hasattr(model, 'res_blocks'):  # ResidualMLP
        # First layer
        if hasattr(model.first_layer.layers[0], 'weight'):
            weight = model.first_layer.layers[0].weight
            if weight.grad is not None:
                grad_info["First Layer"] = torch.norm(weight.grad).item()

        # Residual blocks
        for i, block in enumerate(model.res_blocks):
            if hasattr(block.block[0].layers[0], 'weight'):
                weight = block.block[0].layers[0].weight
                if weight.grad is not None:
                    grad_info[f"Res Block {i} (first)"] = torch.norm(weight.grad).item()

            if hasattr(block.block[1].layers[0], 'weight'):
                weight = block.block[1].layers[0].weight
                if weight.grad is not None:
                    grad_info[f"Res Block {i} (second)"] = torch.norm(weight.grad).item()

        # Output layer
        if hasattr(model.output_layer, 'weight'):
            weight = model.output_layer.weight
            if weight.grad is not None:
                grad_info["Output Layer"] = torch.norm(weight.grad).item()

    return grad_info


def run_experiments(depths, residual_options, epochs):
    """Run experiments for different network depths and residual options."""
    configs = create_configs(depths, residual_options, epochs)
    results = {}

    os.makedirs("mlp_results", exist_ok=True)

    # Load data once
    batch_size = 128
    train_loader, val_loader, test_loader = load_mnist(batch_size)

    for config in configs:
        depth = len(config["model"]["layer_sizes"]) - 1  # -1 for output layer
        residual = config["model"]["residual"]
        model_key = f"depth_{depth}_residual_{residual}"

        print(f"\n{'=' * 50}")
        print(f"Training model with depth {depth}, residual={residual}")
        print(f"{'=' * 50}\n")

        # Create the model
        if residual:
            model = ResidualMLP(
                layer_sizes=config["model"]["layer_sizes"],
                input_shape=config["model"]["input_shape"]
            )
        else:
            model = MLP(
                layer_sizes=config["model"]["layer_sizes"],
                input_shape=config["model"]["input_shape"]
            )

        # Train the model
        model_results = train_model(model, train_loader, val_loader, test_loader, config)
        results[model_key] = model_results

        # Gradient analysis (single batch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.zero_grad()  # Clear previous gradients
        grad_info = analyze_gradients(model, train_loader, device)
        results[model_key]["grad_analysis"] = grad_info

        print(f"\nGradient analysis for {model_key}:")
        for layer, magnitude in grad_info.items():
            print(f"{layer}: {magnitude:.6f}")

        # Save intermediate results
        save_results(results, "mlp_results/intermediate_results.pkl")

    return results


def save_results(results, filename):
    """Save results to a file."""
    import pickle
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")


def load_results(filename):
    """Load results from a file."""
    import pickle
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_results(results, depths, residual_options):
    """Plot training curves and test accuracies."""
    # Create results directory
    os.makedirs("mlp_results", exist_ok=True)

    # Plot training and validation losses
    plt.figure(figsize=(12, 8))
    for depth in depths:
        for residual in residual_options:
            model_key = f"depth_{depth}_residual_{residual}"
            if model_key in results:
                label = f"Depth {depth}, {'Residual' if residual else 'Non-Residual'}"
                linestyle = '-' if residual else '--'
                plt.plot(results[model_key]["val_losses"], linestyle=linestyle, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Epoch for Different Network Depths')
    plt.legend()
    plt.grid(True)
    plt.savefig("mlp_results/validation_losses.png")

    # Plot test accuracies
    plt.figure(figsize=(12, 8))
    residual_accs = []
    non_residual_accs = []

    for depth in depths:
        for residual in residual_options:
            model_key = f"depth_{depth}_residual_{residual}"
            if model_key in results:
                if residual:
                    residual_accs.append((depth, results[model_key]["test_acc"]))
                else:
                    non_residual_accs.append((depth, results[model_key]["test_acc"]))

    # Sort by depth
    residual_accs.sort()
    non_residual_accs.sort()

    residual_depths = [d for d, _ in residual_accs]
    residual_acc_values = [a for _, a in residual_accs]

    non_residual_depths = [d for d, _ in non_residual_accs]
    non_residual_acc_values = [a for _, a in non_residual_accs]

    plt.plot(residual_depths, residual_acc_values, 'o-', label='Residual')
    plt.plot(non_residual_depths, non_residual_acc_values, 's--', label='Non-Residual')
    plt.xlabel('Network Depth')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs. Network Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig("mlp_results/test_accuracies.png")

    # Plot gradient magnitudes across network depth
    plt.figure(figsize=(12, 8))
    for depth in depths:
        for residual in residual_options:
            model_key = f"depth_{depth}_residual_{residual}"
            if model_key in results:
                label = f"Depth {depth}, {'Residual' if residual else 'Non-Residual'}"
                linestyle = '-' if residual else '--'
                plt.plot(results[model_key]["grad_magnitudes"], linestyle=linestyle, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Gradient Magnitudes vs. Epoch for Different Network Depths')
    plt.legend()
    plt.grid(True)
    plt.savefig("mlp_results/gradient_magnitudes.png")

    # Create a bar plot for gradient analysis of the deepest networks
    max_depth = max(depths)
    res_key = f"depth_{max_depth}_residual_True"
    non_res_key = f"depth_{max_depth}_residual_False"

    if res_key in results and non_res_key in results:
        plt.figure(figsize=(14, 8))

        # Residual network gradients
        res_grad_data = results[res_key]["grad_analysis"]
        layers = list(res_grad_data.keys())
        magnitudes = list(res_grad_data.values())

        x = np.arange(len(layers))
        width = 0.35

        plt.bar(x - width / 2, magnitudes, width, label=f'Residual (Depth {max_depth})')

        # Non-residual network gradients
        if non_res_key in results:
            non_res_grad_data = results[non_res_key]["grad_analysis"]
            non_res_layers = list(non_res_grad_data.keys())
            non_res_magnitudes = list(non_res_grad_data.values())

            # If different number of layers, adjust
            min_len = min(len(layers), len(non_res_layers))
            plt.bar(x[:min_len] + width / 2, non_res_magnitudes[:min_len], width,
                    label=f'Non-Residual (Depth {max_depth})')

        plt.xlabel('Layer')
        plt.ylabel('Gradient Magnitude')
        plt.title('Gradient Magnitudes Across Layers')
        plt.xticks(x, layers, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig("mlp_results/layer_gradient_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Compare MLP and ResidualMLP performance")
    parser.add_argument("--depths", nargs='+', type=int, default=[5, 10, 20, 30, 50, 100],
                        help="Network depths to compare")
    parser.add_argument("--load", type=str, default=None,
                        help="Load previous results from file")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    depths = args.depths
    residual_options = [False, True]  # Non-residual and residual

    if args.load:
        # Load previous results
        results = load_results(args.load)
        print(f"Loaded results from {args.load}")
    else:
        # Run experiments
        results = run_experiments(depths, residual_options, args.epochs)
        save_results(results, "mlp_results/final_results.pkl")

    # Plot results
    plot_results(results, depths, residual_options)
    print("Results plotted and saved to the 'mlp_results' directory")


if __name__ == "__main__":
    main()