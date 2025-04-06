from rich.console import Console
import argparse
import os

from DLA_Labs.L1_CNNs.CAM.model import ResNetCAM, train_resnet_cam
from DLA_Labs.L1_CNNs.CAM.visualization import visualize_cam, visualize_random_imagenette_samples, visualize_imagenette_by_index
from DLA_Labs.L1_CNNs.CAM.utils import load_config, save_config, DEFAULT_CONFIG
from DLA_Labs.L1_CNNs.CAM.data import get_imagenette_class_mapping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Class Activation Mapping for ResNet on Imagenette")
    parser.add_argument("--config", type=str, default="cam_config.yaml", help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class index for CAM")
    parser.add_argument("--train", action="store_true", help="Train a new model")

    # Arguments for extended functionality
    parser.add_argument("--image", type=str, default=None, help="Path to a specific image file to visualize")
    parser.add_argument("--random", type=int, default=0, help="Number of random Imagenette samples to visualize")
    parser.add_argument("--index", type=int, default=None, help="Index of a specific Imagenette sample to visualize")
    parser.add_argument("--dataset", type=str, default="val", choices=["train", "val"],
                        help="Dataset to use (train or val)")
    parser.add_argument("--output_dir", type=str, default="cam_results/", help="Directory to save visualizations")

    return parser.parse_args()


def main():
    """Main function to demonstrate CAM functionality."""
    console = Console()

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config) if args.config else DEFAULT_CONFIG

    # If config file wasn't found, save the default config
    if args.config and config == DEFAULT_CONFIG:
        console.print(f"[yellow]Configuration file not found. Using default configuration.[/yellow]")
        save_config(DEFAULT_CONFIG, args.config)

    # Train or load model
    model = None
    if args.train:
        console.print("[bold green]Training ResNet-CAM model on Imagenette...[/bold green]")
        model = train_resnet_cam(config)
    elif args.model:
        console.print(f"[bold blue]Loading pretrained model from {args.model}...[/bold blue]")
        model = ResNetCAM(args.model)
    else:
        console.print("[bold yellow]No model specified. Initializing untrained model...[/bold yellow]")
        model = ResNetCAM()

    # Create output directory if needed
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get Imagenette class names
    imagenette_classes = list(get_imagenette_class_mapping().values())

    # Handle different visualization options
    if args.image:
        # Visualize CAM for a given image path
        original_filename = os.path.basename(args.image)
        output_filename = f"cam_{original_filename}"
        console.print(f"[bold magenta]Generating CAM for {args.image}...[/bold magenta]")
        output_path = os.path.join(args.output_dir, output_filename) if args.output_dir else None
        img_np, cam_heatmap, overlay, pred_class = visualize_cam(
            args.image, model, args.class_idx, output_path, imagenette_classes
        )

        # Print prediction result
        console.print(
            f"Predicted class: {imagenette_classes[pred_class] if pred_class < len(imagenette_classes) else f'Class {pred_class}'}")

    elif args.random > 0:
        # Visualize CAM for random images from Imagenette
        console.print(f"[bold magenta]Generating CAM for {args.random} random {args.dataset} images...[/bold magenta]")
        results = visualize_random_imagenette_samples(
            model, n_samples=args.random, dataset_type=args.dataset,
            data_dir=config["dataset"]["data_dir"], img_size=config["dataset"]["img_size"],
            target_class=args.class_idx, output_dir=args.output_dir
        )

        # Print results summary
        console.print("\n[bold cyan]Results Summary:[/bold cyan]")
        for i, res in enumerate(results):
            console.print(
                f"Image {i + 1}: Index {res['index']}, Actual class: {res['true_class_readable']}, " +
                f"Predicted: {res['pred_label_readable']}")

    elif args.index is not None:
        # Visualize CAM for a specific image by index
        console.print(f"[bold magenta]Generating CAM for {args.dataset} image at index {args.index}...[/bold magenta]")
        output_path = os.path.join(args.output_dir, f"index_{args.index}_cam.png") if args.output_dir else None
        result = visualize_imagenette_by_index(
            model, index=args.index, dataset_type=args.dataset,
            data_dir=config["dataset"]["data_dir"], img_size=config["dataset"]["img_size"],
            target_class=args.class_idx, output_path=output_path
        )

        # Print result
        console.print("\n[bold cyan]Result:[/bold cyan]")
        console.print(f"Image Index: {result['index']}")
        console.print(f"Actual class: {result['true_class_readable']} (label {result['true_label']})")
        console.print(f"Predicted: {result['pred_label_readable']} (class {result['pred_class']})")

    else:
        console.print("[bold yellow]No visualization option specified. Use --image, --random, or --index[/bold yellow]")


if __name__ == "__main__":
    main()