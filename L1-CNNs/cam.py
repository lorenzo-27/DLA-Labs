import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
import cv2
from pathlib import Path
import random


class ResNetCAM(nn.Module):
    """
    Class Activation Mapping implementation for ResNet models.
    Implementation based on the paper:
    "Learning Deep Features for Discriminative Localization" by Zhou et al. (CVPR'16)
    """

    def __init__(self, model_path=None):
        """
        Initialize the ResNet CAM model.

        Args:
            model_path: Path to a pre-trained model checkpoint (optional)
        """
        super().__init__()

        # Load pretrained ResNet-18 with updated parameter
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        # Replace the final fully connected layer with a new one for Imagenette (10 classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

        # Store the feature extractor (all layers before the final FC layer)
        self.features_extractor = nn.Sequential(*list(self.model.children())[:-2])

        # Store the FC weights separately for CAM computation
        self.fc_weights = None

        # Flag to determine when to extract features for CAM
        self.return_features = False
        self.last_features = None

        # Set device for model - MOVED THIS BEFORE LOADING THE MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model if provided
        if model_path:
            self._load_pretrained(model_path)

        # Get FC weights for CAM computation
        self._update_fc_weights()

        # Move model to device
        self.to(self.device)

    def _load_pretrained(self, model_path):
        """Load a pretrained model checkpoint with flexible key matching."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get the state dictionary from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Get the model's state dict keys
        model_keys = set(self.model.state_dict().keys())

        # Try different prefix patterns to match the keys
        new_state_dict = {}

        # Check each key in the loaded state dict
        for k, v in state_dict.items():
            # Try different prefix transformations
            possible_keys = [
                k,  # Original key
                k[6:] if k.startswith('model.') else k,  # Remove 'model.' prefix
                'model.' + k  # Add 'model.' prefix
            ]

            # Use the first matching key
            for possible_key in possible_keys:
                if possible_key in model_keys:
                    new_state_dict[possible_key] = v
                    break

        # Load the matched keys
        self.model.load_state_dict(new_state_dict, strict=False)

        # Print loading stats
        loaded_keys = set(new_state_dict.keys())
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys

        print(f"Loaded {len(loaded_keys)} keys successfully")
        print(f"Missing {len(missing_keys)} keys in checkpoint")
        print(f"Found {len(unexpected_keys)} unexpected keys in checkpoint")

        self.model.eval()

    def _update_fc_weights(self):
        """Extract the weights from the final FC layer for CAM computation."""
        self.fc_weights = self.model.fc.weight.data.to(self.device)

    def forward(self, x):
        """
        Forward pass through the model.
        During training, return only logits.
        When return_features is True, store the features for later CAM visualization.

        Args:
            x: Input tensor

        Returns:
            logits: Class predictions
        """
        # Ensure input is on the same device as model
        x = x.to(self.device)

        # Extract features using all layers except the last FC
        features = self.features_extractor(x)

        # Store features for CAM if needed
        if self.return_features:
            self.last_features = features

        # Global Average Pooling
        pooled = torch.mean(features, dim=[2, 3])

        # FC layer
        logits = self.model.fc(pooled)

        return logits

    def extract_features(self, x):
        """
        Extract features for CAM visualization.

        Args:
            x: Input tensor

        Returns:
            logits: Class predictions
            features: Feature maps before the final FC layer
        """
        # Ensure input is on the same device as model
        x = x.to(self.device)

        # Set flag to extract features
        self.return_features = True

        # Forward pass
        logits = self(x)

        # Get stored features
        features = self.last_features

        # Reset flag
        self.return_features = False
        self.last_features = None

        return logits, features

    def get_cam(self, input_tensor, class_idx=None):
        """
        Generate Class Activation Maps for the given input.

        Args:
            input_tensor: Input image tensor
            class_idx: Index of the class for which to generate CAM.
                      If None, use the predicted class.

        Returns:
            cam: Class Activation Map tensor
            pred_class: Predicted or specified class index
        """
        # Ensure input is on the same device as model
        input_tensor = input_tensor.to(self.device)

        # Forward pass to get logits and features
        logits, features = self.extract_features(input_tensor)

        # Get predicted class if not specified
        if class_idx is None:
            # Get predicted class index
            pred_class = torch.argmax(logits, dim=1).item()
        else:
            pred_class = class_idx

        # Get weights for the predicted class
        target_weights = self.fc_weights[pred_class, :]

        # Reshape weights to match feature map dimensions for broadcasting
        target_weights = target_weights.view(1, -1, 1, 1)

        # Compute CAM as weighted sum of feature maps
        cam = torch.sum(target_weights * features, dim=1)

        return cam, pred_class


def load_imagenette(batch_size, data_dir='./data', img_size=160):
    """
    Load the Imagenette dataset with appropriate transformations.

    Args:
        batch_size: batch size
        data_dir: directory where the data is or will be stored
        img_size: image resolution

    Returns:
        train_loader, val_loader, classes: data loaders and class names
    """
    # Transformations for Imagenette
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create Imagenette data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Dataset loading
    try:
        # Try loading from local directory first
        train_dataset = datasets.ImageFolder(root=str(data_path / 'imagenette2-160' / 'train'),
                                             transform=transform_train)
        val_dataset = datasets.ImageFolder(root=str(data_path / 'imagenette2-160' / 'val'),
                                           transform=transform_val)
    except FileNotFoundError:
        # If not found, download using a script
        import subprocess
        import os

        print("Downloading Imagenette dataset...")
        download_script = """
        wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -P {0}
        tar -xzf {0}/imagenette2-160.tgz -C {0}
        """.format(data_dir)

        subprocess.run(download_script, shell=True)

        # Now try loading again
        train_dataset = datasets.ImageFolder(root=str(data_path / 'imagenette2-160' / 'train'),
                                             transform=transform_train)
        val_dataset = datasets.ImageFolder(root=str(data_path / 'imagenette2-160' / 'val'),
                                           transform=transform_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)

    # Get class names (folder names)
    classes = train_dataset.classes

    return train_loader, val_loader, classes, train_dataset, val_dataset


def train_resnet_cam(config):
    """
    Train a ResNet model on Imagenette dataset.

    Args:
        config: Configuration dictionary

    Returns:
        trained model
    """
    from train import train_model, get_logger

    # Set up logger
    log = get_logger()

    # Load data
    batch_size = config["training"]["batch_size"]
    img_size = config["dataset"].get("img_size", 160)
    data_dir = config["dataset"]["data_dir"]

    train_loader, val_loader, classes, _, _ = load_imagenette(batch_size, data_dir, img_size)
    log.info(f"Imagenette dataset loaded with image size {img_size}px")

    # Initialize model
    model = ResNetCAM()
    log.info("ResNet-CAM model initialized")

    # Train model
    results = train_model(model, config, train_loader, val_loader, val_loader)

    return results["model"]


def get_imagenette_class_mapping():
    """
    Returns a mapping from ImageNet class codes used in folder names to human-readable class names.

    Returns:
        dict: Mapping from folder name codes to readable class names
    """
    code_to_name = {
        "n01440764": "tench",
        "n02102040": "English springer",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03445777": "golf ball",
        "n03425413": "gas pump",
        "n03888257": "parachute",
        "n03417042": "garbage truck"
    }
    return code_to_name


def visualize_cam_from_tensor(img_tensor, model, target_class=None, classes=None):
    """
    Generate and visualize Class Activation Maps for a given image tensor.

    Args:
        img_tensor: Input image tensor (already preprocessed)
        model: Trained ResNetCAM model
        target_class: Target class index (optional)
        classes: List of class names (optional)

    Returns:
        Tuple of (original_image, cam_image, overlaid_image, pred_class)
    """
    # Get model device
    device = model.device

    # Ensure model is in eval mode
    model.eval()

    # Move tensor to model's device
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Convert tensor to numpy for visualization
    # Denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img_tensor.cpu() * std + mean

    # Convert to numpy and transpose
    img_np = img_denorm.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    # Forward pass
    with torch.no_grad():
        # Get CAM and predicted class
        cam, pred_class = model.get_cam(input_tensor, target_class)

        # Move to CPU and convert to numpy
        cam = cam.cpu().numpy()[0]  # Shape: H x W

        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

        # Apply jet colormap
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_np, 0.6, cam_heatmap, 0.4, 0)

    # Get prediction class name
    pred_label = f"Class {pred_class}"
    if classes and pred_class < len(classes):
        pred_label = classes[pred_class]

        # Cerca di convertire il codice della classe in un nome leggibile
        class_mapping = get_imagenette_class_mapping()
        if pred_label in class_mapping:
            pred_label = class_mapping[pred_label]

    # Create visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam_heatmap)
    plt.title(f"CAM for {pred_label}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()

    return img_np, cam_heatmap, overlay, pred_class


def visualize_random_imagenette_samples(model, n_samples=5, dataset_type='val', data_dir='./data',
                                        img_size=160, target_class=None, output_dir=None):
    """
    Generate and visualize Class Activation Maps for randomly selected images from Imagenette.

    Args:
        model: Trained ResNetCAM model
        n_samples: Number of random samples to visualize
        dataset_type: 'train' or 'val' dataset
        data_dir: Directory containing the Imagenette dataset
        img_size: Image resolution
        target_class: Target class index (optional)
        output_dir: Directory to save visualizations (optional)

    Returns:
        List of visualizations
    """
    # Load Imagenette dataset
    _, _, classes, train_dataset, val_dataset = load_imagenette(batch_size=1, data_dir=data_dir, img_size=img_size)

    # Get class mapping
    class_mapping = get_imagenette_class_mapping()

    # Select the dataset
    dataset = train_dataset if dataset_type == 'train' else val_dataset

    # Get random indices
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    # Prepare output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # Process each random sample
    for i, idx in enumerate(indices):
        # Get image and label
        img_tensor, label = dataset[idx]

        # Get class name from class mapping if available
        class_name = classes[label]
        readable_class = class_mapping.get(class_name, class_name)

        print(f"Processing random sample {i + 1}/{n_samples} (dataset index: {idx}, class: {readable_class})")

        # Visualize CAM
        output_path = f"{output_dir}/random_cam_{i}.png" if output_dir else None

        try:
            img_np, cam_heatmap, overlay, pred_class = visualize_cam_from_tensor(
                img_tensor, model, target_class, classes
            )

            # Get predicted class readable name
            pred_name = classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}"
            pred_readable = class_mapping.get(pred_name, pred_name)

            # Store results
            results.append({
                'index': idx,
                'true_label': label,
                'true_class': classes[label],
                'true_class_readable': readable_class,
                'pred_class': pred_class,
                'pred_label': classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}",
                'pred_label_readable': pred_readable
            })

            # Save visualization if output directory is specified
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')

            plt.show()

        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            continue

    return results


def visualize_imagenette_by_index(model, index, dataset_type='val', data_dir='./data',
                                  img_size=160, target_class=None, output_path=None):
    """
    Generate and visualize Class Activation Map for a specific image from Imagenette by index.

    Args:
        model: Trained ResNetCAM model
        index: Index of the image in the dataset
        dataset_type: 'train' or 'val' dataset
        data_dir: Directory containing the Imagenette dataset
        img_size: Image resolution
        target_class: Target class index (optional)
        output_path: Path to save visualization (optional)

    Returns:
        Visualization results
    """
    # Load Imagenette dataset
    _, _, classes, train_dataset, val_dataset = load_imagenette(batch_size=1, data_dir=data_dir, img_size=img_size)

    # Get class mapping
    class_mapping = get_imagenette_class_mapping()

    # Select the dataset
    dataset = train_dataset if dataset_type == 'train' else val_dataset

    # Check if index is valid
    if index < 0 or index >= len(dataset):
        raise ValueError(f"Index {index} is out of range (0-{len(dataset) - 1})")

    # Get image and label
    img_tensor, label = dataset[index]

    # Get class name from class mapping if available
    class_name = classes[label]
    readable_class = class_mapping.get(class_name, class_name)

    print(f"Processing image at index {index} (class: {readable_class})")

    # Visualize CAM
    img_np, cam_heatmap, overlay, pred_class = visualize_cam_from_tensor(
        img_tensor, model, target_class, classes
    )

    # Get predicted class readable name
    pred_name = classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}"
    pred_readable = class_mapping.get(pred_name, pred_name)

    # Save visualization if output_path is specified
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    plt.show()

    # Return results
    return {
        'index': index,
        'true_label': label,
        'true_class': classes[label],
        'true_class_readable': readable_class,
        'pred_class': pred_class,
        'pred_label': classes[pred_class] if pred_class < len(classes) else f"Class {pred_class}",
        'pred_label_readable': pred_readable
    }


def main():
    """Main function to demonstrate CAM functionality."""
    from rich.console import Console
    import yaml
    import argparse
    import os

    console = Console()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Class Activation Mapping for ResNet on Imagenette")
    parser.add_argument("--config", type=str, default="cam_config.yaml", help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class index for CAM")
    parser.add_argument("--train", action="store_true", help="Train a new model")

    # New arguments for extended functionality
    parser.add_argument("--random", type=int, default=0, help="Number of random Imagenette samples to visualize")
    parser.add_argument("--index", type=int, default=None, help="Index of a specific Imagenette sample to visualize")
    parser.add_argument("--dataset", type=str, default="val", choices=["train", "val"],
                        help="Dataset to use (train or val)")
    parser.add_argument("--output_dir", type=str, default="cam_results/", help="Directory to save visualizations")

    args = parser.parse_args()

    # Default configuration
    default_config = {
        "model": {
            "name": "resnet_cam",
            "input_shape": [3, 160, 160],
            "num_classes": 10,
            "residual": True
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 20,
            "weight_decay": 0.0001,
            "checkpoint_dir": "checkpoints/",
            "log_every": 10,
            "save_every": 5,
            "early_stopping_patience": 5,
            "optimizer": "adam",
            "scheduler": "reduce_lr"
        },
        "dataset": {
            "name": "imagenette",
            "data_dir": "data/",
            "img_size": 160
        }
    }

    # Load configuration if file exists
    config = default_config
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[yellow]Configuration file not found. Using default configuration.[/yellow]")
            # Save default configuration
            with open("cam_config.yaml", "w") as f:
                yaml.dump(default_config, f)

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

    # Handle different visualization options
    if args.random > 0:
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
                f"Image {i + 1}: Index {res['index']}, Actual class: {res['true_class_readable']}, Predicted: {res['pred_label_readable']}")

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
        console.print("[bold yellow]No visualization option specified. Use --random or --index[/bold yellow]")


if __name__ == "__main__":
    main()