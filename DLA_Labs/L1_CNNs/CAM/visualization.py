import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path

from data import load_imagenette, get_imagenette_class_mapping


def prepare_tensor_for_display(img_tensor):
    """
    Convert a normalized tensor to a displayable numpy array.

    Args:
        img_tensor: Input image tensor (normalized)

    Returns:
        np.array: Image as numpy array suitable for display
    """
    # Denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img_tensor.cpu() * std + mean

    # Convert to numpy and transpose
    img_np = img_denorm.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def create_cam_heatmap(cam, img_shape):
    """
    Create a heatmap from CAM tensor.

    Args:
        cam: CAM tensor
        img_shape: Shape of the target image (height, width)

    Returns:
        np.array: Heatmap as numpy array
    """
    # Normalize CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (img_shape[1], img_shape[0]))

    # Apply jet colormap
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

    return cam_heatmap


def overlay_heatmap(img_np, cam_heatmap, alpha=0.6):
    """
    Overlay heatmap on original image.

    Args:
        img_np: Original image as numpy array
        cam_heatmap: Heatmap as numpy array
        alpha: Transparency factor

    Returns:
        np.array: Overlaid image
    """
    return cv2.addWeighted(img_np, alpha, cam_heatmap, 1 - alpha, 0)


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

    # Prepare original image for display
    img_np = prepare_tensor_for_display(img_tensor)

    # Forward pass
    with torch.no_grad():
        # Get CAM and predicted class
        cam, pred_class = model.get_cam(input_tensor, target_class)

        # Move to CPU and convert to numpy
        cam = cam.cpu().numpy()[0]  # Shape: H x W

        # Create heatmap from CAM
        cam_heatmap = create_cam_heatmap(cam, img_np.shape[:2])

        # Overlay heatmap on original image
        overlay = overlay_heatmap(img_np, cam_heatmap, alpha=0.6)

    # Get prediction class name
    pred_label = f"Class {pred_class}"
    if classes and pred_class < len(classes):
        pred_label = classes[pred_class]

        # Get human-readable class name
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


def visualize_cam(image_path, model, target_class=None, output_path=None, classes=None):
    """
    Generate and visualize Class Activation Maps for a given image file.

    Args:
        image_path: Path to the input image
        model: Trained ResNetCAM model
        target_class: Target class index (optional)
        output_path: Path to save visualization (optional)
        classes: List of class names (optional)

    Returns:
        Tuple of (original_image, cam_image, overlaid_image, pred_class)
    """
    # Ensure model is on the correct device
    device = model.device
    model.eval()

    # Load and preprocess image
    from PIL import Image
    from torchvision import transforms

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img)

    # Use the existing function to visualize CAM from the tensor
    img_np, cam_heatmap, overlay, pred_class = visualize_cam_from_tensor(
        img_tensor, model, target_class, classes
    )

    # Save visualization if output_path is specified
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    # Return results
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
        output_path_file = f"{output_dir}/random_cam_{i}.png" if output_dir else None

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
            if output_path_file:
                plt.savefig(output_path_file, bbox_inches='tight')

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