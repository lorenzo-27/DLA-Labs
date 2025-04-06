import torch
import torch.nn as nn
from torchvision import models


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

        # Set device for model - MOVED THIS BEFORE LOADING THE MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            CAM: Class Activation Map tensor
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


def train_resnet_cam(config):
    """
    Train a ResNet model on Imagenette dataset.

    Args:
        config: Configuration dictionary

    Returns:
        trained model
    """

    from DLA_Labs.L1_CNNs.train import train_model, get_logger
    from data import load_imagenette

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
