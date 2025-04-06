import yaml
from pathlib import Path
import logging

# Default configuration
DEFAULT_CONFIG = {
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


def get_logger(name='CAM'):
    """Create and configure a logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

    return logger


def load_config(config_path):
    """Load configuration from file or return default config."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger = get_logger()
        logger.warning(f"Error loading config file: {e}. Using default configuration.")
        return DEFAULT_CONFIG


def save_config(config, config_path='cam_config.yaml'):
    """Save configuration to file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


def ensure_dir(directory):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)