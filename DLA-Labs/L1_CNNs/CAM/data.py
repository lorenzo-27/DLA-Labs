import torch
from torchvision import datasets, transforms
from pathlib import Path
import subprocess


def get_imagenette_transforms(img_size=160):
    """
    Get standard transforms for Imagenette dataset.

    Args:
        img_size: Image resolution

    Returns:
        transform_train, transform_val: transforms for training and validation
    """
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

    return transform_train, transform_val


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
    # Get transforms
    transform_train, transform_val = get_imagenette_transforms(img_size)

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Get class names (folder names)
    classes = train_dataset.classes

    return train_loader, val_loader, classes, train_dataset, val_dataset


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


def get_readable_class_name(class_code):
    """
    Get human-readable class name from class code.

    Args:
        class_code: Class code from Imagenette dataset

    Returns:
        str: Human-readable class name
    """
    class_mapping = get_imagenette_class_mapping()
    return class_mapping.get(class_code, class_code)