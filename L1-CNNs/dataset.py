import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(batch_size, val_size=5000, data_dir='./data'):
    """
    Carica il dataset MNIST e lo divide in train, validation e test.

    Args:
        batch_size: Dimensione del batch
        val_size: Numero di esempi per la validazione
        data_dir: Directory dove salvare i dati

    Returns:
        train_loader, val_loader, test_loader: DataLoader per i dataset
    """
    # Trasformazioni per MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Caricamento dei dataset
    ds_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    ds_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Split train in train e validation
    indices = np.random.permutation(len(ds_train))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    ds_val = Subset(ds_train, val_indices)
    ds_train = Subset(ds_train, train_indices)

    # Creazione dei DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def load_cifar10(batch_size, val_size=5000, data_dir='./data'):
    """
    Carica il dataset CIFAR-10 e lo divide in train, validation e test.

    Args:
        batch_size: Dimensione del batch
        val_size: Numero di esempi per la validazione
        data_dir: Directory dove salvare i dati

    Returns:
        train_loader, val_loader, test_loader: DataLoader per i dataset
    """
    # Trasformazioni per CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Caricamento dei dataset
    ds_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    ds_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Split train in train e validation
    indices = np.random.permutation(len(ds_train))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    ds_val = Subset(ds_train, val_indices)
    ds_train = Subset(ds_train, train_indices)

    # Per la validazione, usiamo la trasformazione di test (senza data augmentation)
    ds_val.dataset.transform = transform_test

    # Creazione dei DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def load_data(config):
    """
    Carica il dataset specificato nella configurazione.

    Args:
        config: Dizionario di configurazione

    Returns:
        train_loader, val_loader, test_loader: DataLoader per i dataset
    """
    dataset_name = config["dataset"]["name"]
    batch_size = config["training"]["batch_size"]
    val_size = config["dataset"]["val_size"]
    data_dir = config["dataset"]["data_dir"]

    if dataset_name.lower() == "mnist":
        return load_mnist(batch_size, val_size, data_dir)
    elif dataset_name.lower() == "cifar10":
        return load_cifar10(batch_size, val_size, data_dir)
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")


def main():
    """Test del modulo di caricamento dei dataset."""
    from rich.console import Console
    import yaml

    console = Console()

    # Test con MNIST
    console.print("\n[bold]Test MNIST:[/bold]")
    mnist_config = {
        "dataset": {"name": "mnist", "val_size": 5000, "data_dir": "./data"},
        "training": {"batch_size": 128}
    }

    train_loader, val_loader, test_loader = load_data(mnist_config)

    console.print(f"Batches in train_loader: {len(train_loader)}")
    console.print(f"Batches in val_loader: {len(val_loader)}")
    console.print(f"Batches in test_loader: {len(test_loader)}")

    # Verifica la forma dei dati
    for images, labels in train_loader:
        console.print(f"MNIST image batch shape: {images.shape}")
        console.print(f"MNIST labels batch shape: {labels.shape}")
        break

    # Test con CIFAR-10
    console.print("\n[bold]Test CIFAR-10:[/bold]")
    cifar_config = {
        "dataset": {"name": "cifar10", "val_size": 5000, "data_dir": "./data"},
        "training": {"batch_size": 128}
    }

    train_loader, val_loader, test_loader = load_data(cifar_config)

    console.print(f"Batches in train_loader: {len(train_loader)}")
    console.print(f"Batches in val_loader: {len(val_loader)}")
    console.print(f"Batches in test_loader: {len(test_loader)}")

    # Verifica la forma dei dati
    for images, labels in train_loader:
        console.print(f"CIFAR-10 image batch shape: {images.shape}")
        console.print(f"CIFAR-10 labels batch shape: {labels.shape}")
        break


if __name__ == "__main__":
    main()