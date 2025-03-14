import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.m_mlp as m_mlp
import models.m_cnn as m_cnn
from dataset import load_data


def get_logger():
    """Configura e restituisce un logger."""
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()
console = Console()


def save_checkpoint(model, optimizer, epoch, loss, acc, config):
    """Salva un checkpoint del modello."""
    checkpoint_dir = config["training"]["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    residual = config["model"]["residual"]
    dataset_name = config["dataset"]["name"]

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_{'residual_' if residual else ''}{dataset_name}_epoch_{epoch}.pt"
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }, checkpoint_path)

    LOG.info(f"Checkpoint salvato in {checkpoint_path}")


def train_epoch(model, dataloader, criterion, optimizer, device) -> Tuple[float, float]:
    """
    Addestra il modello per un'epoca.

    Returns:
        loss_avg: Loss media per batch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistiche
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    loss_avg = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return loss_avg, accuracy


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float]:
    """
    Valuta il modello su un dataset.

    Returns:
        loss_avg: Loss media per batch
        accuracy: Accuratezza in percentuale
    """
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

    loss_avg = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return loss_avg, accuracy


def train_model(model, config, train_loader, val_loader, test_loader):
    """
    Addestra e valuta un modello.

    Args:
        model: Modello da addestrare
        config: Configurazione dell'addestramento
        train_loader: DataLoader per i dati di training
        val_loader: DataLoader per i dati di validazione
        test_loader: DataLoader per i dati di test
    """
    # Ottieni i parametri di training
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping_patience = config["training"]["early_stopping_patience"]
    optimizer_name = config["training"].get("optimizer", "adam")
    scheduler_name = config["training"].get("scheduler", "reduce_lr")

    # Modello su device
    model = model.to(device)

    # Criterio di loss
    criterion = nn.CrossEntropyLoss()

    # Ottimizzatore
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Ottimizzatore non supportato: {optimizer_name}")

    # Scheduler
    if scheduler_name.lower() == "reduce_lr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
    elif scheduler_name.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        raise ValueError(f"Scheduler non supportato: {scheduler_name}")

    # TensorBoard writer
    model_name = config["model"]["name"]
    residual = config["model"]["residual"]
    dataset_name = config["dataset"]["name"]
    run_name = f"{model_name}_{'residual_' if residual else ''}{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        LOG.info(f"Epoch {epoch}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        LOG.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc}")

        # Val
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        LOG.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc}")

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Learning rate scheduler
        if scheduler_name.lower() == "reduce_lr":
            scheduler.step(val_loss)
        else:  # cosine annealing
            scheduler.step()

        # Salvare checkpoint
        if epoch % config["training"]["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, config)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            LOG.info(f"Nuovo miglior modello con accuracy: {best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            LOG.info(f"Nessun miglioramento da {epochs_without_improvement} epoche")

        if epochs_without_improvement >= early_stopping_patience:
            LOG.info(f"Early stopping dopo {epoch} epoche")
            break

        # Ripristina il miglior modello
    model.load_state_dict(best_model_state)

    # Valutazione sul test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    LOG.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc}")

    # Salva il modello finale
    save_checkpoint(model, optimizer, num_epochs, test_loss, test_acc, config)

    # Chiudi TensorBoard writer
    writer.close()

    return {
        "model": model,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc
    }


def main():
    """Funzione principale per l'addestramento del modello."""
    parser = argparse.ArgumentParser(description="Addestramento di modelli di reti neurali")
    parser.add_argument("--config", type=str, required=True, help="Path al file di configurazione YAML")
    args = parser.parse_args()

    # Carica la configurazione
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    console.print(f"[bold green]Configurazione caricata:[/bold green]")
    console.print(config)

    # Imposta seed per riproducibilità
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # Carica i dati
    train_loader, val_loader, test_loader = load_data(config)
    LOG.info(f"Dataset {config['dataset']['name']} caricato")

    # Carica il modello
    model_name = config["model"]["name"]
    if model_name.lower() == "mlp":
        model_module = m_mlp
    elif model_name.lower() == "cnn":
        model_module = m_cnn
    else:
        raise ValueError(f"Modello non supportato: {model_name}")

    model = model_module.load_model(config)
    LOG.info(f"Modello {model_name} caricato")

    # Addestramento
    results = train_model(model, config, train_loader, val_loader, test_loader)

    LOG.info(f"Addestramento completato!")
    LOG.info(f"Miglior accuracy di validazione: {results['best_val_acc']:.2f}%")
    LOG.info(f"Accuracy di test: {results['test_acc']:.2f}%")


if __name__ == "__main__":
    main()