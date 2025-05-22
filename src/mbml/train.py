import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbml.dataset import create_dataloaders
from mbml.model import LSTMWinPredictor


class MaskedBCELoss(nn.Module):
    """
    Binary Cross Entropy loss that ignores padded regions.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: Tensor of shape (batch_size, seq_len)
            targets: Tensor of shape (batch_size, seq_len)
            mask: Tensor of shape (batch_size, seq_len) with 1s for valid positions and 0s for padding

        Returns:
            torch.Tensor: Scalar loss value
        """
        losses = self.bce(predictions, targets)  # Element-wise BCE loss
        masked_losses = losses * mask  # Apply mask to zero out padded regions
        # Sum losses and divide by sum of mask (number of valid elements)
        return masked_losses.sum() / mask.sum()


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        predictions = model(features)

        loss = criterion(predictions, labels, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_samples += mask.sum().item()

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            predictions = model(features)
            loss = criterion(predictions, labels, mask)

            total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

    return total_loss / total_samples


def evaluate_metrics(model, data_loader, device):
    """
    Evaluate the model on the test set and compute metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            features = batch["features"].to(device)
            labels = batch["labels"]
            mask = batch["mask"]

            predictions = model(features).cpu()

            all_preds.append(predictions)
            all_labels.append(labels)
            all_masks.append(mask)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate metrics only on non-padded regions
    valid_preds = all_preds[all_masks > 0]
    valid_labels = all_labels[all_masks > 0]

    # Binary accuracy
    binary_preds = (valid_preds > 0.5).float()
    accuracy = (binary_preds == valid_labels).float().mean().item()

    # Area under ROC curve
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(valid_labels.numpy(), valid_preds.numpy())
    except ValueError:
        # This can happen if there's only one class in the labels
        auc = float("nan")

    # Precision, recall, F1 score
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels.numpy(), binary_preds.numpy(), average="binary"
    )

    return {"accuracy": accuracy, "auc": auc, "precision": precision, "recall": recall, "f1": f1}


def train_model(
    csv_path,
    model_save_dir,
    input_dim=None,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=10,
    feature_cols=None,
    device=None,
):
    """
    Train the LSTM model.

    Args:
        csv_path: Path to the CSV data file
        model_save_dir: Directory to save the model and results
        input_dim: Input dimension (number of features). If None, determined from data
        hidden_dim: Hidden dimension of the LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 penalty)
        patience: Number of epochs to wait for improvement before early stopping
        feature_cols: Optional list of feature column names to use
        device: Device to use for training ('cpu' or 'cuda')

    Returns:
        Dictionary with training history and best metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create save directory
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path, batch_size=batch_size, feature_cols=feature_cols, chronological=True
    )

    # Get input dimension from data if not provided
    if input_dim is None:
        sample = next(iter(train_loader))
        input_dim = sample["features"].shape[-1]
        print(f"Inferred input_dim from data: {input_dim}")

    # Create model
    model = LSTMWinPredictor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    model = model.to(device)
    print(f"Model:\n{model}")

    # Loss function, optimizer, scheduler
    criterion = MaskedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=patience // 3, verbose=True
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    start_time = time.time()

    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{num_epochs}, LR: {current_lr:.2e}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        # Print progress
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0

            # Save best model
            torch.save(best_model_state, model_save_dir / "best_model.pt")
            print(f"✓ New best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"! No improvement for {epochs_without_improvement} epochs")

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = evaluate_metrics(model, test_loader, device)
    print(f"Test Metrics: {metrics}")

    # Save training history and metrics
    results = {
        "training_history": history,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
        "training_time": training_time,
        "model_params": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        "training_params": {
            "batch_size": batch_size,
            "num_epochs": epoch + 1,  # Actual number of epochs trained
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
        },
    }

    with open(model_save_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogy(history["learning_rate"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.tight_layout()
    plt.savefig(model_save_dir / "training_history.png")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for CS:GO round win prediction")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--model_save_dir", type=str, default="./saved_models", help="Directory to save the model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    train_model(
        csv_path=args.csv_path,
        model_save_dir=args.model_save_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )
