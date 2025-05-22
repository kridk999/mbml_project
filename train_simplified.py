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

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds
from mbml.enhanced_model import ResponsiveWinPredictor


class MaskedBCELoss(nn.Module):
    """
    Binary Cross Entropy loss that ignores padded regions and applies optional weighting.
    """

    def __init__(self, event_weight=3.0):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.event_weight = event_weight

    def forward(self, predictions, targets, mask, player_counts=None):
        """
        Calculate masked loss with optional weighting for event frames.

        Args:
            predictions: Tensor of shape (batch_size, seq_len)
            targets: Tensor of shape (batch_size, seq_len)
            mask: Tensor of shape (batch_size, seq_len) with 1s for valid positions
            player_counts: Optional tensor of shape (batch_size, seq_len, count_features)
                           Used to detect events like kills for higher weighting

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Calculate base loss
        losses = self.bce(predictions, targets)

        # Create weights tensor (initially all 1s)
        weights = torch.ones_like(losses)

        # If player counts are provided, detect events (kills) and apply higher weights
        if player_counts is not None:
            # Get current and previous counts
            current_counts = player_counts

            # Shift to get previous frame counts (padding first frame)
            prev_counts = torch.cat([player_counts[:, :1], player_counts[:, :-1]], dim=1)

            # Calculate absolute changes in player counts
            count_changes = torch.abs(current_counts - prev_counts).sum(dim=2)

            # Apply higher weight when player count changes (indicating kills)
            event_frames = (count_changes > 0).float()
            weights = weights + (event_frames * (self.event_weight - 1.0))

        # Apply weights and mask
        masked_weighted_losses = losses * weights * mask

        # Normalize by sum of weights in valid positions
        return masked_weighted_losses.sum() / (weights * mask).sum()


def process_features(dataframe, feature_cols=None):
    """
    Extract features from dataframe and identify special feature indices.

    Returns:
        feature_cols: List of feature column names
        player_count_indices: Indices of player count features
        equipment_indices: Indices of equipment value features
    """
    if feature_cols is None:
        # Automatically select numerical features
        exclude_cols = ["match_id", "round_idx", "ctTeam", "tTeam", "rnd_winner", "clock"]
        numerical_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]

    # Find player count features (columns with 'alive' in the name)
    player_count_cols = [col for col in feature_cols if "alive" in col.lower()]
    player_count_indices = [feature_cols.index(col) for col in player_count_cols]

    # Find equipment value features
    equipment_cols = [col for col in feature_cols if "eqVal" in col]
    equipment_indices = [feature_cols.index(col) for col in equipment_cols]

    print(f"Found {len(player_count_indices)} player count features: {player_count_cols}")
    print(f"Found {len(equipment_indices)} equipment features: {equipment_cols}")

    return feature_cols, player_count_indices, equipment_indices


def emphasize_features(features, player_count_indices, equipment_indices):
    """Simple feature emphasis function that can be applied during training."""
    emphasized = features.clone()

    # Apply emphasis to player count features (multiply by 2.0)
    for col in player_count_indices:
        emphasized[:, :, col] = emphasized[:, :, col] * 2.0

    # Apply emphasis to equipment value features (multiply by 1.5)
    for col in equipment_indices:
        emphasized[:, :, col] = emphasized[:, :, col] * 1.5

    return emphasized


def train_simplified(
    csv_path,
    model_save_dir,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=10,
    event_weight=3.0,
    use_wandb=True,
    device=None,
):
    """
    Train the enhanced responsive model with a simplified approach.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create save directory
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested and available
    if use_wandb and WANDB_AVAILABLE:
        run_name = f"responsive_lstm_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="csgo_win_prediction",
            name=run_name,
            config={
                "model_type": "ResponsiveWinPredictor",
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "event_weight": event_weight,
            },
        )

    # Read some data to detect feature columns
    print(f"Loading data from {csv_path} to identify features")
    sample_df = pd.read_csv(csv_path, nrows=1000)
    feature_cols, player_count_indices, equipment_indices = process_features(sample_df)

    # Create the dataset (without transform to avoid pickling issues)
    dataset = CSGORoundDataset(csv_path=csv_path, feature_cols=feature_cols, preload=True)

    # Calculate dataset size and splits
    dataset_size = len(dataset)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    # Chronological split (no random assignment)
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Dataset split chronologically: {train_size} train, {val_size} val, {test_size} test")

    # Create DataLoaders WITHOUT multiprocessing (num_workers=0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for chronological order
        num_workers=0,  # No multiprocessing
        collate_fn=collate_variable_length_rounds,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_length_rounds
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_length_rounds
    )

    # Get input dimension from data
    print("Loading a sample batch to determine input dimensions")
    sample = next(iter(train_loader))
    input_dim = sample["features"].shape[-1]
    print(f"Input dimension: {input_dim}")

    # Create model
    print("Creating responsive model")
    model = ResponsiveWinPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        player_count_indices=player_count_indices,
        equipment_indices=equipment_indices,
        use_entmax=True,
    )

    model = model.to(device)
    print(f"Model:\n{model}")

    # Loss function
    criterion = MaskedBCELoss(event_weight=event_weight)

    # Optimizer and scheduler
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
        model.train()
        total_train_loss = 0
        total_train_samples = 0

        for batch in tqdm(train_loader, desc="Training"):
            # Get data
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            # Emphasize important features directly
            features = emphasize_features(features, player_count_indices, equipment_indices)

            # Extract player counts for weighted loss
            player_counts = None
            if player_count_indices:
                player_counts = features[:, :, player_count_indices]

            # Forward pass
            optimizer.zero_grad()
            predictions = model(features)

            # Calculate loss
            loss = criterion(predictions, labels, mask, player_counts)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            total_train_loss += loss.item() * mask.sum().item()
            total_train_samples += mask.sum().item()

        train_loss = total_train_loss / total_train_samples

        # Validate
        model.eval()
        total_val_loss = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)

                # Emphasize important features
                features = emphasize_features(features, player_count_indices, equipment_indices)

                # Extract player counts for weighted loss
                player_counts = None
                if player_count_indices:
                    player_counts = features[:, :, player_count_indices]

                # Forward pass
                predictions = model(features)

                # Calculate loss
                loss = criterion(predictions, labels, mask, player_counts)

                # Track loss
                total_val_loss += loss.item() * mask.sum().item()
                total_val_samples += mask.sum().item()

        val_loss = total_val_loss / total_val_samples

        # Update scheduler
        scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        # Print progress
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Log to wandb if enabled
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr})

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
    print("\nEvaluating test metrics...")
    model.eval()
    test_predictions = []
    test_labels = []
    test_masks = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Get data
            features = batch["features"].to(device)
            labels = batch["labels"]
            mask = batch["mask"]

            # Emphasize important features
            features = emphasize_features(features, player_count_indices, equipment_indices)

            # Get predictions
            predictions = model(features).cpu()

            # Save results
            test_predictions.append(predictions)
            test_labels.append(labels)
            test_masks.append(mask)

    # Concatenate results
    test_predictions = torch.cat(test_predictions, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_masks = torch.cat(test_masks, dim=0)

    # Calculate metrics on valid positions only
    valid_preds = test_predictions[test_masks > 0]
    valid_labels = test_labels[test_masks > 0]

    # Binary accuracy
    binary_preds = (valid_preds > 0.5).float()
    accuracy = (binary_preds == valid_labels).float().mean().item()

    # Area under ROC curve
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    try:
        auc = roc_auc_score(valid_labels.numpy(), valid_preds.numpy())
    except ValueError:
        # This can happen if there's only one class in the labels
        auc = float("nan")

    # Precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels.numpy(), binary_preds.numpy(), average="binary"
    )

    metrics = {"accuracy": accuracy, "auc": auc, "precision": precision, "recall": recall, "f1": f1}

    print(f"Test Metrics: {metrics}")

    # Create visualizations directory
    viz_dir = model_save_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    # Create visualizations for a subset of test examples
    print("Creating visualizations...")
    max_viz = 20
    viz_indices = np.random.choice(len(test_indices), min(max_viz, len(test_indices)), replace=False)

    for i in viz_indices:
        # Get original dataset index directly from test_indices
        original_idx = test_indices[i]

        # Get item data
        item = dataset[original_idx]
        match_id, round_idx = item["metadata"]
        features = item["features"].unsqueeze(0).to(device)
        labels = item["labels"].unsqueeze(0)

        # Apply feature emphasis
        features = emphasize_features(features, player_count_indices, equipment_indices)

        # Get predictions
        with torch.no_grad():
            preds = model(features).cpu().squeeze(0)

        # Extract valid sequence
        seq_len = features.size(1)

        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot predictions
        axes[0].plot(preds.numpy(), "b-", label="Win Probability")
        axes[0].axhline(y=labels[0, 0].item(), color="r", linestyle="--", label="Ground Truth")
        axes[0].set_ylabel("Win Probability")
        axes[0].set_title(f"Match {match_id}, Round {round_idx}")
        axes[0].legend()
        axes[0].grid(True)

        # Plot player counts
        if player_count_indices:
            player_counts = features.cpu().squeeze(0).numpy()
            for i, col in enumerate(player_count_indices):
                label = feature_cols[col]
                axes[1].plot(player_counts[:, col], label=label)

            axes[1].set_ylabel("Players Alive")
            axes[1].set_xlabel("Frame")
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(viz_dir / f"match_{match_id}_round_{round_idx}.png")
        plt.close(fig)

    # Save training history and metrics
    results = {
        "training_history": history,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
        "training_time": training_time,
        "model_params": {
            "model_type": "ResponsiveWinPredictor",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "player_count_indices": player_count_indices,
            "equipment_indices": equipment_indices,
        },
        "training_params": {
            "batch_size": batch_size,
            "num_epochs": epoch + 1,  # Actual number of epochs trained
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "event_weight": event_weight,
        },
        "feature_cols": feature_cols,
    }

    with open(model_save_dir / "training_results.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, (np.ndarray, list)) and len(str(v)) > 1000 else v for k, v in results.items()},
            f,
            indent=4,
        )

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

    # Finish wandb run if active
    if use_wandb and WANDB_AVAILABLE:
        # Log final metrics
        wandb.log(
            {
                "best_val_loss": best_val_loss,
                "test_accuracy": metrics["accuracy"],
                "test_auc": metrics["auc"],
                "test_f1": metrics["f1"],
            }
        )

        # Upload key visualizations
        wandb.log({"training_history": wandb.Image(str(model_save_dir / "training_history.png"))})

        wandb.finish()

    print(f"Training completed. Results saved to {model_save_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train responsive LSTM model for CS:GO round win prediction")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument(
        "--model_save_dir", type=str, default="./saved_models/responsive", help="Directory to save the model"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--event_weight", type=float, default=3.0, help="Weight factor for frames with events")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model with simplified approach
    train_simplified(
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
        event_weight=args.event_weight,
        use_wandb=not args.no_wandb,
        device=device,
    )
