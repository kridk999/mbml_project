"""
CSGO Win Prediction model that predicts if the T side will win a round.
Label 1 = T side wins, Label 0 = CT side wins (consistent across all matches).
"""
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

import wandb
from mbml.data_transform import DeltaFeatures, FeatureEmphasis
from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds, create_dataloaders
from mbml.enhanced_model import ResponsiveWinPredictor


class MaskedLoss(nn.Module):
    """
    Loss function that ignores padded regions and applies optional weighting.
    """

    def __init__(self, base_loss="bce", event_weight=3.0):
        super().__init__()
        self.base_loss = base_loss
        self.event_weight = event_weight

        if base_loss == "bce":
            self.criterion = nn.BCELoss(reduction="none")
        elif base_loss == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss function: {base_loss}")

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
        losses = self.criterion(predictions, targets)

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
    equipment_cols = [col for col in feature_cols if "eqval" in col.lower()]
    equipment_indices = [feature_cols.index(col) for col in equipment_cols]

    print(f"Found {len(player_count_indices)} player count features: {player_count_cols}")
    print(f"Found {len(equipment_indices)} equipment features: {equipment_cols}")

    return feature_cols, player_count_indices, equipment_indices


class FeatureTransform:
    """
    Feature transform class that is picklable for multiprocessing.
    """

    def __init__(self, player_count_indices, equipment_indices, delta_feature_indices=None):
        self.player_count_indices = player_count_indices
        self.equipment_indices = equipment_indices
        self.delta_feature_indices = delta_feature_indices

    def __call__(self, features):
        # Apply feature emphasis
        emphasis = FeatureEmphasis(
            player_count_emphasis=2.0,
            equipment_emphasis=1.5,
            player_count_cols=self.player_count_indices,
            equipment_cols=self.equipment_indices,
        )
        features = emphasis(features)

        # Add delta features if specified
        if self.delta_feature_indices:
            delta_transform = DeltaFeatures(key_feature_indices=self.delta_feature_indices)
            features = delta_transform(features)

        return features


def get_feature_transform(player_count_indices, equipment_indices, delta_feature_indices=None):
    """
    Create a feature transform pipeline.
    """
    return FeatureTransform(player_count_indices, equipment_indices, delta_feature_indices)


def train_epoch(model, train_loader, optimizer, criterion, device, player_count_indices=None):
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

        # Extract player counts if available and needed
        player_counts = None
        if player_count_indices is not None:
            player_counts = features[:, :, player_count_indices]

        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)

        # Calculate loss
        loss = criterion(predictions, labels, mask, player_counts)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_samples += mask.sum().item()

    return total_loss / total_samples


def validate(model, val_loader, criterion, device, player_count_indices=None):
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

            # Extract player counts if available and needed
            player_counts = None
            if player_count_indices is not None:
                player_counts = features[:, :, player_count_indices]

            # Forward pass
            predictions = model(features)

            # Calculate loss
            loss = criterion(predictions, labels, mask, player_counts)

            total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

    return total_loss / total_samples


def collect_predictions(model, data_loader, device):
    """
    Collect predictions and ground truth for analysis.
    """
    model.eval()
    all_metadata = []
    all_features = []
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting Predictions"):
            features = batch["features"].to(device)
            labels = batch["labels"].cpu()
            mask = batch["mask"].cpu()
            metadata = batch["metadata"]

            predictions = model(features).cpu()

            all_metadata.extend(metadata)
            all_features.append(features.cpu())
            all_preds.append(predictions)
            all_labels.append(labels)
            all_masks.append(mask)

    # Concatenate all tensors
    all_features = torch.cat(all_features, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    return {
        "metadata": all_metadata,
        "features": all_features,
        "predictions": all_preds,
        "labels": all_labels,
        "masks": all_masks,
    }


def evaluate_metrics(predictions_dict):
    """
    Compute evaluation metrics from predictions.
    """
    predictions = predictions_dict["predictions"]
    labels = predictions_dict["labels"]
    masks = predictions_dict["masks"]

    # Calculate metrics only on non-padded regions
    valid_preds = predictions[masks > 0]
    valid_labels = labels[masks > 0]

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

    return {"accuracy": accuracy, "auc": auc, "precision": precision, "recall": recall, "f1": f1}


def analyze_event_responsiveness(predictions_dict, player_count_indices, equipment_indices):
    """
    Analyze how responsive the model is to events like kills.
    """
    features = predictions_dict["features"]
    predictions = predictions_dict["predictions"]
    masks = predictions_dict["masks"]

    # Extract player counts
    player_counts = features[:, :, player_count_indices]

    # Calculate changes in player counts (absolute sum of differences)
    player_count_changes = torch.zeros_like(predictions)

    for i in range(1, player_counts.shape[1]):
        player_count_changes[:, i] = torch.abs(player_counts[:, i] - player_counts[:, i - 1]).sum(dim=1)

    # Calculate prediction changes
    prediction_changes = torch.zeros_like(predictions)
    for i in range(1, predictions.shape[1]):
        prediction_changes[:, i] = torch.abs(predictions[:, i] - predictions[:, i - 1])

    # Analyze correlations
    # Get only valid entries
    valid_count_changes = player_count_changes[masks > 0].numpy()
    valid_pred_changes = prediction_changes[masks > 0].numpy()

    # Events are where player count changes
    events = valid_count_changes > 0

    # Average prediction change during events vs. non-events
    event_pred_changes = valid_pred_changes[events]
    nonevent_pred_changes = valid_pred_changes[~events]

    avg_event_change = event_pred_changes.mean() if len(event_pred_changes) > 0 else 0
    avg_nonevent_change = nonevent_pred_changes.mean() if len(nonevent_pred_changes) > 0 else 0

    # Correlation between count changes and prediction changes
    from scipy.stats import pearsonr

    try:
        correlation, p_value = pearsonr(valid_count_changes, valid_pred_changes)
    except ValueError:
        correlation, p_value = 0, 1

    return {
        "avg_pred_change_on_events": float(avg_event_change),
        "avg_pred_change_on_nonevents": float(avg_nonevent_change),
        "change_ratio": float(avg_event_change / max(avg_nonevent_change, 1e-5)),
        "correlation": float(correlation),
        "p_value": float(p_value),
    }


def visualize_predictions(predictions_dict, output_dir, feature_cols, player_count_indices, equipment_indices, max_viz=20):
    """
    Create visualizations of model predictions to analyze responsiveness.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    features = predictions_dict["features"]
    predictions = predictions_dict["predictions"]
    labels = predictions_dict["labels"]
    masks = predictions_dict["masks"]
    metadata = predictions_dict["metadata"]

    # Select a subset of rounds to visualize
    num_rounds = min(len(metadata), max_viz)
    indices = np.random.choice(len(metadata), num_rounds, replace=False)

    for idx in indices:
        match_id, round_idx = metadata[idx]

        # Get predictions, features, etc. for this round
        round_features = features[idx]
        round_preds = predictions[idx]
        round_labels = labels[idx]
        round_mask = masks[idx]

        # Get valid (non-padded) length
        valid_len = int(round_mask.sum().item())

        # Extract only valid frames
        valid_features = round_features[:valid_len].numpy()
        valid_preds = round_preds[:valid_len].numpy()
        valid_labels = round_labels[:valid_len].numpy()

        # Extract player counts
        player_counts = valid_features[:, player_count_indices]

        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot predictions vs ground truth
        axes[0].plot(valid_preds, "b-", label="T Win Probability")
        axes[0].axhline(y=valid_labels[0], color="r", linestyle="-", label="Ground Truth (1=T win, 0=CT win)")
        axes[0].set_ylabel("T Side Win Probability")
        axes[0].set_title(f"Match {match_id}, Round {round_idx}")
        axes[0].legend()
        axes[0].grid(True)

        # Plot CT/T player counts - properly identify columns by name
        ct_alive_idx = None
        t_alive_idx = None
        
        # Get the column names for each player_count_index
        player_count_cols = [feature_cols[idx] for idx in player_count_indices]
        
        for i, col_name in enumerate(player_count_cols):
            if "num_ct_alive" in col_name.lower():
                ct_alive_idx = player_count_indices[i]
            elif "num_t_alive" in col_name.lower():
                t_alive_idx = player_count_indices[i]
        
        # Always display player counts, even if one is missing
        axes[1].set_ylabel("Players Alive")
        axes[1].set_ylim(0, 5)  # Set y-axis limits to 0-5 players (max team size in CS:GO)
        axes[1].grid(True)  # Add grid for better visibility
        
        # Enhanced normalization function that handles various value ranges
        def normalize_player_count(values, expected_max=5.0):
            """Normalize player count values to 0-5 range"""
            values_min = values.min()
            values_max = values.max()
            
            # If values are already in valid range, return as-is
            if values_max <= expected_max and values_min >= 0:
                return values
                
            # Different normalization strategies based on the data range
            if values_max > expected_max:
                # Case 1: Max exceeds expected (like 0-10 instead of 0-5)
                scale_factor = expected_max / values_max if values_max > 0 else 1.0
                normalized = values * scale_factor
                print(f"Normalizing player count from range {values_min:.1f}-{values_max:.1f} to 0-{expected_max} (scale factor: {scale_factor:.2f})")
                return normalized
            else:
                # Just return the values if they don't need normalization
                return values
        
        # Plot CT players - Blue line
        if ct_alive_idx is not None:
            ct_values = valid_features[:, ct_alive_idx]
            ct_normalized = normalize_player_count(ct_values)
            axes[1].plot(ct_normalized, "b-", label="CT Players", linewidth=2.0)
            
            print(f"CT alive original values (first 5): {ct_values[:5]}")
            print(f"CT alive normalized values (first 5): {ct_normalized[:5]}")
        else:
            # If CT data not found, show a placeholder with zeros
            axes[1].plot([0] * valid_len, "b--", label="CT Players (missing)", linewidth=2.0)
        
        # Plot T players - Red line with higher z-order to ensure visibility
        if t_alive_idx is not None:
            t_values = valid_features[:, t_alive_idx]
            t_normalized = normalize_player_count(t_values)
            axes[1].plot(t_normalized, "r-", label="T Players", linewidth=2.0, zorder=10)
            
            print(f"T alive original values (first 5): {t_values[:5]}")
            print(f"T alive normalized values (first 5): {t_normalized[:5]}")
        else:
            # If T data not found, show a placeholder with zeros
            axes[1].plot([0] * valid_len, "r--", label="T Players (missing)", linewidth=2.0, zorder=10)
        
        axes[1].legend()
        axes[1].grid(True)

        # Plot equipment values
        if len(equipment_indices) > 0:
            # Get the column names for each equipment_index
            equipment_cols = [feature_cols[idx] for idx in equipment_indices]
            
            ct_eq_idx = None
            t_eq_idx = None
            
            # Properly identify equipment values by column name
            for i, col_name in enumerate(equipment_cols):
                if "cteqval" in col_name.lower():
                    ct_eq_idx = equipment_indices[i]
                elif "teqval" in col_name.lower():
                    t_eq_idx = equipment_indices[i]
            
            # Always show equipment values, even if missing
            axes[2].set_ylabel("Equipment Value")
            axes[2].set_xlabel("Frame")
            
            # Plot CT equipment
            if ct_eq_idx is not None:
                axes[2].plot(valid_features[:, ct_eq_idx], "b-", label="CT Equipment")
            else:
                # If CT equipment data not found, show a placeholder
                axes[2].plot([0] * valid_len, "b--", label="CT Equipment (missing)")
                
            # Plot T equipment
            if t_eq_idx is not None:
                axes[2].plot(valid_features[:, t_eq_idx], "r-", label="T Equipment") 
            else:
                # If T equipment data not found, show a placeholder
                axes[2].plot([0] * valid_len, "r--", label="T Equipment (missing)")
                
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"match_{match_id}_round_{round_idx}.png"))
        plt.close(fig)


def train_responsive_model(
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
    feature_cols=None,
    event_weight=3.0,
    use_wandb=True,
    device=None,
):
    """
    Train the enhanced responsive model.

    Args:
        csv_path: Path to the CSV data file
        model_save_dir: Directory to save the model and results
        hidden_dim: Hidden dimension of the LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay (L2 penalty)
        patience: Number of epochs to wait for improvement before early stopping
        feature_cols: Optional list of feature column names to use
        event_weight: Weight factor for frames with events (kills)
        use_wandb: Whether to log to Weights & Biases
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

    # Initialize wandb if requested
    if use_wandb:
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
    feature_cols, player_count_indices, equipment_indices = process_features(sample_df, feature_cols)

    # Create feature transform
    transform = get_feature_transform(player_count_indices, equipment_indices)
    # Create dataloaders with chronological splitting to avoid look-ahead bias
    print("Creating dataloaders with chronological splitting")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        feature_cols=feature_cols,
        chronological=True,
        transform=transform,
        num_workers=0,  # Disable multiprocessing to avoid pickling issues
    )

    # Get input dimension from data
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
    )

    model = model.to(device)
    print(f"Model:\n{model}")

    # Loss function weighted to emphasize event frames
    criterion = MaskedLoss(base_loss="bce", event_weight=event_weight)

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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, player_count_indices)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, player_count_indices)

        # Update scheduler
        scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        # Print progress
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Log to wandb if enabled
        if use_wandb:
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

    # Collect predictions and evaluate
    print("\nCollecting test predictions...")
    test_predictions = collect_predictions(model, test_loader, device)

    # Calculate metrics
    print("Evaluating test metrics...")
    metrics = evaluate_metrics(test_predictions)
    print(f"Test Metrics: {metrics}")

    # Analyze event responsiveness
    print("Analyzing model responsiveness to events...")
    responsiveness = analyze_event_responsiveness(test_predictions, player_count_indices, equipment_indices)
    print(f"Responsiveness: {responsiveness}")

    # Create visualizations
    print("Creating visualizations...")
    viz_dir = model_save_dir / "visualizations"
    visualize_predictions(test_predictions, viz_dir, feature_cols, player_count_indices, equipment_indices, max_viz=20)

    # Save training history and metrics
    results = {
        "training_history": history,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
        "responsiveness": responsiveness,
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

    # Finish wandb run if active
    if use_wandb:
        # Log final metrics
        wandb.log(
            {
                "best_val_loss": best_val_loss,
                "test_accuracy": metrics["accuracy"],
                "test_auc": metrics["auc"],
                "test_f1": metrics["f1"],
                "responsiveness_ratio": responsiveness["change_ratio"],
                "responsiveness_correlation": responsiveness["correlation"],
            }
        )

        # Upload key visualizations
        wandb.log({"training_history": wandb.Image(model_save_dir / "training_history.png")})

        wandb.finish()

    return results, model


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

    # Train model
    train_responsive_model(
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
