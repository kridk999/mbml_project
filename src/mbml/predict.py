import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds
from mbml.model import LSTMWinPredictor


def load_model(model_path, config_path=None, device=None):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model state
        config_path: Optional path to model configuration file
        device: Device to load the model on

    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First, load the state dict to inspect dimensions
    state_dict = torch.load(model_path, map_location=device)

    # Extract dimensions from the model weights
    if "lstm.weight_ih_l0" in state_dict:
        # The input dimension is the second dimension of the LSTM input weights
        # Shape is [hidden_dim*4, input_dim]
        input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4  # LSTM uses 4 gates
        print(f"Detected input_dim={input_dim}, hidden_dim={hidden_dim} from model weights")
    else:
        print("WARNING: Could not detect dimensions from model weights")

    # Try to get other parameters from config if available
    num_layers = 1  # Default
    dropout = 0.0  # Default

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            model_params = config.get("model_params", {})
            # We already detected input_dim and hidden_dim from weights
            num_layers = model_params.get("num_layers", 1)
            dropout = model_params.get("dropout", 0.0)
            print(f"Using num_layers={num_layers}, dropout={dropout} from config")
        except Exception as e:
            print(f"Error reading config file: {e}")
    else:
        print("No config file found or config path invalid. Using default parameters for num_layers and dropout.")

    # Create model with detected dimensions
    model = LSTMWinPredictor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)

    # Load the state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path} to {device}")
    return model


def predict_round_probabilities(model, features, device=None):
    """
    Predict win probabilities for a sequence of round frames.

    Args:
        model: Trained LSTM model
        features: Tensor of shape (seq_len, input_dim) or (batch_size, seq_len, input_dim)
        device: Device to perform inference on

    Returns:
        Tensor of win probabilities with shape (seq_len,) or (batch_size, seq_len)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add batch dimension if needed
    if len(features.shape) == 2:
        features = features.unsqueeze(0)

    features = features.to(device)

    with torch.no_grad():
        predictions = model(features)

    # Remove batch dimension if it was added
    if predictions.shape[0] == 1:
        predictions = predictions.squeeze(0)

    return predictions.cpu()


def visualize_round_predictions(predictions, labels=None, match_id=None, round_idx=None, seconds=None, save_path=None):
    """
    Visualize predictions for a single round.

    Args:
        predictions: Tensor of shape (seq_len,) with win probabilities
        labels: Optional tensor of ground truth labels
        match_id: Optional match ID for the title
        round_idx: Optional round index for the title
        seconds: Optional time points for the x-axis
        save_path: Optional path to save the figure

    Returns:
        The matplotlib figure
    """
    fig, ax = plt.figure(figsize=(10, 5)), plt.gca()

    # Convert to numpy arrays
    predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions

    # Create x-axis values
    x = seconds if seconds is not None else np.arange(len(predictions))

    # Plot predictions
    ax.plot(x, predictions, "b-", label="Win Probability")

    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)

    # Plot labels if available
    if labels is not None:
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        ax.plot(x, labels, "r-", alpha=0.5, label="Ground Truth")

    # Set labels and title
    ax.set_xlabel("Time (seconds)" if seconds is not None else "Frame")
    ax.set_ylabel("Win Probability")
    title = f"Round Predictions"
    if match_id is not None:
        title += f" - Match: {match_id}"
    if round_idx is not None:
        title += f", Round: {round_idx}"
    ax.set_title(title)

    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def batch_predict_rounds(
    model, csv_path, feature_cols=None, output_dir=None, batch_size=32, num_rounds=None, device=None
):
    """
    Make predictions for multiple rounds in the dataset.

    Args:
        model: Trained LSTM model
        csv_path: Path to the CSV data file
        feature_cols: Optional list of feature column names
        output_dir: Directory to save predictions and visualizations
        batch_size: Batch size for inference
        num_rounds: Optional number of rounds to predict (None for all)
        device: Device to perform inference on

    Returns:
        DataFrame with predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    dataset = CSGORoundDataset(
        csv_path=csv_path,
        feature_cols=feature_cols,
        preload=False,  # Don't preload to save memory
    )

    # Limit number of rounds if specified
    if num_rounds is not None:
        dataset.rounds = dataset.rounds[:num_rounds]

    # Create DataLoader
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_variable_length_rounds
    )

    # Make predictions
    all_predictions = []
    all_labels = []
    all_metadata = []

    for batch in tqdm(data_loader, desc="Predicting"):
        features = batch["features"].to(device)
        labels = batch["labels"]
        mask = batch["mask"]
        metadata = batch["metadata"]

        predictions = model(features).cpu()

        # Store results
        all_predictions.append(predictions)
        all_labels.append(labels)
        all_metadata.extend(metadata)

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create results DataFrame
    results = []
    for i, (match_id, round_idx) in enumerate(all_metadata):
        # Get data for this round
        pred = all_predictions[i]
        label = all_labels[i]
        mask = all_predictions[i] >= 0  # Just to get the same shape as the prediction

        # Get non-padded values
        valid_preds = pred[mask]
        valid_labels = label[mask] if label.shape == pred.shape else None

        # Add to results
        results.append(
            {
                "match_id": match_id,
                "round_idx": round_idx,
                "predictions": valid_preds.tolist(),
                "labels": valid_labels.tolist() if valid_labels is not None else None,
            }
        )

        # Create visualization if requested
        if output_dir:
            fig_path = output_dir / f"match_{match_id}_round_{round_idx}.png"
            visualize_round_predictions(
                predictions=valid_preds, labels=valid_labels, match_id=match_id, round_idx=round_idx, save_path=fig_path
            )
            plt.close()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results if requested
    if output_dir:
        results_df.to_csv(output_dir / "predictions.csv", index=False)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Predict with trained LSTM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model file")
    parser.add_argument("--config_path", type=str, help="Path to the model configuration file")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--output_dir", type=str, help="Directory to save predictions and visualizations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_rounds", type=int, help="Number of rounds to predict (None for all)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model_path, args.config_path, device)

    # Make predictions
    batch_predict_rounds(
        model=model,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_rounds=args.num_rounds,
        device=device,
    )


if __name__ == "__main__":
    main()
