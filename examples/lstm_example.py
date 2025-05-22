"""
Example script demonstrating how to use the LSTM model for CS:GO round win prediction.

This script shows a complete example of loading data, training a model,
and making predictions.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mbml.dataset import create_dataloaders

# Import from our package
from mbml.model import LSTMWinPredictor
from mbml.predict import batch_predict_rounds, load_model, visualize_round_predictions
from mbml.train import train_model


def main():
    # Set paths
    csv_path = "./round_frame_data.csv"
    model_dir = "./models/lstm_example"
    output_dir = "./reports/figures/lstm_predictions"

    # Ensure directories exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create dataloaders
    print("\n1. Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        preload=True,  # Set to False for very large datasets
    )

    # Get feature dimension from the data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["features"].shape[-1]
    print(f"Input feature dimension: {input_dim}")

    # 2. Create and train the model
    print("\n2. Creating and training the model...")
    model_params = {"input_dim": input_dim, "hidden_dim": 64, "num_layers": 2, "dropout": 0.2}

    # Either train a new model or load an existing one
    model_path = os.path.join(model_dir, "best_model.pt")

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = LSTMWinPredictor(**model_params)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training new model")
        train_model(
            csv_path=csv_path,
            model_save_dir=model_dir,
            **model_params,
            batch_size=32,
            num_epochs=10,  # Use more epochs for better performance
            learning_rate=0.001,
            weight_decay=1e-5,
            patience=5,
        )
        model = LSTMWinPredictor(**model_params)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    # 3. Make predictions on test data
    print("\n3. Making predictions on test data...")
    # Get one batch from test loader for demonstration
    test_batch = next(iter(test_loader))
    features = test_batch["features"].to(device)
    labels = test_batch["labels"]
    mask = test_batch["mask"]
    metadata = test_batch["metadata"]

    # Make predictions
    with torch.no_grad():
        predictions = model(features).cpu()

    # 4. Visualize predictions for one example
    print("\n4. Visualizing predictions...")
    # Select first example in batch
    example_id = 0
    example_features = features[example_id]
    example_preds = predictions[example_id]
    example_labels = labels[example_id]
    example_mask = mask[example_id]
    example_metadata = metadata[example_id]

    # Get non-padded values
    valid_preds = example_preds[example_mask > 0]
    valid_labels = example_labels[example_mask > 0]

    # Create visualization
    match_id, round_idx = example_metadata
    fig_path = os.path.join(output_dir, f"example_match_{match_id}_round_{round_idx}.png")

    fig = visualize_round_predictions(
        predictions=valid_preds, labels=valid_labels, match_id=match_id, round_idx=round_idx, save_path=fig_path
    )

    print(f"Visualization saved to {fig_path}")
    print("\nExample script completed!")


if __name__ == "__main__":
    main()
