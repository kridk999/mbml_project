"""
Helper script to make predictions with a trained model while ensuring feature compatibility.
This script ensures that the same features used during training are used during prediction.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds
from mbml.model import LSTMWinPredictor


def main():
    parser = argparse.ArgumentParser(description="Run predictions with compatible features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV data")
    parser.add_argument(
        "--output_dir", type=str, default="./reports/figures/predictions", help="Directory for output visualizations"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--max_viz", type=int, default=50, help="Maximum number of visualizations to create")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model and detect input dimensions
    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)

    # Extract dimensions from the model weights
    if "lstm.weight_ih_l0" in state_dict:
        input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        hidden_dim = state_dict["lstm.weight_ih_l0"].shape[0] // 4
        print(f"Detected input_dim={input_dim}, hidden_dim={hidden_dim} from model weights")
    else:
        print("ERROR: Could not detect dimensions from model weights")
        return

    # Detect number of layers from state dict keys
    num_layers = 1
    for key in state_dict.keys():
        if "lstm.weight_ih_l" in key:
            layer = int(key.split("_")[-1][1:])  # Extract layer number
            num_layers = max(num_layers, layer + 1)

    print(f"Detected {num_layers} LSTM layers in the model")

    # Create a compatible feature list with exactly the right number of features
    # First, load a sample of the dataframe to get column names
    print(f"Loading sample data from {args.csv_path}")
    df_sample = pd.read_csv(args.csv_path, nrows=100)

    # Get numerical columns that we can use as features
    numerical_cols = df_sample.select_dtypes(include=["number"]).columns.tolist()

    # Remove columns that are typically not used as features
    exclude_cols = ["match_id", "round_idx", "clock", "rnd_winner"]
    potential_features = [col for col in numerical_cols if col not in exclude_cols]

    if len(potential_features) < input_dim:
        print(
            f"""ERROR: The model expects {input_dim} features,
            but we only found {len(potential_features)} potential features"""
        )
        print(f"Available columns: {numerical_cols}")
        return

    # Select exactly the number of features the model expects
    selected_features = potential_features[:input_dim]

    print(f"\nUsing these {len(selected_features)} features to match model's input dimension:")
    for i, feature in enumerate(selected_features):
        print(f"  {i + 1}. {feature}")

    # Create the model with correct dimensions
    model = LSTMWinPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,  # Use some dropout as the original model likely did
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create dataset with the specific features
    print(f"\nCreating dataset with selected features...")
    dataset = CSGORoundDataset(
        csv_path=args.csv_path,
        feature_cols=selected_features,
        preload=False,  # Don't preload to save memory
    )

    # Create DataLoader
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_variable_length_rounds
    )

    # Process batch by batch and generate visualizations directly
    print("\nProcessing batches and creating visualizations...")

    num_visualized = 0
    max_visualizations = args.max_viz

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches"):
            features = batch["features"].to(device)
            labels = batch["labels"]
            mask = batch["mask"]
            metadata = batch["metadata"]

            predictions = model(features).cpu()

            # Process each item in the batch
            for i in range(len(metadata)):
                match_id, round_idx = metadata[i]

                # Get predictions and labels for this round
                round_preds = predictions[i]
                round_labels = labels[i]
                round_mask = mask[i]

                # Get non-padded values
                valid_mask = round_mask > 0
                valid_preds = round_preds[valid_mask]
                valid_labels = round_labels[valid_mask]

                # Create visualization
                fig_path = output_dir / f"match_{match_id}_round_{round_idx}.png"

                plt.figure(figsize=(10, 5))
                ax = plt.gca()

                # Plot predictions
                ax.plot(valid_preds.numpy(), "b-", label="Win Probability")

                # Add horizontal line at 0.5
                ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)

                # Plot ground truth
                ax.plot(valid_labels.numpy(), "r-", alpha=0.5, label="Ground Truth")

                # Set labels and title
                ax.set_xlabel("Frame")
                ax.set_ylabel("Win Probability")
                title = f"Match: {match_id}, Round: {round_idx}"
                ax.set_title(title)

                # Set y-axis limits
                ax.set_ylim(-0.05, 1.05)

                # Add legend
                ax.legend()

                # Add grid
                ax.grid(True, alpha=0.3)

                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close()

                num_visualized += 1

                # Limit number of visualizations for performance
                if num_visualized >= max_visualizations:
                    print(f"Created {num_visualized} visualizations (limit reached)")
                    break

            # Check again after processing batch
            if num_visualized >= max_visualizations:
                break

    print(f"\nPrediction complete! Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
