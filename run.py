#!/usr/bin/env python
"""
Main script to train and evaluate LSTM model for CS:GO round win prediction.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch

from mbml.predict import batch_predict_rounds, load_model
from mbml.train import train_model


def main():
    parser = argparse.ArgumentParser(description="CS:GO Round Win Prediction with LSTM")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument(
        "--mode", type=str, choices=["train", "predict", "train_and_predict"], default="train", help="Mode of operation"
    )

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Output directories
    parser.add_argument("--model_dir", type=str, help="Directory to save/load model files")
    parser.add_argument("--output_dir", type=str, help="Directory to save prediction results")

    # Prediction parameters
    parser.add_argument("--model_path", type=str, help="Path to the trained model (for prediction)")
    parser.add_argument("--num_rounds", type=int, help="Number of rounds to predict (None for all)")

    # Other parameters
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create timestamped directory if not specified
    if not args.model_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_dir = f"./saved_models/lstm_model_{timestamp}"

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Default output directory if not specified
    if not args.output_dir:
        args.output_dir = model_dir / "predictions"

    # Define model path
    model_path = args.model_path if args.model_path else model_dir / "best_model.pt"
    config_path = model_dir / "training_results.json"

    # Train mode
    if args.mode in ["train", "train_and_predict"]:
        print(f"\nStarting training with the following parameters:")
        print(f"  CSV path: {args.csv_path}")
        print(f"  Model directory: {model_dir}")
        print(f"  Hidden dimension: {args.hidden_dim}")
        print(f"  Number of layers: {args.num_layers}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Device: {device}\n")

        # Train the model
        train_results = train_model(
            csv_path=args.csv_path,
            model_save_dir=model_dir,
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

        print("\nTraining complete!")
        print(f"Best validation loss: {train_results['best_val_loss']:.6f}")
        print(f"Test metrics: {train_results['test_metrics']}")

    # Predict mode
    if args.mode in ["predict", "train_and_predict"]:
        print(f"\nStarting prediction with the following parameters:")
        print(f"  CSV path: {args.csv_path}")
        print(f"  Model path: {model_path}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Number of rounds: {args.num_rounds if args.num_rounds else 'all'}")
        print(f"  Device: {device}\n")

        # Load the model
        model = load_model(model_path, config_path, device)

        # Make predictions
        predictions_df = batch_predict_rounds(
            model=model,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_rounds=args.num_rounds,
            device=device,
        )

        print("\nPrediction complete!")
        print(f"Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
