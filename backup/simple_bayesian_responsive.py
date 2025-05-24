#!/usr/bin/env python3
"""
Simple Bayesian Responsive Visualization 
=======================================

Simplified version that creates Bayesian visualizations matching responsive model style.
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load the simple demo that we know works
import sys
sys.path.append('/Users/kristofferkjaer/Desktop/mbml_project')

from simple_bayesian_lstm import SimpleBayesianLSTM


def load_model_and_data():
    """Load model and create simple test data."""
    print("Loading Bayesian model...")
    
    # Load model params
    results_path = Path("saved_models/simple_bayesian/training_results.json")
    with open(results_path) as f:
        results = json.load(f)
    
    params = results['model_params']
    
    # Create model
    model = SimpleBayesianLSTM(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    # Load weights
    device = torch.device("cpu")
    checkpoint = torch.load("saved_models/simple_bayesian/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create simple test data (12 features, 50 time steps)
    test_data = torch.randn(1, 50, 12)
    
    return model, test_data, device


def predict_with_uncertainty(model, data, num_samples=30):
    """Get uncertainty predictions."""
    model.train()  # Enable dropout
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(data)
            predictions.append(pred[0].cpu().numpy())  # Remove batch dim
    
    predictions = np.array(predictions)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    lower_95 = np.percentile(predictions, 2.5, axis=0)
    upper_95 = np.percentile(predictions, 97.5, axis=0)
    
    return mean_pred, std_pred, lower_95, upper_95


def create_responsive_style_plot(mean_pred, std_pred, lower_95, upper_95, test_data):
    """Create 3-subplot visualization matching responsive model."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    seq_len = len(mean_pred)
    frames = np.arange(seq_len)
    
    # Subplot 1: Predictions with uncertainty
    axes[0].plot(frames, mean_pred, "b-", label="Bayesian Mean Prediction", linewidth=2)
    axes[0].fill_between(frames, mean_pred - std_pred, mean_pred + std_pred, 
                        alpha=0.3, color="blue", label="±1σ Uncertainty")
    axes[0].fill_between(frames, lower_95, upper_95, 
                        alpha=0.2, color="blue", label="95% Confidence")
    
    # Simulate ground truth
    ground_truth = 0.7  # Example: T team wins
    axes[0].axhline(y=ground_truth, color="r", linestyle="-", 
                   label="Ground Truth (1=T win, 0=CT win)", linewidth=2)
    
    axes[0].set_ylabel("Win Probability")
    axes[0].set_title("Bayesian LSTM - Uncertainty Estimation")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Subplot 2: Simulated player counts (features 0 and 1)
    ct_players = np.clip(np.abs(test_data[0, :seq_len, 0].numpy()) * 2.5, 0, 5)
    t_players = np.clip(np.abs(test_data[0, :seq_len, 1].numpy()) * 2.5, 0, 5)
    
    axes[1].plot(frames, ct_players, "b-", label="CT Players", linewidth=2.0)
    axes[1].plot(frames, t_players, "r-", label="T Players", linewidth=2.0)
    axes[1].set_ylabel("Players Alive")
    axes[1].set_ylim(0, 5)
    axes[1].legend()
    axes[1].grid(True)
    
    # Subplot 3: Simulated equipment values (features 2 and 3)
    ct_equipment = np.abs(test_data[0, :seq_len, 2].numpy()) * 5000
    t_equipment = np.abs(test_data[0, :seq_len, 3].numpy()) * 5000
    
    axes[2].plot(frames, ct_equipment, "b-", label="CT Equipment")
    axes[2].plot(frames, t_equipment, "r-", label="T Equipment")
    axes[2].set_ylabel("Equipment Value")
    axes[2].set_xlabel("Frame")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    print("🎯 BAYESIAN RESPONSIVE VISUALIZATION")
    print("=" * 50)
    
    # Load model and data
    model, test_data, device = load_model_and_data()
    
    # Get predictions with uncertainty
    print("Generating uncertainty predictions...")
    mean_pred, std_pred, lower_95, upper_95 = predict_with_uncertainty(model, test_data)
    
    # Create visualization
    print("Creating responsive-style visualization...")
    fig = create_responsive_style_plot(mean_pred, std_pred, lower_95, upper_95, test_data)
    
    # Save
    output_dir = Path("visualizations/bayesian_responsive")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bayesian_responsive_demo.png"
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    # Also show uncertainty stats
    final_pred = mean_pred[-1]
    final_std = std_pred[-1]
    print(f"\n📊 Final Prediction: {final_pred:.3f} ± {final_std:.3f}")
    print(f"   95% CI: [{lower_95[-1]:.3f}, {upper_95[-1]:.3f}]")
    
    plt.close(fig)
    print("\n🎊 Visualization complete!")


if __name__ == "__main__":
    main()
