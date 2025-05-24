#!/usr/bin/env python3
"""
Simple demo script to show uncertainty estimation with the Bayesian LSTM.
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pyro.infer import Predictive

from simple_bayesian_lstm import SimpleBayesianLSTM


def load_sample_data():
    """Load a sample of data for demonstration."""
    print("Loading sample data...")
    
    # Load the dataset 
    df = pd.read_csv('./round_frame_data1.csv')
    
    # Get first available match/round
    match_id = df['match_id'].iloc[0]
    round_idx = 0
    
    # Filter for this specific round
    round_data = df[(df['match_id'] == match_id) & (df['round_idx'] == round_idx)]
    
    if len(round_data) == 0:
        print("No data found, using first 100 rows")
        round_data = df.iloc[:100]
    
    # Select numerical features (excluding metadata)
    exclude_cols = ['match_id', 'round_idx', 'tick', 'rnd_winner', 'team1_score', 'team2_score']
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Use first 12 features to match expected input size
    if len(feature_cols) > 12:
        feature_cols = feature_cols[:12]
    elif len(feature_cols) < 12:
        # Pad with some of the excluded columns if needed
        extra_cols = ['team1_score', 'team2_score', 'tick']
        for col in extra_cols:
            if col in numerical_cols and len(feature_cols) < 12:
                feature_cols.append(col)
    
    X = round_data[feature_cols].values.astype(np.float32)
    
    # Pad to exactly 12 features if needed
    if X.shape[1] < 12:
        padding = np.zeros((X.shape[0], 12 - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, padding], axis=1)
    elif X.shape[1] > 12:
        X = X[:, :12]
    
    print(f"Loaded {len(X)} frames with {X.shape[1]} features")
    return torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension


def demonstrate_uncertainty(model, data, num_samples=10):
    """Demonstrate uncertainty estimation using dropout as approximation."""
    print(f"\nGenerating {num_samples} prediction samples...")
    
    # Enable training mode to activate dropout for uncertainty estimation
    model.train()  # This enables dropout for Monte Carlo sampling
    predictions = []
    
    # Generate multiple samples using dropout
    for i in range(num_samples):
        with torch.no_grad():
            pred = model(data)
            predictions.append(pred.squeeze().cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    print(f"Final prediction: {mean_pred[-1]:.3f} ± {std_pred[-1]:.3f}")
    print(f"Uncertainty range: {np.max(predictions[:, -1]) - np.min(predictions[:, -1]):.3f}")
    
    return predictions, mean_pred, std_pred


def create_visualization(predictions, mean_pred, std_pred):
    """Create uncertainty visualization."""
    print("Creating visualization...")
    
    frames = np.arange(len(mean_pred))
    
    plt.figure(figsize=(12, 6))
    
    # Plot individual samples (light lines)
    for pred in predictions[:5]:  # Show first 5 samples
        plt.plot(frames, pred, alpha=0.3, color='blue', linewidth=0.8)
    
    # Plot mean prediction
    plt.plot(frames, mean_pred, color='red', linewidth=2, label='Mean Prediction')
    
    # Plot confidence interval
    upper_bound = mean_pred + 1.96 * std_pred
    lower_bound = mean_pred - 1.96 * std_pred
    plt.fill_between(frames, lower_bound, upper_bound, alpha=0.3, color='red',
                     label='95% Confidence Interval')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Threshold')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Win Probability')
    plt.title('Simple Bayesian LSTM - Uncertainty Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = Path('./visualizations/simple_bayesian')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'uncertainty_demo.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_dir / 'uncertainty_demo.png'}")
    
    plt.show()


def main():
    """Main demonstration."""
    print("🎯 SIMPLE BAYESIAN LSTM DEMO")
    print("=" * 50)
    
    # Load trained model
    print("Loading trained model...")
    try:
        model_path = "./saved_models/simple_bayesian/best_model.pt"
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        model = SimpleBayesianLSTM(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load Pyro parameters
        if 'pyro_params' in checkpoint:
            pyro.get_param_store().set_state(checkpoint['pyro_params'])
        
        print("✅ Model loaded successfully!")
        print(f"   Input dim: {checkpoint['input_dim']}")
        print(f"   Hidden dim: {checkpoint['hidden_dim']}")
        print(f"   Layers: {checkpoint['num_layers']}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the model first using train_simple_bayesian.py")
        return
    
    # Load sample data
    data = load_sample_data()
    
    # Demonstrate uncertainty
    predictions, mean_pred, std_pred = demonstrate_uncertainty(model, data)
    
    # Create visualization
    create_visualization(predictions, mean_pred, std_pred)
    
    print("\n🎊 DEMO COMPLETE!")
    print("✅ Demonstrated:")
    print("   - Simple Bayesian LSTM loading")
    print("   - Uncertainty estimation via sampling") 
    print("   - Confidence interval visualization")


if __name__ == "__main__":
    main()
