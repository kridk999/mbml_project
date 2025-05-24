#!/usr/bin/env python3
"""
Quick Bayesian LSTM Prediction Demo
===================================

A simplified script that shows the key concepts of Bayesian prediction and visualization.
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

from simple_bayesian_lstm import SimpleBayesianLSTM


def quick_demo():
    """Quick demonstration of Bayesian prediction."""
    print("🎯 QUICK BAYESIAN LSTM DEMO")
    print("=" * 40)
    
    # Load model parameters
    results_path = Path("saved_models/simple_bayesian/training_results.json")
    with open(results_path) as f:
        results = json.load(f)
    
    params = results['model_params']
    print(f"📊 Model: {params['input_dim']} → {params['hidden_dim']} → 1")
    
    # Create and load model
    model = SimpleBayesianLSTM(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    checkpoint = torch.load("saved_models/simple_bayesian/best_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    pyro.get_param_store().load("saved_models/simple_bayesian/best_model.pt")
    print("✅ Model loaded!")
    
    # Generate sample data
    print("🎲 Generating sample data...")
    sample_length = 30
    sample_data = torch.randn(1, sample_length, params['input_dim'])
    
    # Make predictions with uncertainty
    print("🔮 Making predictions with uncertainty...")
    model.train()  # Enable dropout for uncertainty
    
    num_samples = 20
    predictions = []
    
    for i in range(num_samples):
        with torch.no_grad():
            pred = model(sample_data)
            predictions.append(pred.squeeze().cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    print(f"📈 Results:")
    print(f"   • Final prediction: {mean_pred[-1]:.3f} ± {std_pred[-1]:.3f}")
    print(f"   • Uncertainty range: {np.max(predictions[:, -1]) - np.min(predictions[:, -1]):.3f}")
    
    # Create visualization
    print("🎨 Creating visualization...")
    frames = np.arange(len(mean_pred))
    
    plt.figure(figsize=(14, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    # Show some individual samples
    for i in range(min(10, num_samples)):
        plt.plot(frames, predictions[i], alpha=0.3, color='lightblue', linewidth=1)
    
    plt.plot(frames, mean_pred, color='red', linewidth=2, label='Mean Prediction')
    plt.fill_between(frames, mean_pred - std_pred, mean_pred + std_pred, 
                     alpha=0.3, color='red', label='±1σ')
    plt.fill_between(frames, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                     alpha=0.2, color='red', label='±2σ')
    
    plt.xlabel('Frame')
    plt.ylabel('Win Probability')
    plt.title('Bayesian LSTM Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty over time
    plt.subplot(2, 2, 2)
    plt.plot(frames, std_pred, color='purple', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Standard Deviation')
    plt.title('Prediction Uncertainty Over Time')
    plt.grid(True, alpha=0.3)
    
    # Final prediction distribution
    plt.subplot(2, 2, 3)
    final_preds = predictions[:, -1]
    plt.hist(final_preds, bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(mean_pred[-1], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_pred[-1]:.3f}')
    plt.xlabel('Final Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Final Prediction Distribution')
    plt.legend()
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    stats = {
        'Mean': mean_pred[-1],
        'Std Dev': std_pred[-1],
        'Min': np.min(predictions[:, -1]),
        'Max': np.max(predictions[:, -1])
    }
    
    bars = plt.bar(range(len(stats)), list(stats.values()), 
                   color=['blue', 'red', 'green', 'orange'])
    plt.xticks(range(len(stats)), list(stats.keys()))
    plt.ylabel('Value')
    plt.title('Summary Statistics')
    
    # Add value labels on bars
    for bar, val in zip(bars, stats.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualization
    vis_dir = Path("visualizations/simple_bayesian")
    vis_dir.mkdir(parents=True, exist_ok=True)
    output_path = vis_dir / "quick_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_path}")
    plt.close()
    
    # Show training history
    print("\n📊 Training Performance:")
    train_history = results['training_history']
    print(f"   • Training time: {results['training_time']:.1f}s")
    print(f"   • Final train loss: {train_history['train_loss'][-1]:.1f}")
    print(f"   • Final val loss: {train_history['val_loss'][-1]:.3f}")
    
    print("\n🎊 DEMO COMPLETE!")
    print(f"📁 Output saved to: {output_path}")


if __name__ == "__main__":
    quick_demo()
