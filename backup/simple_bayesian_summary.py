#!/usr/bin/env python3
"""
Simple Bayesian LSTM - Complete Workflow Summary
================================================

This script demonstrates the complete simple Bayesian LSTM implementation that:
1. Wraps the existing LSTM model with Pyro for Bayesian training
2. Maintains the exact same output structure as the responsive model
3. Adds uncertainty estimation through Monte Carlo dropout
4. Provides the simplest possible Bayesian implementation

Usage:
    python simple_bayesian_summary.py
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

from simple_bayesian_lstm import SimpleBayesianLSTM


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")


def load_model_info():
    """Load and display model information."""
    print_header("SIMPLE BAYESIAN LSTM - MODEL INFO")
    
    # Load training results
    results_path = Path("saved_models/simple_bayesian/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        print("📊 Training Results:")
        print(f"   • Model Type: {results['model_params']['model_type']}")
        print(f"   • Input Dimension: {results['model_params']['input_dim']}")
        print(f"   • Hidden Dimension: {results['model_params']['hidden_dim']}")
        print(f"   • Number of Layers: {results['model_params']['num_layers']}")
        print(f"   • Dropout Rate: {results['model_params']['dropout']}")
        print(f"   • Training Time: {results['training_time']:.1f} seconds")
        print(f"   • Final Validation Loss: {results['best_val_loss']:.4f}")
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(results['training_history']['train_loss'], 'b-', label='Training Loss')
        plt.plot(results['training_history']['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final loss comparison
        plt.subplot(1, 2, 2)
        final_train = results['training_history']['train_loss'][-1]
        final_val = results['training_history']['val_loss'][-1]
        plt.bar(['Training', 'Validation'], [final_train, final_val], 
                color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Final Loss')
        plt.title('Final Training vs Validation Loss')
        
        # Save plot
        vis_dir = Path("visualizations/simple_bayesian")
        vis_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(vis_dir / "training_summary.png", dpi=150, bbox_inches='tight')
        print(f"   • Training curves saved to: {vis_dir / 'training_summary.png'}")
        plt.close()
    
    return results


def demonstrate_architecture():
    """Show the architecture comparison."""
    print_header("ARCHITECTURE COMPARISON")
    
    print("🏗️  Simple Bayesian LSTM Architecture:")
    print("   ┌─────────────────────────────────────┐")
    print("   │        SimpleBayesianLSTM           │")
    print("   │  ┌─────────────────────────────────┐ │")
    print("   │  │      LSTMWinPredictor           │ │")
    print("   │  │  ┌─────────────────────────────┐ │ │")
    print("   │  │  │    LSTM (64 hidden, 2 layer)│ │ │")
    print("   │  │  │    + Linear (64 -> 1)       │ │ │")
    print("   │  │  │    + Sigmoid activation     │ │ │")
    print("   │  │  └─────────────────────────────┘ │ │")
    print("   │  └─────────────────────────────────┘ │")
    print("   │           + Pyro probabilistic      │")
    print("   │           + Monte Carlo dropout     │")
    print("   └─────────────────────────────────────┘")
    print("\n✨ Key Features:")
    print("   • Same exact LSTM architecture as responsive model")
    print("   • Pyro wrapper for Bayesian training")
    print("   • Uncertainty estimation via dropout sampling")
    print("   • Identical output format and structure")


def run_quick_demo():
    """Run a quick uncertainty estimation demo."""
    print_header("UNCERTAINTY ESTIMATION DEMO")
    
    # Load model
    print("📦 Loading trained Bayesian model...")
    device = torch.device("cpu")
    
    # Load model parameters from training results
    results_path = Path("saved_models/simple_bayesian/training_results.json")
    with open(results_path) as f:
        results = json.load(f)
    
    params = results['model_params']
    model = SimpleBayesianLSTM(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    # Load trained weights
    checkpoint = torch.load("saved_models/simple_bayesian/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    pyro.get_param_store().load("saved_models/simple_bayesian/best_model.pt")
    
    print("✅ Model loaded successfully!")
    
    # Create sample data
    print("🎲 Generating sample data...")
    sample_data = torch.randn(1, 50, params['input_dim'])  # 1 batch, 50 timesteps, 12 features
    
    # Generate uncertainty estimates
    print("🔮 Estimating uncertainty...")
    model.train()  # Enable dropout
    predictions = []
    
    num_samples = 20
    for i in range(num_samples):
        with torch.no_grad():
            pred = model(sample_data)
            predictions.append(pred.squeeze().cpu().numpy())
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Display results
    print(f"📈 Results for {num_samples} Monte Carlo samples:")
    print(f"   • Final prediction: {mean_pred[-1]:.3f} ± {std_pred[-1]:.3f}")
    print(f"   • Uncertainty range: {np.max(predictions[:, -1]) - np.min(predictions[:, -1]):.3f}")
    print(f"   • Average uncertainty: {np.mean(std_pred):.3f}")
    print(f"   • Max uncertainty: {np.max(std_pred):.3f}")
    
    return predictions, mean_pred, std_pred


def create_summary_visualization(predictions, mean_pred, std_pred):
    """Create a comprehensive summary visualization."""
    print_header("CREATING SUMMARY VISUALIZATION")
    
    frames = np.arange(len(mean_pred))
    
    plt.figure(figsize=(15, 10))
    
    # Main uncertainty plot
    plt.subplot(2, 2, 1)
    for i, pred in enumerate(predictions[:10]):  # Show first 10 samples
        alpha = 0.6 if i < 5 else 0.3
        plt.plot(frames, pred, alpha=alpha, color='lightblue', linewidth=1)
    
    plt.plot(frames, mean_pred, color='red', linewidth=2, label='Mean Prediction')
    plt.fill_between(frames, mean_pred - std_pred, mean_pred + std_pred, 
                     alpha=0.3, color='red', label='±1 std')
    plt.fill_between(frames, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                     alpha=0.2, color='red', label='±2 std')
    plt.xlabel('Frame')
    plt.ylabel('Win Probability')
    plt.title('Bayesian LSTM Uncertainty Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty evolution
    plt.subplot(2, 2, 2)
    plt.plot(frames, std_pred, color='purple', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Standard Deviation')
    plt.title('Uncertainty Over Time')
    plt.grid(True, alpha=0.3)
    
    # Prediction distribution at final frame
    plt.subplot(2, 2, 3)
    final_preds = predictions[:, -1]
    plt.hist(final_preds, bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(mean_pred[-1], color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pred[-1]:.3f}')
    plt.axvline(mean_pred[-1] - std_pred[-1], color='orange', linestyle=':', label=f'-1σ: {mean_pred[-1] - std_pred[-1]:.3f}')
    plt.axvline(mean_pred[-1] + std_pred[-1], color='orange', linestyle=':', label=f'+1σ: {mean_pred[-1] + std_pred[-1]:.3f}')
    plt.xlabel('Final Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Final Prediction Distribution')
    plt.legend()
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    stats = {
        'Mean Prediction': mean_pred[-1],
        'Uncertainty (σ)': std_pred[-1],
        'Min Prediction': np.min(predictions[:, -1]),
        'Max Prediction': np.max(predictions[:, -1]),
        'Range': np.max(predictions[:, -1]) - np.min(predictions[:, -1])
    }
    
    y_pos = np.arange(len(stats))
    values = list(stats.values())
    plt.barh(y_pos, values, color=['blue', 'red', 'green', 'orange', 'purple'])
    plt.yticks(y_pos, list(stats.keys()))
    plt.xlabel('Value')
    plt.title('Summary Statistics')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    # Save visualization
    vis_dir = Path("visualizations/simple_bayesian")
    vis_dir.mkdir(parents=True, exist_ok=True)
    output_path = vis_dir / "complete_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary visualization saved to: {output_path}")
    plt.close()
    
    return output_path


def main():
    """Run the complete summary demonstration."""
    print("🎯 SIMPLE BAYESIAN LSTM - COMPLETE WORKFLOW SUMMARY")
    print("=" * 60)
    
    # Load and display model info
    results = load_model_info()
    
    # Show architecture
    demonstrate_architecture()
    
    # Run uncertainty demo
    predictions, mean_pred, std_pred = run_quick_demo()
    
    # Create comprehensive visualization
    output_path = create_summary_visualization(predictions, mean_pred, std_pred)
    
    # Final summary
    print_header("WORKFLOW COMPLETE! 🎊")
    print("✅ Successfully demonstrated:")
    print("   • Simple Bayesian LSTM implementation")
    print("   • Pyro-based probabilistic training")
    print("   • Monte Carlo uncertainty estimation")
    print("   • Same output structure as responsive model")
    print("   • Complete visualization of results")
    print(f"\n📁 All outputs saved to: visualizations/simple_bayesian/")
    print(f"🖼️  View summary: {output_path}")
    
    print("\n🎯 Key Achievement:")
    print("   Created the SIMPLEST possible Bayesian implementation")
    print("   that wraps existing LSTM without changing its output!")


if __name__ == "__main__":
    main()
