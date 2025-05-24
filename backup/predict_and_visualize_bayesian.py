#!/usr/bin/env python3
"""
Bayesian LSTM Prediction and Visualization Script
=================================================

This script demonstrates how to:
1. Load a trained Bayesian LSTM model
2. Make predictions with uncertainty estimation
3. Create comprehensive visualizations
4. Compare predictions with ground truth

Usage:
    python predict_and_visualize_bayesian.py
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm

from simple_bayesian_lstm import SimpleBayesianLSTM
from src.mbml.dataset import CSGORoundDataset


def load_trained_model(model_path="saved_models/simple_bayesian"):
    """Load the trained Bayesian LSTM model."""
    print("🚀 Loading trained Bayesian LSTM model...")
    
    # Load training results to get model parameters
    results_path = Path(model_path) / "training_results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    params = results['model_params']
    print(f"   • Input dim: {params['input_dim']}")
    print(f"   • Hidden dim: {params['hidden_dim']}")
    print(f"   • Layers: {params['num_layers']}")
    print(f"   • Dropout: {params['dropout']}")
    
    # Create model
    model = SimpleBayesianLSTM(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    # Load trained weights
    device = torch.device("cpu")
    checkpoint = torch.load(Path(model_path) / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    pyro.get_param_store().load(str(Path(model_path) / "best_model.pt"))
    
    print("✅ Model loaded successfully!")
    return model, params


def load_test_data(csv_path="round_frame_data.csv", num_rounds=5):
    """Load test data for prediction."""
    print(f"📊 Loading test data from {csv_path}...")
    
    # Load dataset
    dataset = CSGORoundDataset(csv_path, split="test")
    
    # Get a few rounds for testing
    test_rounds = []
    labels = []
    
    for i in range(min(num_rounds, len(dataset))):
        round_data = dataset[i]
        test_rounds.append(round_data["features"])
        labels.append(round_data["labels"])
    
    print(f"✅ Loaded {len(test_rounds)} test rounds")
    return test_rounds, labels


def predict_with_uncertainty(model, data, num_samples=50):
    """Make predictions with uncertainty estimation."""
    print(f"🔮 Generating {num_samples} prediction samples...")
    
    # Enable dropout for uncertainty estimation
    model.train()
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Sampling"):
            pred = model(data)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    min_pred = np.min(predictions, axis=0)
    max_pred = np.max(predictions, axis=0)
    
    return {
        'predictions': predictions,
        'mean': mean_pred,
        'std': std_pred,
        'min': min_pred,
        'max': max_pred
    }


def create_uncertainty_visualization(results, ground_truth=None, round_idx=0):
    """Create comprehensive uncertainty visualization."""
    print("🎨 Creating uncertainty visualization...")
    
    predictions = results['predictions']
    mean_pred = results['mean'][round_idx]
    std_pred = results['std'][round_idx]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Bayesian LSTM Uncertainty Analysis - Round {round_idx + 1}', fontsize=16, fontweight='bold')
    
    frames = np.arange(len(mean_pred))
    
    # 1. Main prediction with uncertainty bands
    ax1 = axes[0, 0]
    for i in range(min(10, len(predictions))):
        ax1.plot(frames, predictions[i, round_idx], alpha=0.3, color='lightblue', linewidth=1)
    
    ax1.plot(frames, mean_pred, color='red', linewidth=2, label='Mean Prediction')
    ax1.fill_between(frames, mean_pred - std_pred, mean_pred + std_pred, 
                     alpha=0.3, color='red', label='±1σ')
    ax1.fill_between(frames, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                     alpha=0.2, color='red', label='±2σ')
    
    if ground_truth is not None:
        ax1.plot(frames, ground_truth[round_idx], color='green', linewidth=2, 
                linestyle='--', label='Ground Truth')
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Win Probability')
    ax1.set_title('Prediction with Uncertainty Bands')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Uncertainty evolution
    ax2 = axes[0, 1]
    ax2.plot(frames, std_pred, color='purple', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Uncertainty Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Final prediction distribution
    ax3 = axes[0, 2]
    final_predictions = predictions[:, round_idx, -1]
    ax3.hist(final_predictions, bins=20, alpha=0.7, color='green', edgecolor='black', density=True)
    ax3.axvline(mean_pred[-1], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_pred[-1]:.3f}')
    
    if ground_truth is not None:
        ax3.axvline(ground_truth[round_idx][-1], color='orange', linestyle=':', linewidth=2,
                   label=f'Truth: {ground_truth[round_idx][-1]:.3f}')
    
    ax3.set_xlabel('Final Prediction')
    ax3.set_ylabel('Density')
    ax3.set_title('Final Prediction Distribution')
    ax3.legend()
    
    # 4. Prediction confidence intervals
    ax4 = axes[1, 0]
    confidence_levels = [0.5, 0.68, 0.95, 0.99]
    colors = ['darkred', 'red', 'orange', 'yellow']
    
    for conf, color in zip(confidence_levels, colors):
        lower = np.percentile(predictions[:, round_idx], (1-conf)*50, axis=0)
        upper = np.percentile(predictions[:, round_idx], (1+conf)*50, axis=0)
        ax4.fill_between(frames, lower, upper, alpha=0.4, color=color, 
                        label=f'{conf*100:.0f}% CI')
    
    ax4.plot(frames, mean_pred, color='black', linewidth=2, label='Mean')
    if ground_truth is not None:
        ax4.plot(frames, ground_truth[round_idx], color='green', linewidth=2, 
                linestyle='--', label='Ground Truth')
    
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Win Probability')
    ax4.set_title('Confidence Intervals')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Uncertainty heatmap
    ax5 = axes[1, 1]
    if len(predictions) > 1:
        # Show uncertainty across multiple rounds
        uncertainty_matrix = results['std'][:min(5, len(results['std']))]  # First 5 rounds
        im = ax5.imshow(uncertainty_matrix, aspect='auto', cmap='viridis')
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Round')
        ax5.set_title('Uncertainty Heatmap')
        plt.colorbar(im, ax=ax5, label='Standard Deviation')
    
    # 6. Statistics summary
    ax6 = axes[1, 2]
    stats = {
        'Mean Final': f'{mean_pred[-1]:.3f}',
        'Std Final': f'{std_pred[-1]:.3f}',
        'Min Final': f'{results["min"][round_idx][-1]:.3f}',
        'Max Final': f'{results["max"][round_idx][-1]:.3f}',
        'Range': f'{results["max"][round_idx][-1] - results["min"][round_idx][-1]:.3f}'
    }
    
    y_pos = np.arange(len(stats))
    values = [float(v) for v in stats.values()]
    bars = ax6.barh(y_pos, values, color=['blue', 'red', 'green', 'orange', 'purple'])
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(list(stats.keys()))
    ax6.set_xlabel('Value')
    ax6.set_title('Summary Statistics')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax6.text(val + 0.01, i, f'{val:.3f}', va='center')
    
    plt.tight_layout()
    return fig


def compare_predictions_vs_truth(results, ground_truth):
    """Compare Bayesian predictions with ground truth."""
    print("📊 Comparing predictions with ground truth...")
    
    mean_predictions = results['mean']
    std_predictions = results['std']
    
    # Calculate metrics for each round
    metrics = []
    
    for i, (pred_mean, pred_std, truth) in enumerate(zip(mean_predictions, std_predictions, ground_truth)):
        # Only compare where we have ground truth data
        valid_mask = ~np.isnan(truth)
        if not np.any(valid_mask):
            continue
            
        pred_mean_valid = pred_mean[valid_mask]
        pred_std_valid = pred_std[valid_mask]
        truth_valid = truth[valid_mask]
        
        # Calculate metrics
        mae = np.mean(np.abs(pred_mean_valid - truth_valid))
        rmse = np.sqrt(np.mean((pred_mean_valid - truth_valid) ** 2))
        
        # Check if ground truth falls within confidence intervals
        lower_95 = pred_mean_valid - 1.96 * pred_std_valid
        upper_95 = pred_mean_valid + 1.96 * pred_std_valid
        coverage_95 = np.mean((truth_valid >= lower_95) & (truth_valid <= upper_95))
        
        lower_68 = pred_mean_valid - pred_std_valid
        upper_68 = pred_mean_valid + pred_std_valid
        coverage_68 = np.mean((truth_valid >= lower_68) & (truth_valid <= upper_68))
        
        metrics.append({
            'round': i,
            'mae': mae,
            'rmse': rmse,
            'coverage_68': coverage_68,
            'coverage_95': coverage_95,
            'mean_uncertainty': np.mean(pred_std_valid)
        })
    
    # Create comparison visualization
    if metrics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Bayesian LSTM Performance Analysis', fontsize=14, fontweight='bold')
        
        rounds = [m['round'] for m in metrics]
        
        # MAE and RMSE
        ax1 = axes[0, 0]
        mae_values = [m['mae'] for m in metrics]
        rmse_values = [m['rmse'] for m in metrics]
        ax1.plot(rounds, mae_values, 'bo-', label='MAE')
        ax1.plot(rounds, rmse_values, 'ro-', label='RMSE')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Error')
        ax1.set_title('Prediction Errors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Coverage
        ax2 = axes[0, 1]
        cov_68 = [m['coverage_68'] for m in metrics]
        cov_95 = [m['coverage_95'] for m in metrics]
        ax2.plot(rounds, cov_68, 'go-', label='68% CI')
        ax2.plot(rounds, cov_95, 'mo-', label='95% CI')
        ax2.axhline(0.68, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(0.95, color='magenta', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Coverage')
        ax2.set_title('Confidence Interval Coverage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Uncertainty
        ax3 = axes[1, 0]
        uncertainties = [m['mean_uncertainty'] for m in metrics]
        ax3.plot(rounds, uncertainties, 'ko-')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Mean Uncertainty')
        ax3.set_title('Uncertainty Levels')
        ax3.grid(True, alpha=0.3)
        
        # Summary metrics
        ax4 = axes[1, 1]
        avg_metrics = {
            'Avg MAE': np.mean(mae_values),
            'Avg RMSE': np.mean(rmse_values),
            'Avg 68% Cov': np.mean(cov_68),
            'Avg 95% Cov': np.mean(cov_95),
            'Avg Uncertainty': np.mean(uncertainties)
        }
        
        y_pos = np.arange(len(avg_metrics))
        values = list(avg_metrics.values())
        bars = ax4.barh(y_pos, values, color=['blue', 'red', 'green', 'magenta', 'orange'])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(list(avg_metrics.keys()))
        ax4.set_xlabel('Value')
        ax4.set_title('Average Metrics')
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax4.text(val + 0.01, i, f'{val:.3f}', va='center')
        
        plt.tight_layout()
        return fig, metrics
    
    return None, metrics


def main():
    """Main prediction and visualization workflow."""
    print("🎯 BAYESIAN LSTM PREDICTION & VISUALIZATION")
    print("=" * 60)
    
    # Load trained model
    model, params = load_trained_model()
    
    # Load test data
    test_rounds, ground_truth = load_test_data(num_rounds=3)
    
    # Make predictions with uncertainty for each round
    all_results = []
    
    for i, round_data in enumerate(test_rounds):
        print(f"\n🔮 Processing Round {i + 1}/{len(test_rounds)}")
        
        # Add batch dimension
        input_data = round_data.unsqueeze(0)
        
        # Get predictions
        results = predict_with_uncertainty(model, input_data, num_samples=30)
        all_results.append(results)
        
        # Create visualization for this round
        fig = create_uncertainty_visualization(results, ground_truth, round_idx=0)
        
        # Save visualization
        vis_dir = Path("visualizations/simple_bayesian")
        vis_dir.mkdir(parents=True, exist_ok=True)
        output_path = vis_dir / f"round_{i+1}_uncertainty.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved: {output_path}")
        plt.close(fig)
    
    # Combine results for comparison
    combined_results = {
        'mean': np.array([r['mean'][0] for r in all_results]),
        'std': np.array([r['std'][0] for r in all_results]),
        'min': np.array([r['min'][0] for r in all_results]),
        'max': np.array([r['max'][0] for r in all_results])
    }
    
    # Compare with ground truth
    comparison_fig, metrics = compare_predictions_vs_truth(combined_results, ground_truth)
    if comparison_fig:
        comparison_path = vis_dir / "performance_analysis.png"
        comparison_fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Performance analysis saved: {comparison_path}")
        plt.close(comparison_fig)
    
    # Print summary
    print("\n🎊 PREDICTION & VISUALIZATION COMPLETE!")
    print("=" * 60)
    print("✅ Generated:")
    print(f"   • {len(test_rounds)} individual round uncertainty visualizations")
    print("   • Performance analysis comparing predictions vs ground truth")
    print("   • Comprehensive uncertainty quantification")
    
    if metrics:
        avg_mae = np.mean([m['mae'] for m in metrics])
        avg_coverage_95 = np.mean([m['coverage_95'] for m in metrics])
        print(f"\n📊 Average Performance:")
        print(f"   • MAE: {avg_mae:.3f}")
        print(f"   • 95% Coverage: {avg_coverage_95:.1%}")
    
    print(f"\n📁 All visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
