"""
This script evaluates a trained LSTM model on CS:GO round data and produces
detailed performance metrics and visualizations.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds
from mbml.model import LSTMWinPredictor
from mbml.predict import load_model


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", output_path=None):
    """Plot a confusion matrix with seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels or ["Lose", "Win"],
        yticklabels=labels or ["Lose", "Win"],
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Confusion matrix saved to {output_path}")

    return plt.gcf()


def plot_roc_curve(y_true, y_score, output_path=None):
    """Plot the ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"ROC curve saved to {output_path}")

    return plt.gcf(), roc_auc


def plot_precision_recall_curve(y_true, y_score, output_path=None):
    """Plot the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.3f})")
    plt.axhline(y=sum(y_true) / len(y_true), color="red", linestyle="--", label="Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"PR curve saved to {output_path}")

    return plt.gcf(), pr_auc


def plot_threshold_impact(y_true, y_score, output_path=None):
    """Plot the impact of different thresholds on precision, recall and F1."""
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, "b-", label="Precision")
    plt.plot(thresholds, recalls, "r-", label="Recall")
    plt.plot(thresholds, f1_scores, "g-", label="F1 Score")
    plt.plot(thresholds, accuracies, "y-", label="Accuracy")

    # Find optimal threshold for F1
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]

    plt.axvline(
        x=best_threshold, color="k", linestyle="--", label=f"Best Threshold = {best_threshold:.2f} (F1 = {best_f1:.3f})"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Impact of Threshold on Classification Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Threshold impact plot saved to {output_path}")

    return plt.gcf(), best_threshold


def evaluate_model(model, data_loader, device, output_dir=None, threshold=0.5):
    """
    Evaluate a trained model on a dataset and generate performance metrics.

    Args:
        model: The trained model
        data_loader: DataLoader for the evaluation data
        device: Device to use for inference
        output_dir: Directory to save evaluation results
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary of evaluation metrics
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_preds = []
    all_labels = []
    all_masks = []
    all_metadata = []

    # Collect predictions and labels
    print("Collecting model predictions...")
    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            labels = batch["labels"]
            mask = batch["mask"]
            metadata = batch["metadata"]

            predictions = model(features).cpu()

            all_preds.append(predictions)
            all_labels.append(labels)
            all_masks.append(mask)
            all_metadata.extend(metadata)

    # Concatenate tensors
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Filter out padding
    valid_indices = all_masks > 0
    valid_preds = all_preds[valid_indices].numpy()
    valid_labels = all_labels[valid_indices].numpy()

    # Binary predictions
    binary_preds = (valid_preds >= threshold).astype(int)

    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = {
        "accuracy": accuracy_score(valid_labels, binary_preds),
        "precision": precision_score(valid_labels, binary_preds),
        "recall": recall_score(valid_labels, binary_preds),
        "f1": f1_score(valid_labels, binary_preds),
    }

    # Generate plots
    if output_dir:
        print("Generating evaluation plots...")

        # ROC curve
        _, roc_auc = plot_roc_curve(valid_labels, valid_preds, output_path=output_dir / "roc_curve.png")
        metrics["roc_auc"] = roc_auc

        # Precision-Recall curve
        _, pr_auc = plot_precision_recall_curve(valid_labels, valid_preds, output_path=output_dir / "pr_curve.png")
        metrics["pr_auc"] = pr_auc

        # Confusion matrix
        plot_confusion_matrix(valid_labels, binary_preds, output_path=output_dir / "confusion_matrix.png")

        # Threshold impact
        _, best_threshold = plot_threshold_impact(
            valid_labels, valid_preds, output_path=output_dir / "threshold_impact.png"
        )
        metrics["best_threshold"] = best_threshold

        # Save metrics to file
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)

        print(f"All evaluation results saved to {output_dir}")

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics.get('roc_auc', 'Not calculated')}")
    print(f"PR AUC:    {metrics.get('pr_auc', 'Not calculated')}")

    if "best_threshold" in metrics:
        print(f"Best Threshold: {metrics['best_threshold']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the data CSV")
    parser.add_argument("--output_dir", type=str, default="./reports/evaluation", help="Directory for outputs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension (if config unavailable)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (if config unavailable)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    config_path = str(Path(args.model_path).parent / "training_results.json")
    model = load_model(args.model_path, config_path, device)

    # Create dataset & dataloader
    print(f"Loading data from {args.csv_path}")
    dataset = CSGORoundDataset(csv_path=args.csv_path)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_variable_length_rounds
    )

    # Evaluate the model
    evaluate_model(
        model=model, data_loader=data_loader, device=device, output_dir=args.output_dir, threshold=args.threshold
    )


if __name__ == "__main__":
    main()
