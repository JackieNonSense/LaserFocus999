"""
Unified evaluation script for all methods.

This script provides a standard way to evaluate any detection+classification method
to ensure fair comparison across the team.

Usage:
    python unified_evaluation.py --method faster-rcnn --results_path path/to/predictions
"""

import argparse
import json
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.metrics import (
    calculate_classification_metrics,
    calculate_detection_metrics,
    save_metrics,
    print_metrics_summary
)


def load_predictions(predictions_path):
    """
    Load predictions from file.

    Expected format: JSON file with structure:
    {
        "predictions": [
            {
                "image_id": "img_001.jpg",
                "boxes": [[x1, y1, x2, y2], ...],
                "labels": [0, 1, ...],
                "scores": [0.95, 0.87, ...]
            },
            ...
        ]
    }
    """
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    return data['predictions']


def load_ground_truths(gt_path):
    """
    Load ground truth annotations.

    Expected format: JSON file with COCO-style annotations
    """
    with open(gt_path, 'r') as f:
        data = json.load(f)
    return data


def evaluate_method(predictions, ground_truths, class_names=None):
    """
    Evaluate a method using standard metrics.

    Args:
        predictions: List of predictions
        ground_truths: Ground truth annotations
        class_names: List of class names

    Returns:
        dict: Complete evaluation results
    """
    results = {}

    # 1. Detection metrics (mAP)
    print("Calculating detection metrics...")
    try:
        detection_metrics = calculate_detection_metrics(predictions, ground_truths)
        results['detection'] = detection_metrics
    except NotImplementedError:
        print("Warning: Detection metrics not implemented yet")
        results['detection'] = None

    # 2. Classification metrics
    print("Calculating classification metrics...")

    # Extract labels from predictions and ground truths
    y_true = []
    y_pred = []
    y_scores = []

    for pred, gt in zip(predictions, ground_truths):
        # TODO: Match predictions to ground truths based on IoU
        # For now, assume they are aligned
        if 'labels' in pred and 'labels' in gt:
            y_true.extend(gt['labels'])
            y_pred.extend(pred['labels'])
            if 'scores' in pred:
                y_scores.extend(pred['scores'])

    classification_metrics = calculate_classification_metrics(
        y_true, y_pred,
        y_scores=y_scores if y_scores else None,
        class_names=class_names
    )
    results['classification'] = classification_metrics

    return results


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation for all methods')
    parser.add_argument('--method', type=str, required=True,
                       help='Method name (e.g., faster-rcnn, yolo)')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--ground_truths', type=str, required=True,
                       help='Path to ground truth annotations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results JSON')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='List of class names')

    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)

    print(f"Loading ground truths from {args.ground_truths}...")
    ground_truths = load_ground_truths(args.ground_truths)

    # Evaluate
    print(f"\nEvaluating {args.method}...")
    results = evaluate_method(predictions, ground_truths, args.class_names)

    # Print summary
    if results['classification']:
        print_metrics_summary(results['classification'])

    # Save results
    if args.output is None:
        args.output = f"../results/{args.method}_results.json"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_metrics(results, str(output_path))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
