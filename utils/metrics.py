"""
Shared utility functions for computing evaluation metrics.

All team members should use these functions to ensure consistent metric calculation.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import json


def calculate_detection_metrics(predictions, ground_truths, iou_thresholds=[0.5, 0.75]):
    """
    Calculate detection metrics (mAP) at different IoU thresholds.

    Args:
        predictions: List of predicted bounding boxes with scores and labels
        ground_truths: List of ground truth bounding boxes with labels
        iou_thresholds: List of IoU thresholds for mAP calculation

    Returns:
        dict: Dictionary containing mAP scores at different thresholds
    """
    # TODO: Implement mAP calculation
    # This should be implemented using COCO evaluation or similar
    raise NotImplementedError("Detection metrics calculation to be implemented")


def calculate_classification_metrics(y_true, y_pred, y_scores=None, class_names=None):
    """
    Calculate classification metrics: Precision, Recall, F1, Accuracy, AUC.

    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        y_scores: Prediction scores/probabilities for AUC (optional)
        class_names: List of class names for per-class metrics (optional)

    Returns:
        dict: Dictionary containing all classification metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate precision, recall, f1-score per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Calculate macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Calculate weighted-averaged metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculate AUC if scores are provided
    auc = None
    if y_scores is not None:
        try:
            # For multi-class, use one-vs-rest AUC
            from sklearn.preprocessing import label_binarize
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            auc = roc_auc_score(y_true_bin, y_scores, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            auc = None

    # Prepare per-class metrics
    per_class_metrics = {}
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(precision))]

    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }

    # Prepare overall metrics
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "auc": float(auc) if auc is not None else None,
        "per_class": per_class_metrics
    }

    return metrics


def calculate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Calculate confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names (optional)

    Returns:
        numpy.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def save_metrics(metrics, output_path):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def load_metrics(input_path):
    """
    Load metrics from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        dict: Dictionary of metrics
    """
    with open(input_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def print_metrics_summary(metrics):
    """
    Print a formatted summary of metrics.

    Args:
        metrics: Dictionary of metrics from calculate_classification_metrics()
    """
    print("=" * 60)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro):  {metrics['f1_macro']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC (macro):       {metrics['auc']:.4f}")
    print("=" * 60)

    print("\nPER-CLASS METRICS:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<20} {class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} {class_metrics['f1_score']:<12.4f}")
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Example data
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 1, 2, 0, 2, 2]
    y_scores = np.random.rand(9, 3)  # Mock probability scores for 3 classes

    class_names = ["Insect_A", "Insect_B", "Insect_C"]

    # Calculate metrics
    metrics = calculate_classification_metrics(y_true, y_pred, y_scores, class_names)

    # Print summary
    print_metrics_summary(metrics)

    # Save metrics
    save_metrics(metrics, "example_metrics.json")
