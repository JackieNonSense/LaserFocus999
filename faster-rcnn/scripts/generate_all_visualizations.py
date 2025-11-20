"""
Complete visualization script for Faster R-CNN results.
Generates all figures needed for the report section 4.2.

Generates:
1. Per-class AP bar chart
2. Confusion matrix
3. Precision-Recall curves for each class
4. Sample prediction visualizations

Usage:
    python scripts/generate_all_visualizations.py \
        --results faster-rcnn-results/results.json \
        --predictions faster-rcnn-results/coco_instances_results.json \
        --coco-json outputs/coco_annotations/test_coco.json \
        --image-dir /path/to/test/images \
        --output-dir outputs/report_figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import cv2

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Class names mapping (insect_class_0 to insect_class_11)
CLASS_NAMES = [
    "Ants",          # 0
    "Bees",          # 1
    "Beetles",       # 2
    "Caterpillars",  # 3
    "Earthworms",    # 4
    "Earwigs",       # 5
    "Grasshoppers",  # 6
    "Moths",         # 7
    "Slugs",         # 8
    "Snails",        # 9
    "Wasps",         # 10
    "Weevils"        # 11
]

COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 0),      # Dark Blue
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Dark Red
    (128, 128, 0),    # Dark Cyan
    (128, 0, 128),    # Dark Magenta
    (0, 128, 128),    # Dark Yellow
]


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_per_class_ap(results, output_path):
    """
    Plot per-class Average Precision as a bar chart.

    Args:
        results: Dictionary containing AP metrics
        output_path: Path to save the figure
    """
    # Extract per-class AP values
    ap_values = []
    class_labels = []

    for i in range(12):
        key = f"AP-insect_class_{i}"
        if key in results['bbox']:
            ap_value = results['bbox'][key]
            # Handle NaN values
            if isinstance(ap_value, (int, float)) and not np.isnan(ap_value):
                ap_values.append(ap_value)
            else:
                ap_values.append(0.0)
            class_labels.append(CLASS_NAMES[i])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars
    x_pos = np.arange(len(class_labels))
    bars = ax.bar(x_pos, ap_values, color='steelblue', alpha=0.8, edgecolor='black')

    # Color bars based on performance
    for i, (bar, val) in enumerate(zip(bars, ap_values)):
        if val >= 50:
            bar.set_color('forestgreen')
        elif val >= 30:
            bar.set_color('orange')
        else:
            bar.set_color('crimson')

    # Customize plot
    ax.set_xlabel('Insect Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Precision (AP)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Average Precision - Faster R-CNN', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(ap_values):
        ax.text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # Add mean AP line
    mean_ap = np.mean(ap_values)
    ax.axhline(y=mean_ap, color='red', linestyle='--', linewidth=2,
               label=f'Mean AP: {mean_ap:.2f}')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-class AP chart to {output_path}")
    plt.close()


def compute_confusion_matrix(predictions, coco_data, output_path, iou_threshold=0.5):
    """
    Compute and plot confusion matrix.

    Args:
        predictions: List of prediction dictionaries
        coco_data: COCO format annotations
        output_path: Path to save the figure
        iou_threshold: IoU threshold for matching predictions to ground truth
    """
    # Group predictions by image
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)

    # Group ground truth by image
    gt_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        gt_by_image[ann['image_id']].append(ann)

    y_true = []
    y_pred = []

    # Match predictions to ground truth
    for img_id in gt_by_image.keys():
        gt_anns = gt_by_image[img_id]
        preds = pred_by_image.get(img_id, [])

        # Match each GT to best prediction
        for gt in gt_anns:
            gt_class = gt['category_id']
            gt_box = gt['bbox']

            best_iou = 0
            best_pred_class = None

            # Find best matching prediction
            for pred in preds:
                iou = compute_iou(gt_box, pred['bbox'])
                if iou > best_iou and pred['score'] >= 0.5:
                    best_iou = iou
                    best_pred_class = pred['category_id']

            if best_iou >= iou_threshold and best_pred_class is not None:
                y_true.append(gt_class)
                y_pred.append(best_pred_class)
            else:
                # Missed detection
                y_true.append(gt_class)
                y_pred.append(-1)  # Placeholder for missed

    # Create confusion matrix (excluding missed detections for cleaner visualization)
    valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if len(y_true_valid) == 0:
        print("Warning: No valid predictions for confusion matrix")
        return

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(12)))

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax,
                square=True, linewidths=0.5)

    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Faster R-CNN (Normalized)', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {output_path}")
    plt.close()


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Compute intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_pr_curves(predictions, coco_data, output_path):
    """
    Compute and plot Precision-Recall curves for each class.

    Args:
        predictions: List of prediction dictionaries
        coco_data: COCO format annotations
        output_path: Path to save the figure
    """
    # Group predictions and GT by class
    pred_by_class = defaultdict(list)
    gt_count_by_class = defaultdict(int)

    # Count ground truth instances per class
    for ann in coco_data.get('annotations', []):
        gt_count_by_class[ann['category_id']] += 1

    # Group predictions by image and class
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)
        pred_by_class[pred['category_id']].append(pred)

    # Group GT by image
    gt_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        gt_by_image[ann['image_id']].append(ann)

    # Compute PR curve for each class
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for class_id in range(12):
        ax = axes[class_id]

        # Get all predictions for this class
        class_preds = [p for p in predictions if p['category_id'] == class_id]

        if len(class_preds) == 0 or gt_count_by_class[class_id] == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{CLASS_NAMES[class_id]} (AP: N/A)')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # Sort by confidence
        class_preds.sort(key=lambda x: x['score'], reverse=True)

        # Compute precision and recall at each threshold
        precisions = []
        recalls = []

        tp = 0
        fp = 0
        matched_gt = set()

        for pred in class_preds:
            img_id = pred['image_id']
            pred_box = pred['bbox']

            # Find best matching GT box
            best_iou = 0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gt_by_image[img_id]):
                if gt['category_id'] == class_id:
                    gt_key = (img_id, gt_idx)
                    if gt_key not in matched_gt:
                        iou = compute_iou(pred_box, gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_key

            if best_iou >= 0.5 and best_gt_idx is not None:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / gt_count_by_class[class_id]

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            prec_at_recall = [p for r, p in zip(recalls, precisions) if r >= t]
            if len(prec_at_recall) > 0:
                ap += max(prec_at_recall) / 11

        # Plot PR curve
        ax.plot(recalls, precisions, linewidth=2, color='darkblue')
        ax.fill_between(recalls, 0, precisions, alpha=0.2, color='skyblue')
        ax.set_title(f'{CLASS_NAMES[class_id]}\n(AP: {ap*100:.1f}%)', fontsize=10)
        ax.set_xlabel('Recall', fontsize=9)
        ax.set_ylabel('Precision', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('Precision-Recall Curves - Faster R-CNN', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PR curves to {output_path}")
    plt.close()


def visualize_predictions(predictions, coco_data, image_dir, output_dir, num_samples=10):
    """
    Visualize sample predictions.

    Args:
        predictions: List of prediction dictionaries
        coco_data: COCO format annotations
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Group predictions and GT by image
    pred_by_image = defaultdict(list)
    for pred in predictions:
        if pred['score'] >= 0.5:
            pred_by_image[pred['image_id']].append(pred)

    gt_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        gt_by_image[ann['image_id']].append(ann)

    # Image ID to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}

    # Select diverse samples: high confidence and low confidence
    image_ids = list(pred_by_image.keys())

    # Compute average confidence per image
    avg_conf = {}
    for img_id in image_ids:
        preds = pred_by_image[img_id]
        if len(preds) > 0:
            avg_conf[img_id] = sum(p['score'] for p in preds) / len(preds)
        else:
            avg_conf[img_id] = 0

    # Sort by confidence
    sorted_ids = sorted(avg_conf.keys(), key=lambda x: avg_conf[x], reverse=True)

    # Select samples: half high-conf, half low-conf
    half = num_samples // 2
    selected_ids = sorted_ids[:half] + sorted_ids[-half:]

    print(f"\nGenerating {len(selected_ids)} prediction visualizations...")

    for idx, img_id in enumerate(selected_ids[:num_samples]):
        img_info = image_info[img_id]
        img_path = Path(image_dir) / img_info['file_name']

        if not img_path.exists():
            print(f"  Warning: Image not found: {img_path}")
            continue

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Draw predictions
        preds = pred_by_image[img_id]
        for pred in preds:
            x, y, w, h = [int(v) for v in pred['bbox']]
            cat_id = pred['category_id']
            score = pred['score']

            color = COLORS[cat_id % len(COLORS)]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Label
            label = f"{CLASS_NAMES[cat_id]}: {score:.2f}"
            cv2.putText(image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save
        output_path = Path(output_dir) / f"pred_sample_{idx:02d}_{img_info['file_name']}"
        cv2.imwrite(str(output_path), image)
        print(f"  [{idx+1}/{len(selected_ids)}] Saved {output_path.name}")

    print(f"✓ Saved prediction visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate all visualizations for Faster R-CNN")
    parser.add_argument("--results", required=True,
                       help="Path to results.json file")
    parser.add_argument("--predictions", required=True,
                       help="Path to coco_instances_results.json file")
    parser.add_argument("--coco-json", required=True,
                       help="Path to COCO format test annotations")
    parser.add_argument("--image-dir", required=True,
                       help="Directory containing test images")
    parser.add_argument("--output-dir", default="outputs/report_figures",
                       help="Directory to save all figures")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of prediction samples to visualize")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("FASTER R-CNN VISUALIZATION GENERATION")
    print("="*60)

    # Load data
    print("\n1. Loading data...")
    results = load_json(args.results)
    predictions = load_json(args.predictions)
    coco_data = load_json(args.coco_json)
    print(f"   - Loaded {len(predictions)} predictions")
    print(f"   - Loaded {len(coco_data['images'])} images")
    print(f"   - Loaded {len(coco_data.get('annotations', []))} ground truth annotations")

    # Generate visualizations
    print("\n2. Generating per-class AP bar chart...")
    plot_per_class_ap(results, output_dir / "per_class_ap.png")

    print("\n3. Generating confusion matrix...")
    compute_confusion_matrix(predictions, coco_data, output_dir / "confusion_matrix.png")

    print("\n4. Generating Precision-Recall curves...")
    compute_pr_curves(predictions, coco_data, output_dir / "pr_curves.png")

    print("\n5. Generating prediction visualizations...")
    vis_dir = output_dir / "prediction_samples"
    visualize_predictions(predictions, coco_data, args.image_dir, vis_dir, args.num_samples)

    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. per_class_ap.png          - Per-class AP bar chart")
    print(f"  2. confusion_matrix.png      - Confusion matrix")
    print(f"  3. pr_curves.png             - Precision-Recall curves")
    print(f"  4. prediction_samples/       - Sample visualizations ({args.num_samples} images)")
    print("\nYou can now use these figures in your report Section 4.2!\n")


if __name__ == "__main__":
    main()
