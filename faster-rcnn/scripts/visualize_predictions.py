"""
Visualize Faster R-CNN predictions on test images.

This script reads the predictions from evaluation and draws bounding boxes
on test images for visual inspection.

Usage:
    python scripts/visualize_predictions.py \
        --predictions outputs/evaluation/test/coco_instances_results.json \
        --coco-json outputs/coco_annotations/test_coco.json \
        --image-dir /root/autodl-tmp/dataset/test/images \
        --output-dir outputs/visualizations \
        --num-samples 20 \
        --min-score 0.5
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


# Color palette for 12 classes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Class 0: Blue
    (0, 255, 0),      # Class 1: Green
    (0, 0, 255),      # Class 2: Red
    (255, 255, 0),    # Class 3: Cyan
    (255, 0, 255),    # Class 4: Magenta
    (0, 255, 255),    # Class 5: Yellow
    (128, 0, 0),      # Class 6: Dark Blue
    (0, 128, 0),      # Class 7: Dark Green
    (0, 0, 128),      # Class 8: Dark Red
    (128, 128, 0),    # Class 9: Dark Cyan
    (128, 0, 128),    # Class 10: Dark Magenta
    (0, 128, 128),    # Class 11: Dark Yellow
]

CLASS_NAMES = [
    "Insect_0", "Insect_1", "Insect_2", "Insect_3",
    "Insect_4", "Insect_5", "Insect_6", "Insect_7",
    "Insect_8", "Insect_9", "Insect_10", "Insect_11"
]


def load_coco_json(json_path):
    """
    Load COCO format JSON file.

    Returns:
        dict: Mapping from image_id to image info
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    # Group ground truth annotations by image_id
    gt_anns = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        gt_anns[ann['image_id']].append(ann)

    return images, gt_anns


def load_predictions(pred_path):
    """
    Load prediction results.

    Returns:
        dict: Mapping from image_id to list of predictions
    """
    with open(pred_path, 'r') as f:
        predictions = json.load(f)

    # Group predictions by image_id
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)

    return pred_by_image


def draw_bbox(image, bbox, color, label, score, thickness=2):
    """
    Draw a bounding box on the image.

    Args:
        image: OpenCV image (numpy array)
        bbox: [x, y, width, height] in COCO format
        color: BGR color tuple
        label: Class label string
        score: Confidence score
        thickness: Line thickness
    """
    x, y, w, h = [int(v) for v in bbox]

    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    # Prepare label text
    text = f"{label}: {score:.2f}"

    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)

    # Draw background for text
    cv2.rectangle(image,
                  (x, y - text_height - baseline - 5),
                  (x + text_width, y),
                  color,
                  -1)  # Filled rectangle

    # Draw text
    cv2.putText(image, text, (x, y - baseline - 2),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def visualize_image(image_path, predictions, gt_annotations=None,
                    min_score=0.5, show_gt=False):
    """
    Visualize predictions on a single image.

    Args:
        image_path: Path to the image file
        predictions: List of prediction dictionaries
        gt_annotations: List of ground truth annotations (optional)
        min_score: Minimum confidence score to display
        show_gt: Whether to show ground truth boxes

    Returns:
        numpy array: Image with drawn boxes
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    # Draw predictions
    for pred in predictions:
        if pred['score'] >= min_score:
            category_id = pred['category_id']
            bbox = pred['bbox']
            score = pred['score']

            color = COLORS[category_id % len(COLORS)]
            label = CLASS_NAMES[category_id]

            draw_bbox(image, bbox, color, label, score)

    # Optionally draw ground truth (in white with dashed line)
    if show_gt and gt_annotations:
        for ann in gt_annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']

            # Draw GT boxes with dotted white line
            x, y, w, h = [int(v) for v in bbox]
            # Simple dashed line effect by drawing short segments
            for i in range(0, w, 10):
                cv2.line(image, (x + i, y), (x + i + 5, y), (255, 255, 255), 2)
                cv2.line(image, (x + i, y + h), (x + i + 5, y + h), (255, 255, 255), 2)
            for i in range(0, h, 10):
                cv2.line(image, (x, y + i), (x, y + i + 5), (255, 255, 255), 2)
                cv2.line(image, (x + w, y + i), (x + w, y + i + 5), (255, 255, 255), 2)

    return image


def select_samples(pred_by_image, num_samples, strategy='mixed'):
    """
    Select images to visualize.

    Args:
        pred_by_image: Dictionary mapping image_id to predictions
        num_samples: Number of samples to select
        strategy: Selection strategy:
            - 'random': Random selection
            - 'high_conf': Images with highest average confidence
            - 'low_conf': Images with lowest average confidence
            - 'mixed': Mix of high and low confidence

    Returns:
        list: Selected image IDs
    """
    if strategy == 'random':
        image_ids = list(pred_by_image.keys())
        return random.sample(image_ids, min(num_samples, len(image_ids)))

    # Calculate average confidence per image
    image_scores = []
    for img_id, preds in pred_by_image.items():
        if len(preds) > 0:
            avg_score = sum(p['score'] for p in preds) / len(preds)
            image_scores.append((img_id, avg_score))

    # Sort by score
    image_scores.sort(key=lambda x: x[1], reverse=True)

    if strategy == 'high_conf':
        return [img_id for img_id, _ in image_scores[:num_samples]]

    elif strategy == 'low_conf':
        return [img_id for img_id, _ in image_scores[-num_samples:]]

    elif strategy == 'mixed':
        # Take half from high confidence, half from low
        half = num_samples // 2
        high = [img_id for img_id, _ in image_scores[:half]]
        low = [img_id for img_id, _ in image_scores[-half:]]
        return high + low

    return []


def main():
    parser = argparse.ArgumentParser(description="Visualize Faster R-CNN predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--coco-json", type=str, required=True,
                        help="Path to COCO format test annotations")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of images to visualize")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Minimum confidence score to display")
    parser.add_argument("--strategy", type=str, default="mixed",
                        choices=['random', 'high_conf', 'low_conf', 'mixed'],
                        help="Sample selection strategy")
    parser.add_argument("--show-gt", action="store_true",
                        help="Show ground truth boxes (white dashed)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # Load COCO annotations
    images, gt_annotations = load_coco_json(args.coco_json)

    # Load predictions
    pred_by_image = load_predictions(args.predictions)

    print(f"Loaded {len(images)} images and {sum(len(p) for p in pred_by_image.values())} predictions")

    # Select samples
    print(f"Selecting {args.num_samples} samples using '{args.strategy}' strategy...")
    selected_ids = select_samples(pred_by_image, args.num_samples, args.strategy)

    print(f"Visualizing {len(selected_ids)} images...")

    # Process each selected image
    for idx, img_id in enumerate(selected_ids):
        img_info = images[img_id]
        img_filename = img_info['file_name']
        img_path = Path(args.image_dir) / img_filename

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # Get predictions and GT for this image
        predictions = pred_by_image.get(img_id, [])
        gt_anns = gt_annotations.get(img_id, []) if args.show_gt else None

        # Visualize
        vis_image = visualize_image(img_path, predictions, gt_anns,
                                    args.min_score, args.show_gt)

        if vis_image is not None:
            # Save visualization
            output_path = output_dir / f"vis_{idx:03d}_{img_filename}"
            cv2.imwrite(str(output_path), vis_image)

            # Print stats
            num_preds = len([p for p in predictions if p['score'] >= args.min_score])
            avg_conf = sum(p['score'] for p in predictions) / len(predictions) if predictions else 0
            print(f"  [{idx+1}/{len(selected_ids)}] {img_filename}: "
                  f"{num_preds} predictions, avg_conf={avg_conf:.3f}")

    print(f"\nVisualization complete! Saved to {output_dir}")
    print(f"Total images processed: {len(selected_ids)}")


if __name__ == "__main__":
    main()
