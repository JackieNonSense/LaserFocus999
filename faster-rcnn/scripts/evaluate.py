"""
Evaluation script for Faster R-CNN on AgroPest-12 test set.

Usage:
    python scripts/evaluate.py --config configs/faster_rcnn_R50_FPN.yaml \
                               --weights outputs/checkpoints/model_final.pth
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo

from data.dataset import register_all_agropest_splits
from config import add_focal_config
# Ensure any custom ROI heads (e.g., focal) are registered before building models
from models import focal_fast_rcnn  # noqa: F401


def setup_cfg(args):
    """
    Create configs for evaluation.
    """
    cfg = get_cfg()
    add_focal_config(cfg)

    # Load base config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Merge from custom config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Set model weights
    cfg.MODEL.WEIGHTS = args.weights

    # Set to evaluation mode
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold

    cfg.freeze()
    return cfg


def main(args):
    """
    Main evaluation function.
    """
    # Setup configuration
    cfg = setup_cfg(args)

    # Register datasets
    print("Registering datasets...")
    data_root = args.data_root
    coco_json_dir = args.coco_json_dir

    register_all_agropest_splits(data_root, coco_json_dir)

    # Create predictor
    print("Loading model...")
    predictor = DefaultPredictor(cfg)

    # Evaluate on test set
    dataset_name = f"agropest_{args.split}"
    print(f"\nEvaluating on {dataset_name}...")

    # Create evaluator
    output_folder = os.path.join(args.output_dir, args.split)
    os.makedirs(output_folder, exist_ok=True)

    evaluator = COCOEvaluator(dataset_name, cfg, False, output_folder)

    # Build test loader
    val_loader = build_detection_test_loader(cfg, dataset_name)

    # Run inference
    print("Running inference...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))

    # Save results
    results_file = os.path.join(output_folder, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Also save to comparison directory if it exists
    comparison_dir = Path(__file__).parent.parent.parent / "comparison" / "results"
    if comparison_dir.exists():
        comparison_file = comparison_dir / "faster_rcnn_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results also saved to: {comparison_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on AgroPest-12")

    parser.add_argument("--config-file",
                       default="configs/faster_rcnn_R50_FPN.yaml",
                       help="Path to config file")

    parser.add_argument("--weights",
                       required=True,
                       help="Path to model weights (.pth file)")

    parser.add_argument("--data-root",
                       default="../data/AgroPest-12",
                       help="Path to AgroPest-12 dataset root")

    parser.add_argument("--coco-json-dir",
                       default="outputs/coco_annotations",
                       help="Directory containing COCO format JSON files")

    parser.add_argument("--split",
                       default="test",
                       choices=["train", "valid", "test"],
                       help="Dataset split to evaluate on")

    parser.add_argument("--output-dir",
                       default="outputs/evaluation",
                       help="Directory to save evaluation results")

    parser.add_argument("--confidence-threshold",
                       type=float,
                       default=0.5,
                       help="Confidence threshold for detections")

    args = parser.parse_args()

    print("Command Line Args:", args)
    main(args)
