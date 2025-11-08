"""
Training script for Faster R-CNN with Focal Loss and Class Balancing.
Run 2: Address class imbalance problem

Usage:
    python scripts/train_focal.py --config configs/faster_rcnn_R50_FPN_focal.yaml
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data import DatasetCatalog
from detectron2 import model_zoo

from data.dataset import register_all_agropest_splits
# Ensure custom ROI heads are registered
from models import focal_fast_rcnn  # noqa: F401


class FocalTrainer(DefaultTrainer):
    """
    Custom Trainer with class-balanced sampling and COCO evaluation.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for validation set.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build train loader with RepeatFactorTrainingSampler for class balancing.
        """
        dataset_name = cfg.DATASETS.TRAIN[0]
        dataset_dicts = DatasetCatalog.get(dataset_name)

        print(f"\nSetting up class-balanced sampling for {len(dataset_dicts)} images...")

        # Use RepeatFactorTrainingSampler to oversample minority classes
        # repeat_thresh: images with category frequency below this will be repeated
        repeat_thresh = cfg.DATALOADER.get("REPEAT_THRESHOLD", 0.001)

        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts,
            repeat_thresh=repeat_thresh,
        )
        sampler = RepeatFactorTrainingSampler(
            repeat_factors,
            shuffle=True,
            seed=cfg.SEED,
        )

        print(f"RepeatFactorTrainingSampler initialized with threshold={repeat_thresh}")
        print(f"Effective dataset size after resampling: {int(repeat_factors.sum().item())}")

        mapper = DatasetMapper(cfg, is_train=True)

        return build_detection_train_loader(
            cfg,
            mapper=mapper,
            sampler=sampler
        )


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # Load base config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Merge from custom config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # Merge from command line arguments
    cfg.merge_from_list(args.opts)

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg


def main(args):
    """
    Main training function with class balancing.
    """
    # Setup configuration
    cfg = setup_cfg(args)

    # Register datasets
    print("Registering datasets...")
    data_root = args.data_root
    coco_json_dir = args.coco_json_dir

    register_all_agropest_splits(data_root, coco_json_dir)

    # Verify datasets are registered
    try:
        train_data = DatasetCatalog.get("agropest_train")
        print(f"Training set: {len(train_data)} images")

        # Print class distribution
        from collections import Counter
        class_counts = Counter()
        for item in train_data:
            for ann in item.get("annotations", []):
                class_counts[ann["category_id"]] += 1

        print("\nClass distribution in training set:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = 100.0 * count / sum(class_counts.values())
            print(f"  Class {class_id}: {count:5d} instances ({percentage:5.2f}%)")

    except Exception as e:
        print(f"Error loading training set: {e}")
        return

    # Create trainer with class balancing
    print("\nInitializing FocalTrainer with class-balanced sampling...")
    trainer = FocalTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Start training
    print("\nStarting training...")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Evaluation period: {cfg.TEST.EVAL_PERIOD}")
    print(f"Using RepeatFactorTrainingSampler for class balancing")

    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN with Focal Loss and Class Balancing")

    parser.add_argument("--config-file",
                       default="configs/faster_rcnn_R50_FPN_focal.yaml",
                       help="Path to config file")

    parser.add_argument("--data-root",
                       default="../data/AgroPest-12",
                       help="Path to AgroPest-12 dataset root")

    parser.add_argument("--coco-json-dir",
                       default="outputs/coco_annotations",
                       help="Directory containing COCO format JSON files")

    parser.add_argument("--resume",
                       action="store_true",
                       help="Resume from last checkpoint")

    parser.add_argument("--num-gpus",
                       type=int,
                       default=1,
                       help="Number of GPUs to use")

    parser.add_argument("opts",
                       default=None,
                       nargs=argparse.REMAINDER,
                       help="Modify config options using command-line")

    args = parser.parse_args()

    print("Command Line Args:", args)
    print("\n" + "="*80)
    print("Run 2: Focal Loss + Class Balancing")
    print("Strategy: Use RepeatFactorTrainingSampler to oversample minority classes")
    print("="*80 + "\n")

    # Launch training
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
