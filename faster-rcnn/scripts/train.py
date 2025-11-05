"""
Training script for Faster R-CNN on AgroPest-12 dataset.

Usage:
    python scripts/train.py --config configs/faster_rcnn_R50_FPN.yaml
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
from detectron2.data import build_detection_train_loader
from detectron2 import model_zoo

from data.dataset import register_all_agropest_splits


class Trainer(DefaultTrainer):
    """
    Custom Trainer with COCO evaluation during training.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator for validation set.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


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
    Main training function.
    """
    # Setup configuration
    cfg = setup_cfg(args)

    # Register datasets
    print("Registering datasets...")
    data_root = args.data_root
    coco_json_dir = args.coco_json_dir

    register_all_agropest_splits(data_root, coco_json_dir)

    # Verify datasets are registered
    from detectron2.data import DatasetCatalog
    try:
        train_data = DatasetCatalog.get("agropest_train")
        print(f"Training set: {len(train_data)} images")
    except Exception as e:
        print(f"Error loading training set: {e}")
        return

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Start training
    print("\nStarting training...")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Evaluation period: {cfg.TEST.EVAL_PERIOD}")

    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on AgroPest-12")

    parser.add_argument("--config-file",
                       default="configs/faster_rcnn_R50_FPN.yaml",
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

    # Launch training
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
