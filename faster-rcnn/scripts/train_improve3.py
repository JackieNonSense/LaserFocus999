"""
Training script for Faster R-CNN on AgroPest-12 dataset.
Phase 1+2 Implementation: High Resolution + Mosaic Augmentation

Usage:
  cd /root/work/LaserFocus999
  python faster-rcnn/scripts/train_improve3.py \
    --config faster-rcnn/configs/faster_rcnn_R50_FPN_improve3.yaml \
    --data-root /root/autodl-tmp/dataset \
    --coco-json-dir /root/autodl-tmp/dataset/coco_annotations \
    --num-gpus 1
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    build_detection_train_loader,
    detection_utils as utils,
    DatasetCatalog,
)

from data.dataset import register_all_agropest_splits

logger = logging.getLogger("detectron2.train_improve3")


def build_enhanced_augmentation_mapper(is_train: bool = True):
    """
    Build mapper with Phase 1+2 enhancements:
    - Phase 1: Multi-scale training with HIGH RESOLUTION (640-1344)
    - Phase 1: Random horizontal flip
    - Phase 2: Light color jitter
    """
    if is_train:
     
        augs = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344),
                max_size=1600,
                sample_style="choice",
            ),
            # Random horizontal flip
            T.RandomFlip(horizontal=True, vertical=False),

            # Phase 2: Light color jitter
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
        ]
    else:
        # Test time: HIGH RESOLUTION for test too
        augs = [
            T.ResizeShortestEdge(short_edge_length=800, max_size=1600),
        ]

    def mapper(dataset_dict):
        """Apply augmentation and prepare batch."""
        dd = dataset_dict.copy()
        image = utils.read_image(dd["file_name"], format="BGR")
        annos = dd.get("annotations", [])

      
        image, transforms = T.apply_augmentations(augs, image)

        # Pack image tensor: CHW, contiguous
        dd["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())

        # Transform annotations (boxes/masks)
        if len(annos) > 0:
            annos = [
                utils.transform_instance_annotations(a, transforms, image.shape[:2])
                for a in annos
                if a.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dd["instances"] = utils.filter_empty_instances(instances)

        return dd

    return mapper



# Custom Trainer: COCO Evaluator + Enhanced Augmentation Mapper

class Trainer(DefaultTrainer):
    """
    Trainer with:
      - COCO evaluator (for proper mAP metrics)
      - Enhanced augmentation mapper (Phase 1+2)
      - Keep the rest simple & stable
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create COCO evaluator for validation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with enhanced augmentation mapper."""
        return build_detection_train_loader(
            cfg,
            mapper=build_enhanced_augmentation_mapper(is_train=True)
        )



# Configuration Setup

def setup_cfg(args):
    """Load and merge configurations."""
    cfg = get_cfg()

    # 1. Load COCO base config
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    # 2. Load our Phase 1+2 config
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. Command line overrides
    cfg.merge_from_list(args.opts)

    # 4. Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg



# Main Training Loop
def main(args):
    """Main training pipeline."""

    print("\n" + "=" * 70)
    print("REGISTERING DATASETS")
    print("=" * 70)

    register_all_agropest_splits(args.data_root, args.coco_json_dir)

    try:
        train_data = DatasetCatalog.get("agropest_train")
        valid_data = DatasetCatalog.get("agropest_valid")
        print("[OK] Training set:   %d images" % len(train_data))
        print("[OK] Validation set: %d images" % len(valid_data))
    except Exception as e:
        print("[ERROR] Failed to load datasets: %s" % e)
        return

    print("\n" + "=" * 70)
    print("LOADING CONFIGURATION")
    print("=" * 70)

    cfg = setup_cfg(args)
    # Print configuration summary

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print("Model:                Faster R-CNN R50-FPN")
    print("Anchor sizes:         " + str(cfg.MODEL.ANCHOR_GENERATOR.SIZES))
    print("Anchor aspect ratios: " + str(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS))
    print("\nOptimizer:")
    print("  Base LR:            %.4f" % cfg.SOLVER.BASE_LR)
    print("  Momentum:           %.1f" % cfg.SOLVER.MOMENTUM)
    print("  Weight decay:       %.4f" % cfg.SOLVER.WEIGHT_DECAY)
    print("\nSchedule:")
    print("  Max iterations:     %d" % cfg.SOLVER.MAX_ITER)
    print("  LR decay steps:     " + str(cfg.SOLVER.STEPS))
    print("  LR decay factor:    %.1fx" % cfg.SOLVER.GAMMA)
    print("  Warmup iterations:  %d" % cfg.SOLVER.WARMUP_ITERS)
    print("\nData (Phase 1 Improvement):")
    print("  Batch size:         %d" % cfg.SOLVER.IMS_PER_BATCH)
    print("  Train scales:       " + str(cfg.INPUT.MIN_SIZE_TRAIN))
    print("  Test scale:         %d" % cfg.INPUT.MIN_SIZE_TEST)
    print("  Max size:           %d" % cfg.INPUT.MAX_SIZE_TRAIN)
    print("\nEvaluation:")
    print("  Eval period:        Every %d iterations" % cfg.TEST.EVAL_PERIOD)
    print("  Eval dataset:       " + str(cfg.DATASETS.TEST))
    print("\nEnhancements (Phase 1+2):")
    print("  [Phase 1] Higher resolution (1333->1600)")
    print("  [Phase 1] Optimized multi-scale anchors (focus on medium & large)")
    print("  [Phase 1] Longer training (30k -> 40k iterations)")
    print("  [Phase 2] Light color jitter (brightness/contrast/saturation)")
    print("\nOutput:")
    print("  Directory:          " + cfg.OUTPUT_DIR)
    print("=" * 70)

    #Initialize trainer

    print("\n" + "=" * 70)
    print("INITIALIZING TRAINER")
    print("=" * 70)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if args.resume:
        print("[OK] Resuming from last checkpoint")
    else:
        print("[OK] Starting fresh training")

    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("Estimated time: ~80 minutes on RTX 4090")
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    print("Checkpoints saved every %d iterations" % checkpoint_period)
    print("Evaluation every %d iterations" % cfg.TEST.EVAL_PERIOD)
    print("=" * 70 + "\n")

    trainer.train()


    #  Training complete

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Output directory: " + cfg.OUTPUT_DIR)
    print("Final model:      " + cfg.OUTPUT_DIR + "/model_final.pth")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on AgroPest-12 (Phase 1+2: High Res + Mosaic)"
    )

    parser.add_argument(
        "--config-file",
        default="configs/faster_rcnn_R50_FPN_improve3.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--data-root",
        default="/root/autodl-tmp/dataset",
        help="Path to AgroPest-12 dataset root (use external disk to save space)",
    )

    parser.add_argument(
        "--coco-json-dir",
        default="/root/autodl-tmp/dataset/coco_annotations",
        help="Directory containing COCO format JSON files (use external disk)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )

    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (e.g. SOLVER.MAX_ITER 20000)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("COMMAND LINE ARGUMENTS")
    print("=" * 70)
    print("Config file:     " + args.config_file)
    print("Data root:       " + args.data_root)
    print("COCO JSON dir:   " + args.coco_json_dir)
    print("Resume:          " + str(args.resume))
    print("Num GPUs:        %d" % args.num_gpus)
    print("CLI opts:        " + str(args.opts if args.opts else "None"))
    print("=" * 70 + "\n")

    # Launch training
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )