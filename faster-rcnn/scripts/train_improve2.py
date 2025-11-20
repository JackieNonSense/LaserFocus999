"""
Training script for Faster R-CNN on AgroPest-12 dataset.
Phase 1+2 Implementation: Run 1 baseline + light enhancements


Usage:
  cd /root/work/LaserFocus999
  python faster-rcnn/scripts/train_improve2.py \
    --config faster-rcnn/configs/faster_rcnn_R50_FPN_improve2.yaml \
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

# Logger setup
logger = logging.getLogger("detectron2.train_improve2")


#  Core Phase 2 Enhancement: Light Color Augmentation Mapper

def build_light_augmentation_mapper(is_train: bool = True):
    """
    Build mapper with light color augmentation.

    Phase 1 (Run 1): Multi-scale training + horizontal flip
    Phase 2 (Enhancement): Add light color jitter (brightness, contrast, saturation)

    Color augmentation is light:
      - Brightness: ±10% (0.9~1.1)
      - Contrast: ±10% (0.9~1.1)
      - Saturation: ±10% (0.9~1.1)
    """
    if is_train:

        augs = [
            # Phase 1: multi-scale training
            T.ResizeShortestEdge(
                short_edge_length=(512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896),
                max_size=1333,
                sample_style="choice",
            ),
            # Phase 1: random horizontal flip
            T.RandomFlip(horizontal=True, vertical=False),

            # Phase 2: light color jitter
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
        ]
    else:
        # Test time: keep Run 1 config (resize only, no color aug)
        augs = [
            T.ResizeShortestEdge(short_edge_length=704, max_size=1333),
        ]

    def mapper(dataset_dict):
        """Apply augmentation and prepare batch."""
        dd = dataset_dict.copy()
        image = utils.read_image(dd["file_name"], format="BGR")
        annos = dd.get("annotations", [])
        image, transforms = T.apply_augmentations(augs, image)

        # pack image tensor: CHW, contiguous
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


# Custom Trainer: COCO Evaluator + Light Augmentation Mapper
class Trainer(DefaultTrainer):
    """
    Trainer with:
      - COCO evaluator (for proper mAP metrics)
      - Light augmentation mapper (Phase 2 enhancement)
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
        """Build train loader with light augmentation mapper."""
        return build_detection_train_loader(
            cfg,
            mapper=build_light_augmentation_mapper(is_train=True)
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

    # 1. Register datasets
    print("\n" + "=" * 70)
    print(" REGISTERING DATASETS")
    print("=" * 70)

    register_all_agropest_splits(args.data_root, args.coco_json_dir)

    try:
        train_data = DatasetCatalog.get("agropest_train")
        valid_data = DatasetCatalog.get("agropest_valid")
        print(f" Training set:   {len(train_data):,} images")
        print(f" Validation set: {len(valid_data):,} images")
    except Exception as e:
        print(f" Error loading datasets: {e}")
        return

    # 2. Setup configuration
    print("\n" + "=" * 70)
    print("  LOADING CONFIGURATION")
    print("=" * 70)

    cfg = setup_cfg(args)

    # 3. Print configuration summary
    print("\n" + "=" * 70)
    print(" TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model:                Faster R-CNN R50-FPN")
    print(f"Anchor sizes:         {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
    print(f"Anchor aspect ratios: {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}")
    print(f"\nOptimizer:")
    print(f"  Base LR:            {cfg.SOLVER.BASE_LR}")
    print(f"  Momentum:           {cfg.SOLVER.MOMENTUM}")
    print(f"  Weight decay:       {cfg.SOLVER.WEIGHT_DECAY}")
    print(f"\nSchedule:")
    print(f"  Max iterations:     {cfg.SOLVER.MAX_ITER}")
    print(f"  LR decay steps:     {cfg.SOLVER.STEPS}")
    print(f"  LR decay factor:    {cfg.SOLVER.GAMMA}x")
    print(f"  Warmup iterations:  {cfg.SOLVER.WARMUP_ITERS}")
    print(f"\nData:")
    print(f"  Batch size:         {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Train scales:       {cfg.INPUT.MIN_SIZE_TRAIN}")
    print(f"  Test scale:         {cfg.INPUT.MIN_SIZE_TEST}")
    print(f"  Max size:           {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"\nEvaluation:")
    print(f"  Eval period:        Every {cfg.TEST.EVAL_PERIOD} iterations")
    print(f"  Eval dataset:       {cfg.DATASETS.TEST}")
    print(f"\nEnhancements (Phase 1+2):")
    print(f"   Phase 1: Run 1 config (multi-scale anchors, multi-scale training)")
    print(f"   Phase 2: Light color augmentation (±10% brightness/contrast/saturation)")
    print(f"   Phase 2: Extended training (25k → 30k iterations)")
    print(f"\nOutput:")
    print(f"  Directory:          {cfg.OUTPUT_DIR}")
    print("=" * 70)

    # 4. Initialize trainer
    print("\n" + "=" * 70)
    print("INITIALIZING TRAINER")
    print("=" * 70)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if args.resume:
        print(" Resuming from last checkpoint")
    else:
        print(" Starting fresh training")

    # 5. Start training
    print("\n" + "=" * 70)
    print(" STARTING TRAINING")
    print("=" * 70)
    print(f"Estimated time: ~60 minutes on RTX 4090")
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    print(f"Checkpoints saved every {checkpoint_period} iterations")
    print(f"Evaluation every {cfg.TEST.EVAL_PERIOD} iterations")
    print("=" * 70 + "\n")

    trainer.train()

    # 6. Training complete
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Final model:      {cfg.OUTPUT_DIR}/model_final.pth")
    print("=" * 70 + "\n")

# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on AgroPest-12 (Phase 1+2: Run 1 + Light Enhancements)"
    )

    parser.add_argument(
        "--config-file",
        default="configs/faster_rcnn_R50_FPN_improve2.yaml",
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
    print(" COMMAND LINE ARGUMENTS")
    print("=" * 70)
    print(f"Config file:     {args.config_file}")
    print(f"Data root:       {args.data_root}")
    print(f"COCO JSON dir:   {args.coco_json_dir}")
    print(f"Resume:          {args.resume}")
    print(f"Num GPUs:        {args.num_gpus}")
    print(f"CLI opts:        {args.opts if args.opts else 'None'}")
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
