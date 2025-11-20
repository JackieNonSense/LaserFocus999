"""
Training script for Faster R-CNN (Cascade + GIoU) on AgroPest-12.

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
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo


from data.dataset import register_all_agropest_splits


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()


    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))


    if args.config_file:
        cfg.merge_from_file(args.config_file)

  
    cfg.merge_from_list(args.opts)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    return cfg


def main(args):
    """
    Main training function.
    """
    print("Registering datasets...")
    register_all_agropest_splits(args.data_root, args.coco_json_dir)


    cfg = setup_cfg(args)

    print("\n========== Training Config ==========")
    print(f"OUTPUT_DIR:         {cfg.OUTPUT_DIR}")
    print(f"MAX_ITER:           {cfg.SOLVER.MAX_ITER}")
    print(f"EVAL_PERIOD:        {cfg.TEST.EVAL_PERIOD}")
    print(f"IMS_PER_BATCH:      {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"ANCHOR_SIZES[0]:    {cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]}")
    print(f"ROI_HEADS.NAME:     {cfg.MODEL.ROI_HEADS.NAME}")
    print(f"BBOX_LOSS:          {cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE}")
    print("=====================================\n")


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN (Cascade + GIoU) on AgroPest-12")

    parser.add_argument(
        "--config-file",
        default="configs/faster_rcnn_R50_FPN.yaml", 
        help="Path to config file",
    )

    parser.add_argument(
    "--data-root",
    default="/root/autodl-tmp/dataset",  
    help="Path to AgroPest-12 dataset root",
)
    parser.add_argument(
        "--coco-json-dir",
        default="outputs/coco_annotations",         
        help="Directory containing COCO format JSON files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using command-line",
    )

    args = parser.parse_args()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
