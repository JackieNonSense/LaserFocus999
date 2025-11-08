"""Custom config helpers for Faster R-CNN experiments."""

from detectron2.config import CfgNode as CN


def add_focal_config(cfg: CN) -> None:
    """Register focal-loss hyperparameters under ROI_HEADS."""
    if not hasattr(cfg.MODEL.ROI_HEADS, "FOCAL_LOSS_ALPHA"):
        cfg.MODEL.ROI_HEADS.FOCAL_LOSS_ALPHA = 0.25
    if not hasattr(cfg.MODEL.ROI_HEADS, "FOCAL_LOSS_GAMMA"):
        cfg.MODEL.ROI_HEADS.FOCAL_LOSS_GAMMA = 2.0
