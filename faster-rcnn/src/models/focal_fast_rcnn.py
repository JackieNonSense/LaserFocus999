"""Custom ROI heads/predictor with focal loss + registry hooks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    _log_classification_stats,
)


class FocalFastRCNNOutputLayers(FastRCNNOutputLayers):
    """Fast R-CNN predictor that swaps cross-entropy with focal loss."""

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(input_shape=input_shape, **kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @classmethod
    def from_config(cls, cfg, input_shape):
        base = super().from_config(cfg, input_shape)
        base["focal_alpha"] = cfg.MODEL.ROI_HEADS.get("FOCAL_LOSS_ALPHA", 0.25)
        base["focal_gamma"] = cfg.MODEL.ROI_HEADS.get("FOCAL_LOSS_GAMMA", 2.0)
        return base

    def focal_cross_entropy(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Applies multi-class focal loss with optional alpha-balancing."""
        valid_mask = targets != -1
        if valid_mask.sum() == 0:
            return scores.sum() * 0.0

        logits = scores[valid_mask]
        target = targets[valid_mask]

        ce_loss = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.focal_alpha >= 0:
            alpha_factor = torch.ones_like(pt) * (1.0 - self.focal_alpha)
            fg_mask = target != self.num_classes  # background id == num_classes
            alpha_factor[fg_mask] = self.focal_alpha
        else:
            alpha_factor = 1.0

        focal = alpha_factor * (1.0 - pt) ** self.focal_gamma * ce_loss
        return focal.mean()

    def losses(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List) -> Dict[str, torch.Tensor]:
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            device = proposal_deltas.device
            proposal_boxes = torch.empty((0, 4), device=device)
            gt_boxes = torch.empty((0, 4), device=device)

        loss_cls = self.focal_cross_entropy(scores, gt_classes)
        loss_box = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes)
        return {
            "loss_cls": loss_cls * self.loss_weight.get("loss_cls", 1.0),
            "loss_box_reg": loss_box * self.loss_weight.get("loss_box_reg", 1.0),
        }


@ROI_HEADS_REGISTRY.register()
class FocalStandardROIHeads(StandardROIHeads):
    """Standard ROI heads that plug in the focal-loss predictor."""

    @classmethod
    def _init_box_head(cls, cfg, input_shape):  # type: ignore[override]
        ret = super()._init_box_head(cfg, input_shape)
        ret["box_predictor"] = FocalFastRCNNOutputLayers(cfg, ret["box_head"].output_shape)
        return ret
