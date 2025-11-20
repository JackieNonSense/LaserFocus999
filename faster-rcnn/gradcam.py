"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Faster R-CNN.

This module implements Grad-CAM visualization for object detection models,
allowing you to see which regions of the image the model focuses on when
making predictions.

Reference:
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    https://arxiv.org/abs/1610.02391
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Grad-CAM for visualizing what the model is looking at.

    Args:
        model: The detection model (Faster R-CNN)
        target_layer: The layer to compute gradients from (usually the last conv layer)
    """

    def __init__(self, model, target_layer_name="backbone.fpn_output4"):
        """
        Initialize Grad-CAM.

        Args:
            model: Detectron2 model
            target_layer_name: Name of the target layer for Grad-CAM
                For Faster R-CNN with FPN, common choices:
                - "backbone.fpn_output4" (P5, 1/32 scale)
                - "backbone.fpn_output3" (P4, 1/16 scale)
                - "backbone.fpn_output2" (P3, 1/8 scale)
        """
        self.model = model
        self.model.eval()

        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks to capture activations and gradients.
        """
        def forward_hook(module, input, output):
            """Capture the forward pass output (activations)"""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Capture the backward pass gradients"""
            self.gradients = grad_output[0].detach()

        # Find the target layer
        target_layer = self._find_layer(self.model, self.target_layer_name)

        if target_layer is None:
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")

        # Register hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _find_layer(self, model, layer_name):
        """
        Find a layer in the model by name.

        Args:
            model: The model to search
            layer_name: Name of the layer (e.g., "backbone.fpn_output4")

        Returns:
            The layer module or None if not found
        """
        parts = layer_name.split('.')
        current = model

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                # Try to find in children
                found = False
                for name, module in current.named_children():
                    if name == part:
                        current = module
                        found = True
                        break
                if not found:
                    return None

        return current

    def generate_cam(self, image_tensor, target_box=None, target_class=None):
        """
        Generate Grad-CAM heatmap for a specific detection.

        Args:
            image_tensor: Input image tensor [1, 3, H, W]
            target_box: Target bounding box [x1, y1, x2, y2] (optional)
            target_class: Target class index (optional)

        Returns:
            cam: Grad-CAM heatmap [H, W] normalized to [0, 1]
        """
        # Forward pass
        self.model.zero_grad()

        # Prepare input in Detectron2 format
        height, width = image_tensor.shape[-2:]
        inputs = [{
            "image": image_tensor[0],
            "height": height,
            "width": width,
        }]

        # Get model outputs
        with torch.set_grad_enabled(True):
            outputs = self.model(inputs)

        if len(outputs) == 0 or len(outputs[0]["instances"]) == 0:
            # No detections, return empty heatmap
            return np.zeros((height, width), dtype=np.float32)

        instances = outputs[0]["instances"]

        # Select which detection to visualize
        if target_box is not None:
            # Find the detection closest to target_box
            boxes = instances.pred_boxes.tensor
            ious = self._compute_iou(boxes, torch.tensor(target_box).to(boxes.device))
            target_idx = ious.argmax().item()
        elif target_class is not None:
            # Find the first detection of target_class
            classes = instances.pred_classes
            mask = (classes == target_class)
            if mask.sum() == 0:
                return np.zeros((height, width), dtype=np.float32)
            target_idx = mask.nonzero()[0].item()
        else:
            # Use the highest confidence detection
            target_idx = instances.scores.argmax().item()

        # Get the score for backpropagation
        target_score = instances.scores[target_idx]

        # Backward pass
        target_score.backward()

        # Generate CAM
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured. Using fallback.")
            return np.zeros((height, width), dtype=np.float32)

        # Compute weights using global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()  # [H', W']

        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, (width, height))

        return cam

    def _compute_iou(self, boxes1, box2):
        """
        Compute IoU between boxes1 and box2.

        Args:
            boxes1: Tensor of shape [N, 4] (x1, y1, x2, y2)
            box2: Tensor of shape [4] (x1, y1, x2, y2)

        Returns:
            ious: Tensor of shape [N]
        """
        box2 = box2.unsqueeze(0)  # [1, 4]

        # Compute intersection
        x1 = torch.max(boxes1[:, 0], box2[:, 0])
        y1 = torch.max(boxes1[:, 1], box2[:, 1])
        x2 = torch.min(boxes1[:, 2], box2[:, 2])
        y2 = torch.min(boxes1[:, 3], box2[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Compute areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        # Compute union
        union = area1 + area2 - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        return iou


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ - An improved version of Grad-CAM.

    Reference:
        Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
        https://arxiv.org/abs/1710.11063
    """

    def generate_cam(self, image_tensor, target_box=None, target_class=None):
        """
        Generate Grad-CAM++ heatmap (uses weighted gradients).
        """
        # Forward pass
        self.model.zero_grad()

        height, width = image_tensor.shape[-2:]
        inputs = [{
            "image": image_tensor[0],
            "height": height,
            "width": width,
        }]

        with torch.set_grad_enabled(True):
            outputs = self.model(inputs)

        if len(outputs) == 0 or len(outputs[0]["instances"]) == 0:
            return np.zeros((height, width), dtype=np.float32)

        instances = outputs[0]["instances"]

        # Select target detection
        if target_box is not None:
            boxes = instances.pred_boxes.tensor
            ious = self._compute_iou(boxes, torch.tensor(target_box).to(boxes.device))
            target_idx = ious.argmax().item()
        elif target_class is not None:
            classes = instances.pred_classes
            mask = (classes == target_class)
            if mask.sum() == 0:
                return np.zeros((height, width), dtype=np.float32)
            target_idx = mask.nonzero()[0].item()
        else:
            target_idx = instances.scores.argmax().item()

        target_score = instances.scores[target_idx]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((height, width), dtype=np.float32)

        # Grad-CAM++ weighting
        gradients = self.gradients
        activations = self.activations

        # Compute alpha weights
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)

        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom

        # Compute weights
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize and resize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (width, height))

        return cam


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to heatmap.

    Args:
        heatmap: Numpy array [H, W] with values in [0, 1]
        colormap: OpenCV colormap (default: COLORMAP_JET)

    Returns:
        colored_heatmap: RGB image [H, W, 3]
    """
    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR to RGB
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    return colored_heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on image.

    Args:
        image: Original image [H, W, 3] (RGB, uint8)
        heatmap: Heatmap [H, W] with values in [0, 1]
        alpha: Blending factor (0 = only image, 1 = only heatmap)

    Returns:
        overlayed: Blended image [H, W, 3]
    """
    # Apply colormap to heatmap
    colored_heatmap = apply_colormap(heatmap)

    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

    return overlayed
