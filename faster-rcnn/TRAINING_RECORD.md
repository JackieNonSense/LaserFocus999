# Faster R-CNN Training Record

## Model Configuration

**Architecture**: Faster R-CNN with ResNet-50-FPN backbone

**Pre-trained Weights**: COCO Detection model
- Source: `detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl`

**Number of Classes**: 12 (AgroPest-12 insect classes)

## Training Configuration

### Dataset
- **Training Set**: 11,502 images
- **Validation Set**: 1,095 images
- **Test Set**: 546 images
- **Format**: COCO JSON format (converted from YOLO format)
- **Data Root**: `/root/autodl-tmp/dataset`

### Hyperparameters
- **Batch Size**: 4 images per batch
- **Base Learning Rate**: 0.001
- **Learning Rate Schedule**:
  - Steps: [15000, 20000]
  - Gamma: 0.1 (10x decay at each step)
- **Warmup**:
  - Method: Linear
  - Factor: 0.001
  - Iterations: 1000
- **Optimizer**: SGD
  - Momentum: 0.9
  - Weight Decay: 0.0001

### Training Duration
- **Total Iterations**: 25,000
- **Checkpoint Interval**: Every 2,500 iterations
- **Evaluation Interval**: Every 2,500 iterations
- **Total Training Time**: ~50 minutes

### Image Augmentation
- **Training Image Size**:
  - Min: Random choice from [640, 672, 704, 736, 768, 800] pixels
  - Max: 1333 pixels
- **Test Image Size**:
  - Min: 800 pixels
  - Max: 1333 pixels
- **Random Horizontal Flip**: Enabled

### Model Settings
- **RoI Heads**:
  - Batch size per image: 512
  - Score threshold (test): 0.5
- **Detections per Image**: 100

## Hardware & Environment

### Compute Infrastructure
- **Platform**: AutoDL Cloud GPU Service
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **GPU Memory Usage**: ~2.7GB (peak)
- **Number of GPUs**: 1

### Software Environment
- **Operating System**: Linux
- **Python**: 3.8
- **PyTorch**: 2.0.1+cu118
- **CUDA**: 11.8
- **Detectron2**: Built from source (main branch)
- **Conda Environment**: insect

### Key Dependencies
```
torch==2.0.1+cu118
torchvision==0.15.2+cu118
detectron2 (built from source)
opencv-python>=4.8.0
pycocotools>=2.0.6
```

## Training Results

### Validation Set Performance (During Training)
- **mAP (AP@0.50:0.95)**: 41.41%
- **AP@0.50**: 72.02%
- **AP@0.75**: 43.37%

### Test Set Performance (Final Evaluation)
- **mAP (AP@0.50:0.95)**: 41.65%
- **AP@0.50**: 73.06%
- **AP@0.75**: 41.20%
- **AP (small objects)**: N/A
- **AP (medium objects)**: 14.54%
- **AP (large objects)**: 43.80%

### Per-Class Performance (Test Set)
| Class | AP (%) | Performance |
|-------|--------|-------------|
| Class 0 | 26.84 | Poor |
| Class 1 | 42.62 | Medium |
| Class 2 | 26.13 | Poor |
| Class 3 | 27.26 | Poor |
| Class 4 | 27.02 | Poor |
| Class 5 | 33.51 | Medium |
| Class 6 | 29.33 | Poor |
| Class 7 | 76.18 | Excellent |
| Class 8 | 40.47 | Medium |
| Class 9 | 57.17 | Good |
| Class 10 | 55.42 | Good |
| Class 11 | 57.91 | Good |

### Model Stability
- **Validation mAP**: 41.41%
- **Test mAP**: 41.65%
- **Difference**: +0.24% (excellent generalization, no overfitting)

## Output Files

### Model Checkpoints
- **Location**: `outputs/checkpoints/faster_rcnn_R50_FPN/`
- **Final Model**: `model_final.pth`
- **Intermediate Checkpoints**: Saved every 2,500 iterations

### Evaluation Results
- **Test Results**: `outputs/evaluation/test/results.json`
- **Predictions**: `outputs/evaluation/test/coco_instances_results.json`
  - Total predictions: 693 bounding boxes
  - Images with detections: 528/546 (96.7%)
  - High confidence predictions (>0.8): 535/693 (77.2%)

### Training Logs
- **TensorBoard Logs**: `outputs/checkpoints/faster_rcnn_R50_FPN/`
- **Metrics**: `outputs/checkpoints/faster_rcnn_R50_FPN/metrics.json`

## Key Observations

### Strengths
1. **Excellent generalization**: Test performance matches validation (no overfitting)
2. **High confidence predictions**: 77% of predictions have confidence >0.8
3. **Strong performance on certain classes**: Class 7 achieves 76.18% AP
4. **Efficient training**: Completed in ~50 minutes with good convergence

### Weaknesses
1. **Poor small object detection**: AP for small objects is N/A
2. **Class imbalance effects**: Large variance in per-class performance (26%-76%)
3. **Medium object detection**: Only 14.54% AP for medium-sized insects
4. **Inconsistent class performance**: 5 classes below 30% AP

### Possible Improvements
1. Data augmentation for underperforming classes
2. Adjust anchor sizes for better small/medium object detection
3. Class balancing strategies during training
4. Longer training with lower learning rate
5. Multi-scale training and testing

## Date & Timestamp
- **Training Date**: November 6, 2025
- **Training Start**: ~00:10 (server time)
- **Training End**: ~01:00 (server time)
- **Evaluation Completed**: ~01:03 (server time)

---

# Run 1: Small Anchors + Multi-Scale Training

## Changes from Baseline

### Anchor Configuration
- Changed anchor sizes from [32, 64, 128, 256, 512] to [8, 16, 32, 64, 128]
- Rationale: COCO anchors designed for cars/people, too large for insects

### Training Configuration
- Expanded MIN_SIZE_TRAIN from 6 scales to 13 scales (512-896 pixels)
- Added test-time augmentation: multi-scale testing + horizontal flip
- Reduced MIN_SIZE_TEST from 800 to 704 pixels

### Other Settings
- Training iterations: 25,000 (unchanged)
- All other hyperparameters identical to baseline

## Results

### Test Set Performance

| Metric | Baseline (Run 0) | Run 1 | Change |
|--------|------------------|-------|--------|
| mAP | 41.65% | 41.74% | +0.09% |
| AP50 | 73.06% | 72.81% | -0.25% |
| AP75 | 41.20% | 42.56% | +1.36% |
| APm (medium) | 14.54% | 12.39% | -2.15% |
| APl (large) | 43.80% | 43.91% | +0.11% |

### Per-Class Performance Changes

| Class | Baseline | Run 1 | Change |
|-------|----------|-------|--------|
| 0 | 26.84% | 29.50% | +2.66% |
| 1 | 42.62% | 41.93% | -0.69% |
| 2 | 26.13% | 27.97% | +1.84% |
| 3 | 27.26% | 26.73% | -0.53% |
| 4 | 27.02% | 25.64% | -1.38% |
| 5 | 33.51% | 31.92% | -1.59% |
| 6 | 29.33% | 30.26% | +0.93% |
| 7 | 76.18% | 75.61% | -0.57% |
| 8 | 40.47% | 41.68% | +1.21% |
| 9 | 57.17% | 54.95% | -2.22% |
| 10 | 55.42% | 58.48% | +3.06% |
| 11 | 57.91% | 56.21% | -1.70% |

Classes improved: 5, Classes degraded: 7

## Problems Identified

1. Overall mAP improvement negligible (+0.09%)
2. Medium object detection worsened significantly (APm -2.15%)
3. More classes degraded than improved
4. Small anchor hypothesis partially rejected

## Analysis

The anchor size adjustment did not produce the expected improvements:
- Positive: Better bounding box precision (AP75 +1.36%)
- Negative: Degraded medium object detection (APm dropped 2.15%)
- Conclusion: Scale tuning is not the primary bottleneck

Root cause appears to be class imbalance and feature discrimination issues rather than anchor mismatch. Seven classes degraded when anchors were reduced, suggesting the model struggles with distinguishing similar insect species rather than detecting objects at different scales.

## Next Steps

Abandon further anchor optimization. Shift focus to class imbalance problem:
- Implement Focal Loss to address easy/hard example imbalance
- Use class-balanced sampling for minority classes
- Target the 5 poor-performing classes (AP < 30%)

## Training Details
- **Date**: November 8, 2025
- **Training Time**: ~50 minutes
- **Branch**: Yuchao/Faster-rcnn-improve
