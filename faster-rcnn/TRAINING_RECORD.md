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

# Run 1: Multi-Scale Anchors + Multi-Scale Training

## Changes from Baseline

### Anchor Configuration
- Changed from single-scale to multi-scale anchors per FPN level
- Baseline: [[32], [64], [128], [256], [512]]
- Run 1: [[8, 12, 16], [24, 32, 40], [48, 64, 80], [96, 128, 160], [192, 224, 256]]
- Rationale: Cover small-to-large insects comprehensively (8-256px range)

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
| mAP | 41.65% | 43.35% | +1.70% |
| AP50 | 73.06% | 74.41% | +1.35% |
| AP75 | 41.20% | 44.34% | +3.14% |
| APm (medium) | 14.54% | 13.21% | -1.33% |
| APl (large) | 43.80% | 45.57% | +1.77% |

### Per-Class Performance Changes

| Class | Baseline | Run 1 | Change |
|-------|----------|-------|--------|
| 0 | 26.84% | 27.84% | +1.00% |
| 1 | 42.62% | 43.41% | +0.79% |
| 2 | 26.13% | 27.61% | +1.48% |
| 3 | 27.26% | 25.83% | -1.43% |
| 4 | 27.02% | 28.02% | +1.00% |
| 5 | 33.51% | 33.04% | -0.47% |
| 6 | 29.33% | 31.05% | +1.72% |
| 7 | 76.18% | 78.18% | +2.00% |
| 8 | 40.47% | 45.29% | +4.82% |
| 9 | 57.17% | 61.32% | +4.15% |
| 10 | 55.42% | 59.30% | +3.88% |
| 11 | 57.91% | 59.32% | +1.41% |

Classes improved: 9, Classes degraded: 3

## Analysis

Multi-scale anchors produced modest but consistent improvements:
- Overall mAP: +1.70% (41.65% → 43.35%)
- Better bounding box precision: AP75 +3.14%
- Large object detection improved: APl +1.77%
- 9 out of 12 classes improved

However, critical issues remain:
- Medium object detection still problematic (APm 13.21%, down from baseline 14.54%)
- Small object detection completely failed (APs = NaN)
- Classes 0-6 remain below 35% AP
- Class imbalance effects still dominate

## Conclusion

Multi-scale anchors provide incremental improvements but do not solve the core problem. The performance gap between high-performing classes (7, 9, 10, 11: 59-78% AP) and low-performing classes (0-6: 26-33% AP) suggests class imbalance is the primary bottleneck, not anchor configuration.

## Next Steps

Proceed to Run 2: Focal Loss + Class Balancing
- Focal Loss to address easy/hard example imbalance
- RepeatFactorTrainingSampler to oversample minority classes
- Target poor-performing classes (AP < 35%)

## Training Details
- **Date**: November 9, 2025
- **Training Time**: ~50 minutes
- **Branch**: Yuchao/faster-rcnn-run2

---

# Run 2: Focal Loss + Class Balancing (FAILED)

## Changes from Run 1

### Loss Function
- Replaced standard cross-entropy with **Focal Loss**
- Parameters: Alpha=0.25, Gamma=2.0
- Implementation: FocalStandardROIHeads with FocalFastRCNNOutputLayers
- Rationale: Address easy/hard example imbalance

### Data Sampling
- Implemented **RepeatFactorTrainingSampler** for class balancing
- Repeat threshold: 0.001 (oversample classes with <0.1% frequency)
- Effective dataset size after resampling: enlarged training set
- Rationale: Oversample minority classes

### Other Settings
- Identical anchor configuration to Run 1: multi-scale anchors
- All other hyperparameters unchanged

## Training Set Class Distribution

| Class | Instances | Percentage |
|-------|-----------|------------|
| 0 | 2,231 | 14.60% |
| 1 | 1,596 | 10.44% |
| 2 | 1,058 | 6.92% |
| 3 | 1,740 | 11.39% |
| 4 | 1,083 | 7.09% |
| 5 | 1,182 | 7.73% |
| 6 | 1,071 | 7.01% |
| 7 | 1,062 | 6.95% |
| 8 | 918 | 6.01% |
| 9 | 1,199 | 7.85% |
| 10 | 1,167 | 7.64% |
| 11 | 975 | 6.38% |

## Results

### Test Set Performance

| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| mAP | 43.35% | 33.65% | **-9.70%** |
| AP50 | 74.41% | 57.37% | **-17.04%** |
| AP75 | 44.34% | 36.23% | **-8.11%** |
| APm (medium) | 13.21% | 10.92% | **-2.29%** |
| APl (large) | 45.57% | 35.24% | **-10.33%** |

### Per-Class Performance Changes

| Class | Run 1 | Run 2 | Change | Severity |
|-------|-------|-------|--------|----------|
| 0 | 27.84% | 19.30% | -8.54% | Severe |
| 1 | 43.41% | 38.20% | -5.21% | Moderate |
| 2 | 27.61% | 12.24% | **-15.37%** | Critical |
| 3 | 25.83% | 13.39% | **-12.44%** | Critical |
| 4 | 28.02% | 19.66% | -8.36% | Severe |
| 5 | 33.04% | 17.63% | **-15.41%** | Critical |
| 6 | 31.05% | 24.79% | -6.26% | Moderate |
| 7 | 78.18% | 73.50% | -4.68% | Moderate |
| 8 | 45.29% | 24.63% | **-20.66%** | Catastrophic |
| 9 | 61.32% | 52.07% | -9.25% | Severe |
| 10 | 59.30% | 50.58% | -8.72% | Severe |
| 11 | 59.32% | 57.85% | -1.47% | Minor |

**Classes degraded: 12/12 (100%)**
**Average degradation: -9.70%**

## Failure Analysis

### Critical Issues

1. **Universal Performance Collapse**
   - All 12 classes degraded, no improvements whatsoever
   - 4 classes suffered catastrophic/critical degradation (>10% drop)
   - Class 8 lost 20.66% AP - completely destroyed

2. **Focal Loss Backfire**
   - Expected to help hard examples and minority classes
   - Instead, severely damaged performance across all classes
   - May have over-penalized easy examples, disrupting learning

3. **Class Balancing Failure**
   - RepeatFactorTrainingSampler did not improve minority classes
   - Classes 2, 3, 5 (relatively rare) suffered critical degradation
   - Over-sampling may have caused overfitting or noisy gradients

4. **Baseline Performance Superior**
   - Standard cross-entropy outperforms Focal Loss by 9.70% mAP
   - Natural class distribution better than forced balancing
   - Suggests dataset is not severely imbalanced

### Root Cause Hypothesis

1. **Focal Loss Hyperparameters Mismatch**
   - Alpha=0.25, Gamma=2.0 are standard for object detection
   - May not suit insect classification with subtle inter-class differences
   - Focal loss may suppress learning from moderately-hard examples

2. **Sampling-Induced Overfitting**
   - Repeated sampling of rare classes creates artificial data distribution
   - Model may overfit to repeated instances
   - Validation metrics (35.12% mAP) vs Test metrics (33.65% mAP) show generalization gap

3. **Wrong Problem Diagnosis**
   - Assumed class imbalance was primary bottleneck
   - Actual problem may be fine-grained feature discrimination
   - Insect species are visually similar, require robust features not loss reweighting

## Conclusion

**Run 2 is a complete failure.** Focal Loss + Class Balancing produced the worst results of all runs:
- Baseline (Run 0): 41.65% mAP
- Run 1 (Multi-scale anchors): 43.35% mAP ✓ Best
- Run 2 (Focal Loss + Balancing): 33.65% mAP ✗ Worst

**Key Learnings:**
1. Class imbalance is NOT the primary bottleneck for AgroPest-12
2. Standard cross-entropy is superior to Focal Loss for this task
3. Natural data distribution outperforms forced class balancing
4. Multi-scale anchors (Run 1) remain the best improvement strategy

**Recommendation:** Abandon loss function / sampling approaches. Focus on:
- Data augmentation specific to insect images
- Better backbone architectures (e.g., ResNet-101, Swin Transformer)
- Longer training or different learning rate schedules

## Training Details
- **Date**: November 9, 2025
- **Training Time**: ~50 minutes
- **Branch**: Yuchao/faster-rcnn-run2
- **Status**: FAILED - Do not use for final model
