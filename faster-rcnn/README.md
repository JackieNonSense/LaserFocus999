# Faster R-CNN for AgroPest-12 Insect Detection

Faster R-CNN implementation using Detectron2 for insect detection and classification on the AgroPest-12 dataset. This repository contains baseline and improved model configurations, training scripts, evaluation tools, and comprehensive visualization utilities.

## Table of Contents

- [Directory Structure](#directory-structure)
- [File Descriptions](#file-descriptions)
- [Setup](#setup)
- [Usage](#usage)
- [Model Configurations](#model-configurations)
- [Experimental Runs](#experimental-runs)
- [Visualization Tools](#visualization-tools)
- [Results](#results)

---

## Directory Structure

```
faster-rcnn/
├── configs/                           # Model configuration files
│   ├── faster_rcnn_R50_FPN.yaml      # Run 1: Multi-scale anchors (BEST)
│   ├── faster_rcnn_R50_FPN_focal.yaml
│   ├── faster_rcnn_R50_FPN_repeat.yaml
│   ├── faster_rcnn_R50_FPN_improve1.yaml
│   ├── faster_rcnn_R50_FPN_improve2.yaml
│   └── faster_rcnn_R50_FPN_improve3.yaml  # Final improved configuration
│
├── scripts/                           # Executable scripts
│   ├── yolo_to_coco.py               # Dataset format conversion
│   ├── train.py                       # Standard training script
│   ├── train_focal.py                 # Training with Focal Loss
│   ├── train_improve1.py              # Training for improvement 1
│   ├── train_improve2.py              # Training for improvement 2
│   ├── train_improve3.py              # Training for improvement 3
│   ├── evaluate.py                    # Model evaluation script
│   ├── generate_all_visualizations.py # Complete visualization generator
│   ├── visualization.py               # Advanced visualization tools
│   ├── visualize_heatmap.py          # Grad-CAM heatmap visualization
│   └── visualize_predictions.py       # Prediction visualization
│
├── src/                               # Source code modules
│   ├── data/
│   │   └── dataset.py                # COCO dataset registration & loading
│   ├── models/
│   │   ├── __init__.py
│   │   └── focal_fast_rcnn.py        # Focal Loss implementation
│   ├── config.py                      # Configuration utilities
│   ├── engine/                        # Training engine (empty placeholder)
│   └── utils/                         # Utility functions (empty placeholder)
│
├── docs/                              # Documentation
│   └── AUTODL_SETUP.md               # AutoDL cloud setup guide
│
├── faster-rcnn-results/              # Evaluation results
│   ├── results.json                   # Run 1 overall metrics
│   ├── coco_instances_results.json    # Run 1 per-image predictions
│   ├── results_improve3.json          # Improve3 overall metrics
│   └── coco_instances_results_improve3.json  # Improve3 predictions
│
├── outputs/                           # Training outputs (not in git)
│   ├── checkpoints/                   # Model checkpoints (.pth files)
│   ├── coco_annotations/             # Converted COCO format annotations
│   ├── logs/                          # Training logs
│   ├── results/                       # Evaluation results
│   ├── visualizations/               # Visualization outputs
│   └── report_figures/               # Figures for report
│
├── notebooks/                         # Jupyter notebooks (if any)
│
├── gradcam.py                         # Grad-CAM implementation module
├── requirements.txt                   # Python package dependencies
├── environment.yml                    # Conda environment specification
├── README.md                          # This file
└── TRAINING_RECORD.md                # Detailed training logs for all runs
```

---

## File Descriptions

### Configuration Files (`configs/`)

#### `faster_rcnn_R50_FPN.yaml` [BEST MODEL - Run 1]
- **Purpose**: Final production configuration with multi-scale anchors
- **Architecture**: Faster R-CNN with ResNet-50-FPN backbone
- **Key Features**:
  - Multi-scale anchors: `[[8,12,16], [24,32,40], [48,64,80], [96,128,160], [192,224,256]]`
  - 13 training scales (512-896 pixels)
  - Test-time augmentation (multi-scale + horizontal flip)
- **Performance**: mAP@0.5:0.95 = 43.35% (best among all runs)
- **Usage**: Recommended for final model and inference

#### `faster_rcnn_R50_FPN_focal.yaml`
- **Purpose**: Experimental configuration with Focal Loss
- **Key Features**:
  - Focal Loss (alpha=0.25, gamma=2.0) for handling class imbalance
  - Same multi-scale anchors as Run 1
- **Status**: FAILED experiment (mAP dropped to 33.65%)
- **Usage**: Reference only, not recommended

#### `faster_rcnn_R50_FPN_repeat.yaml`
- **Purpose**: Experimental configuration with class resampling
- **Key Features**:
  - RepeatFactorTrainingSampler for minority class oversampling
  - Repeat threshold: 0.08 (mild resampling)
- **Status**: FAILED experiment (mAP = 42.01%)
- **Usage**: Reference only, demonstrates failed approach

#### `faster_rcnn_R50_FPN_improve1.yaml`
- **Purpose**: Improvement experiment 1 configuration
- **Key Features**: Cascade RCNN architecture + GIoU loss
- **Usage**: Part of iterative improvement pipeline

#### `faster_rcnn_R50_FPN_improve2.yaml`
- **Purpose**: Improvement experiment 2 configuration
- **Key Features**: Enhanced training schedule and augmentation
- **Usage**: Part of iterative improvement pipeline

#### `faster_rcnn_R50_FPN_improve3.yaml`
- **Purpose**: Final improvement experiment configuration
- **Key Features**: Combined best practices from improve1 and improve2
- **Results**: Available in `faster-rcnn-results/results_improve3.json`
- **Usage**: Latest experimental configuration

---

### Training Scripts (`scripts/`)

#### `train.py` [PRIMARY TRAINING SCRIPT]
- **Purpose**: Standard training script for all Faster R-CNN configurations
- **Usage**:
  ```bash
  python scripts/train.py \
      --config-file configs/faster_rcnn_R50_FPN.yaml \
      --data-root /path/to/dataset \
      --coco-json-dir outputs/coco_annotations \
      --num-gpus 1
  ```
- **Features**:
  - Supports multi-GPU training
  - Automatic checkpointing every 2,500 iterations
  - Validation evaluation during training
  - TensorBoard logging
- **Class**: `Trainer(DefaultTrainer)` with custom COCO evaluator

#### `train_focal.py`
- **Purpose**: Training script specifically for Focal Loss configuration
- **Differences from `train.py`**:
  - Uses `FocalStandardROIHeads` from `src/models/focal_fast_rcnn.py`
  - Implements Focal Loss for classification
  - Supports `RepeatFactorTrainingSampler`
- **Usage**: For `faster_rcnn_R50_FPN_focal.yaml` configuration only
- **Status**: Experimental, resulted in failed Run 2

#### `train_improve1.py`
- **Purpose**: Training script for improvement experiment 1
- **Key Features**: Cascade RCNN + GIoU loss implementation
- **Usage**: For `faster_rcnn_R50_FPN_improve1.yaml`

#### `train_improve2.py`
- **Purpose**: Training script for improvement experiment 2
- **Key Features**: Enhanced augmentation pipeline
- **Usage**: For `faster_rcnn_R50_FPN_improve2.yaml`

#### `train_improve3.py`
- **Purpose**: Training script for final improvement experiment
- **Key Features**: Combined architecture improvements
- **Usage**: For `faster_rcnn_R50_FPN_improve3.yaml`
- **Results**: Produces checkpoints and logs for improve3 configuration

---

### Evaluation & Visualization Scripts

#### `evaluate.py` [EVALUATION SCRIPT]
- **Purpose**: Evaluate trained models on test/validation sets
- **Usage**:
  ```bash
  python scripts/evaluate.py \
      --config-file configs/faster_rcnn_R50_FPN.yaml \
      --weights outputs/checkpoints/model_final.pth \
      --data-root /path/to/dataset \
      --coco-json-dir outputs/coco_annotations \
      --split test
  ```
- **Outputs**:
  - `results.json`: Overall metrics (mAP, AP50, AP75, per-class AP)
  - `coco_instances_results.json`: Per-image predictions with bounding boxes
- **Metrics Computed**:
  - mAP@[0.5:0.95], mAP@0.5, mAP@0.75
  - AP for small/medium/large objects
  - Per-class Average Precision (12 classes)
- **Features**:
  - COCO-style evaluation
  - Adjustable confidence threshold
  - Saves results to both `outputs/evaluation/` and `faster-rcnn-results/`

#### `generate_all_visualizations.py` [COMPLETE VISUALIZATION SUITE]
- **Purpose**: Generate all figures needed for report Section 4.2
- **Usage**:
  ```bash
  python scripts/generate_all_visualizations.py \
      --results faster-rcnn-results/results.json \
      --predictions faster-rcnn-results/coco_instances_results.json \
      --coco-json outputs/coco_annotations/test_coco.json \
      --image-dir /path/to/test/images \
      --output-dir outputs/report_figures
  ```
- **Generates**:
  1. **per_class_ap.png**: Bar chart showing AP for each of 12 insect classes
     - Color-coded: Green (AP>=50%), Orange (30-50%), Red (<30%)
     - Shows mean AP line
  2. **confusion_matrix.png**: 12x12 normalized confusion matrix
     - Shows misclassification patterns between classes
  3. **pr_curves.png**: Precision-Recall curves for all 12 classes
     - 12 subplots with individual AP values
  4. **prediction_samples/**: Directory with 10 sample visualizations
     - Mix of high-confidence and low-confidence predictions
     - Bounding boxes with class labels and confidence scores
- **Technical Details**:
  - Uses sklearn for confusion matrix computation
  - Implements custom IoU-based matching for predictions
  - Publication-quality figures (300 DPI)
  - Seaborn styling for professional appearance

#### `visualization.py`
- **Purpose**: Advanced visualization utilities and plotting functions
- **Features**:
  - Customizable color schemes for 12 classes
  - Multiple visualization modes
  - Batch processing support
- **Usage**: Can be imported as module or run standalone

#### `visualize_heatmap.py` [GRAD-CAM VISUALIZATION]
- **Purpose**: Generate Grad-CAM heatmaps to visualize model attention
- **Usage**:
  ```bash
  python scripts/visualize_heatmap.py \
      --config configs/faster_rcnn_R50_FPN.yaml \
      --weights outputs/checkpoints/model_final.pth \
      --image-dir /path/to/images \
      --output-dir outputs/heatmaps
  ```
- **Features**:
  - Grad-CAM implementation for Faster R-CNN
  - Multiple target layers supported (P3, P4, P5 in FPN)
  - Overlay heatmaps on original images
  - Batch processing with progress bar
- **Technical Details**:
  - Uses backward hooks to capture gradients
  - Supports FPN multi-scale feature maps
  - Generates color-coded attention maps
- **Use Cases**:
  - Debugging model focus regions
  - Understanding misclassifications
  - Creating interpretability figures for report

#### `visualize_predictions.py`
- **Purpose**: Visualize model predictions with bounding boxes
- **Usage**:
  ```bash
  python scripts/visualize_predictions.py \
      --predictions outputs/evaluation/test/coco_instances_results.json \
      --coco-json outputs/coco_annotations/test_coco.json \
      --image-dir /path/to/test/images \
      --output-dir outputs/visualizations \
      --num-samples 20 \
      --min-score 0.5
  ```
- **Features**:
  - Draws predicted and ground truth bounding boxes
  - Color-coded by class (12 distinct colors)
  - Adjustable confidence threshold
  - Sample selection strategies:
    - `random`: Random sampling
    - `high_conf`: Highest confidence predictions
    - `low_conf`: Lowest confidence predictions
    - `mixed`: Mix of high and low confidence
- **Output**: Annotated images saved to specified directory

---

### Data Processing

#### `yolo_to_coco.py` [DATASET CONVERTER]
- **Purpose**: Convert YOLO format annotations to COCO format for Detectron2
- **Usage**:
  ```bash
  python scripts/yolo_to_coco.py \
      --yolo-dir /path/to/AgroPest-12 \
      --output-dir outputs/coco_annotations \
      --split test  # or train, valid
  ```
- **Input Format** (YOLO):
  ```
  dataset/
  ├── train/
  │   ├── images/
  │   │   └── image001.jpg
  │   └── labels/
  │       └── image001.txt  # class_id center_x center_y width height (normalized)
  ├── valid/
  └── test/
  ```
- **Output Format** (COCO JSON):
  ```json
  {
    "images": [{"id": 0, "file_name": "image001.jpg", "width": 640, "height": 640}],
    "annotations": [{"id": 0, "image_id": 0, "category_id": 5, "bbox": [x, y, w, h]}],
    "categories": [{"id": 0, "name": "Ants"}, ...]
  }
  ```
- **Features**:
  - Handles all three splits (train/valid/test)
  - Validates image-label correspondence
  - Converts normalized coordinates to pixel coordinates
  - Progress bar for batch processing
- **Output Files**:
  - `train_coco.json` (11,502 images)
  - `valid_coco.json` (1,095 images)
  - `test_coco.json` (546 images)

---

### Source Code Modules (`src/`)

#### `src/data/dataset.py` [DATASET REGISTRATION]
- **Purpose**: Register AgroPest-12 dataset with Detectron2's DatasetCatalog
- **Functions**:
  - `register_all_agropest_splits()`: Registers train/valid/test splits
  - `load_agropest_json()`: Loads COCO JSON and returns dataset dicts
- **Usage**: Automatically imported by training/evaluation scripts
- **Registered Datasets**:
  - `agropest_train`: 11,502 training images
  - `agropest_valid`: 1,095 validation images
  - `agropest_test`: 546 test images
- **Features**:
  - Validates JSON file existence
  - Sets correct image root paths
  - Handles 12 insect categories

#### `src/models/focal_fast_rcnn.py`
- **Purpose**: Custom ROI heads with Focal Loss implementation
- **Classes**:
  - `FocalFastRCNNOutputLayers`: Replaces standard cross-entropy with Focal Loss
  - `FocalStandardROIHeads`: Custom ROI heads using Focal Loss layers
- **Focal Loss Formula**:
  ```
  FL(pt) = -alpha(1-pt)^gamma log(pt)
  where pt = model confidence for true class
  ```
- **Parameters**:
  - alpha = 0.25: Balancing factor
  - gamma = 2.0: Focusing parameter
- **Usage**: Used by `train_focal.py` for Run 2 experiment
- **Status**: Experimental feature, resulted in performance degradation

#### `src/config.py`
- **Purpose**: Configuration utilities and helper functions
- **Features**: Custom configuration extensions for Detectron2

---

### Grad-CAM Module

#### `gradcam.py` [GRAD-CAM IMPLEMENTATION]
- **Purpose**: Standalone Grad-CAM (Gradient-weighted Class Activation Mapping) module
- **Reference**: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- **Class**: `GradCAM`
- **Usage**:
  ```python
  from gradcam import GradCAM

  gradcam = GradCAM(model, target_layer_name="backbone.fpn_output4")
  heatmap = gradcam.generate_cam(image, target_class=5)
  overlay = gradcam.overlay_cam_on_image(image, heatmap)
  ```
- **Features**:
  - Works with Detectron2 Faster R-CNN models
  - Supports multiple FPN layers (P3, P4, P5)
  - Generates heatmaps showing model attention regions
  - Can overlay heatmaps on original images
- **Methods**:
  - `generate_cam()`: Compute Grad-CAM heatmap for target class
  - `overlay_cam_on_image()`: Create visualization overlay
  - `save_cam()`: Save heatmap to file
- **Use Cases**:
  - Model interpretability
  - Debugging misclassifications
  - Understanding what features the model learns
  - Creating figures for presentations

---

### Documentation Files

#### `README.md` (This File)
- **Purpose**: Complete project documentation
- **Sections**: Setup, usage, file descriptions, experimental results

#### `TRAINING_RECORD.md` [DETAILED TRAINING LOGS]
- **Purpose**: Comprehensive record of all experimental runs
- **Contents**:
  - **Run 0 (Baseline)**: Initial configuration, 41.65% mAP
  - **Run 1 (Multi-scale Anchors)**: Best model, 43.35% mAP (+1.70%)
  - **Run 2 (Focal Loss)**: Failed experiment, 33.65% mAP (-9.70%)
  - **Run 2a (Mild Resampling)**: Failed experiment, 42.01% mAP (-1.34%)
  - **Improve1-3 Experiments**: Latest improvement iterations
- **Information Per Run**:
  - Configuration changes
  - Training hyperparameters
  - Validation/test results
  - Per-class performance breakdown
  - Failure analysis (for failed runs)
  - Lessons learned
- **Usage**: Reference for understanding experimental progression

#### `docs/AUTODL_SETUP.md`
- **Purpose**: Setup guide for AutoDL cloud platform
- **Contents**:
  - Environment configuration
  - GPU setup instructions
  - Detectron2 installation on AutoDL
  - Common troubleshooting

---

### Environment Files

#### `requirements.txt`
- **Purpose**: Python package dependencies for pip installation
- **Key Packages**:
  - `torch`, `torchvision`: PyTorch framework
  - `opencv-python`: Image processing
  - `pycocotools`: COCO evaluation
  - `matplotlib`, `seaborn`: Visualization
  - `scikit-learn`: Metrics computation
  - `tqdm`: Progress bars
- **Usage**: `pip install -r requirements.txt`
- **Note**: Detectron2 must be installed separately from source

#### `environment.yml`
- **Purpose**: Conda environment specification
- **Environment Name**: `insect-detection`
- **Usage**: `conda env create -f environment.yml`
- **Includes**: All dependencies plus conda-specific packages

---

### Results Files (`faster-rcnn-results/`)

#### `results.json`
- **Purpose**: Overall evaluation metrics for Run 1 (best model)
- **Contents**:
  ```json
  {
    "bbox": {
      "AP": 41.653974832019955,      // mAP@[0.5:0.95]
      "AP50": 73.06246111135664,     // mAP@0.5
      "AP75": 41.20203048229707,     // mAP@0.75
      "APs": NaN,                     // Small objects
      "APm": 14.53810796193033,      // Medium objects
      "APl": 43.80396712539063,      // Large objects
      "AP-insect_class_0": 26.84,    // Ants
      "AP-insect_class_1": 42.62,    // Bees
      ...                             // Classes 2-11
    }
  }
  ```
- **Performance Summary**:
  - mAP@0.5:0.95: 41.65%
  - mAP@0.5: 73.06%
  - Best class (Moths): 76.18% AP
  - Worst class (Beetles): 26.13% AP

#### `coco_instances_results.json`
- **Purpose**: Per-image predictions for Run 1
- **Contents**: Array of 693 prediction objects
- **Format**:
  ```json
  [
    {
      "image_id": 0,
      "category_id": 11,
      "bbox": [60.61, 85.21, 509.46, 481.90],  // [x, y, width, height]
      "score": 0.9810015559196472
    },
    ...
  ]
  ```
- **Statistics**:
  - Total predictions: 693 bounding boxes
  - Images with detections: 528/546 (96.7%)
  - High confidence (>0.8): 535/693 (77.2%)
- **Usage**: Input for `generate_all_visualizations.py` and `visualize_predictions.py`

#### `results_improve3.json` & `coco_instances_results_improve3.json`
- **Purpose**: Results for latest improvement experiment (improve3)
- **Format**: Same as above
- **Usage**: Compare performance improvements across experiments

---

## Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 24GB+ GPU VRAM recommended (or reduce batch size)

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/LaserFocus999.git
cd LaserFocus999/faster-rcnn
```

### 2. Create Environment

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate insect-detection
```

**Option B: Using pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install PyTorch

Install PyTorch according to your CUDA version:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check: https://pytorch.org/get-started/locally/

### 4. Install Detectron2

**From source (recommended):**
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

**Verify installation:**
```bash
python -c "import detectron2; print(detectron2.__version__)"
```

See official guide: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

---

## Usage

### Quick Start: 3-Step Pipeline

#### Step 1: Convert Dataset to COCO Format

```bash
python scripts/yolo_to_coco.py \
    --yolo-dir /path/to/AgroPest-12 \
    --output-dir outputs/coco_annotations \
    --splits train valid test
```

This creates:
- `outputs/coco_annotations/train_coco.json` (11,502 images)
- `outputs/coco_annotations/valid_coco.json` (1,095 images)
- `outputs/coco_annotations/test_coco.json` (546 images)

#### Step 2: Train Model

**Use best configuration (Run 1):**
```bash
python scripts/train.py \
    --config-file configs/faster_rcnn_R50_FPN.yaml \
    --data-root /path/to/AgroPest-12 \
    --coco-json-dir outputs/coco_annotations \
    --num-gpus 1
```

**Monitor training:**
```bash
tensorboard --logdir outputs/checkpoints/faster_rcnn_R50_FPN
```

Training takes approximately 50 minutes on RTX 4090.

#### Step 3: Evaluate Model

```bash
python scripts/evaluate.py \
    --config-file configs/faster_rcnn_R50_FPN.yaml \
    --weights outputs/checkpoints/faster_rcnn_R50_FPN/model_final.pth \
    --data-root /path/to/AgroPest-12 \
    --coco-json-dir outputs/coco_annotations \
    --split test \
    --confidence-threshold 0.5
```

Results saved to:
- `outputs/evaluation/test/results.json`
- `outputs/evaluation/test/coco_instances_results.json`

---

### Generate Visualizations

After evaluation, generate report figures:

```bash
python scripts/generate_all_visualizations.py \
    --results outputs/evaluation/test/results.json \
    --predictions outputs/evaluation/test/coco_instances_results.json \
    --coco-json outputs/coco_annotations/test_coco.json \
    --image-dir /path/to/AgroPest-12/test/images \
    --output-dir outputs/report_figures \
    --num-samples 10
```

This generates:
1. `per_class_ap.png` - Bar chart of per-class AP
2. `confusion_matrix.png` - Confusion matrix heatmap
3. `pr_curves.png` - Precision-Recall curves (12 classes)
4. `prediction_samples/` - Sample predictions with bounding boxes

---

### Create Grad-CAM Heatmaps

Visualize what the model is looking at:

```bash
python scripts/visualize_heatmap.py \
    --config configs/faster_rcnn_R50_FPN.yaml \
    --weights outputs/checkpoints/model_final.pth \
    --image-dir /path/to/test/images \
    --output-dir outputs/heatmaps \
    --target-layer backbone.fpn_output4 \
    --num-samples 20
```

---

## Model Configurations

### Comparison of Configurations

| Config | Description | mAP@0.5:0.95 | Status | Usage |
|--------|-------------|--------------|--------|-------|
| `faster_rcnn_R50_FPN_improve3.yaml` | Cascade RCNN + GIoU | **43.86%** | **BEST** | **Production** |
| `faster_rcnn_R50_FPN.yaml` | Multi-scale anchors + TTA | 43.35% | Good | Recommended |
| `faster_rcnn_R50_FPN_repeat.yaml` | Class resampling | 42.01% | FAILED | Reference only |
| `faster_rcnn_R50_FPN_focal.yaml` | Focal Loss experiment | 33.65% | FAILED | Reference only |
| `faster_rcnn_R50_FPN_improve1.yaml` | Improvement experiment 1 | No results | Experimental | Testing |
| `faster_rcnn_R50_FPN_improve2.yaml` | Improvement experiment 2 | No results | Experimental | Testing |

### Recommended Configuration

For best results, use **`faster_rcnn_R50_FPN_improve3.yaml`** (improve3 - BEST MODEL):

**Key Features:**
- Cascade RCNN architecture with multi-stage refinement
- GIoU (Generalized Intersection over Union) loss
- Multi-scale anchors covering 8-256 pixels
- Advanced bounding box regression
- Batch size: 4 images
- Learning rate: 0.001 with decay schedule
- 25,000 training iterations (approximately 50 minutes)

**Performance:**
- mAP@0.5:0.95: 43.86% (BEST - highest among all experiments)
- mAP@0.5: 75.75%
- mAP@0.75: 45.05%
- Per-class AP range: 28.14% - 75.74%

**Alternative:** Use **`faster_rcnn_R50_FPN.yaml`** (Run 1) for faster training with similar performance (43.35% mAP).

---

## Experimental Runs

### Summary of All Runs

| Run | Strategy | mAP | Change from Baseline | Status |
|-----|----------|-----|---------------------|--------|
| **improve3** | Cascade RCNN + GIoU loss | **43.86%** | **+2.21%** | **BEST** |
| Run 1 | Multi-scale anchors + TTA | 43.35% | +1.70% | Good |
| Run 2a | Mild class resampling | 42.01% | +0.36% | FAILED |
| Run 0 | Baseline (single-scale anchors) | 41.65% | - | Baseline |
| Run 2 | Focal Loss + aggressive resampling | 33.65% | -8.00% | FAILED |
| improve1 | Improvement experiment 1 | No results | - | Experimental |
| improve2 | Improvement experiment 2 | No results | - | Experimental |

**Note:**
- **Run 0-2a**: Baseline and iterative improvement experiments (your work)
- **improve1-3**: Advanced architecture experiments (teammate's work)
- **improve3**: Final best model with Cascade RCNN architecture

Detailed analysis available in `TRAINING_RECORD.md`.

### Key Findings

**What Worked:**
- **Cascade RCNN + GIoU loss** (+2.21% mAP over baseline) - BEST improvement
- Multi-scale anchors (+1.70% mAP)
- Test-time augmentation
- More training scales (6 to 13)
- Advanced bounding box regression with GIoU

**What Failed:**
- Focal Loss (catastrophic -9.70% drop from Run 1)
- Aggressive class resampling (Run 2: -9.70% from Run 1)
- Mild class resampling (Run 2a: -1.34% from Run 1)
- Loss function reweighting approaches

**Key Insights:**
1. Dataset is NOT severely imbalanced (2.43x ratio). Natural class distribution is optimal.
2. Performance bottleneck is fine-grained visual similarity between insect species, not class distribution.
3. Architecture improvements (Cascade RCNN) outperform loss function modifications.
4. GIoU loss provides better localization than standard IoU loss.

---

## Visualization Tools

### 1. Complete Report Figures

Generate all figures for academic report:
```bash
python scripts/generate_all_visualizations.py \
    --results faster-rcnn-results/results.json \
    --predictions faster-rcnn-results/coco_instances_results.json \
    --coco-json outputs/coco_annotations/test_coco.json \
    --image-dir /path/to/test/images \
    --output-dir outputs/report_figures
```

### 2. Prediction Visualization

Visualize specific predictions:
```bash
python scripts/visualize_predictions.py \
    --predictions faster-rcnn-results/coco_instances_results.json \
    --coco-json outputs/coco_annotations/test_coco.json \
    --image-dir /path/to/test/images \
    --output-dir outputs/pred_vis \
    --num-samples 50 \
    --strategy mixed \  # random, high_conf, low_conf, mixed
    --min-score 0.5
```

### 3. Grad-CAM Heatmaps

Generate attention heatmaps:
```bash
python scripts/visualize_heatmap.py \
    --config configs/faster_rcnn_R50_FPN.yaml \
    --weights outputs/checkpoints/model_final.pth \
    --image-dir /path/to/images \
    --output-dir outputs/gradcam \
    --target-layer backbone.fpn_output4 \  # P5 layer (1/32 scale)
    --num-samples 30
```

**Available target layers:**
- `backbone.fpn_output2` (P3, 1/8 scale) - Best for small objects
- `backbone.fpn_output3` (P4, 1/16 scale) - Medium objects
- `backbone.fpn_output4` (P5, 1/32 scale) - Large objects

---

## Results

### Best Model Performance (improve3 - Cascade RCNN + GIoU)

**Overall Metrics:**
- mAP@[0.5:0.95]: **43.86%** (BEST among all experiments)
- mAP@0.5: **75.75%**
- mAP@0.75: **45.05%**
- AP (medium objects): **14.26%**
- AP (large objects): **46.07%**

**Per-Class Performance:**

| Class | AP (%) | Rating |
|-------|--------|--------|
| Moths (7) | 75.74 | Excellent |
| Weevils (11) | 60.83 | Very Good |
| Wasps (10) | 58.39 | Very Good |
| Snails (9) | 56.79 | Very Good |
| Bees (1) | 44.83 | Good |
| Slugs (8) | 40.57 | Good |
| Beetles (2) | 36.51 | Moderate |
| Earwigs (5) | 35.55 | Moderate |
| Grasshoppers (6) | 30.28 | Poor |
| Ants (0) | 29.74 | Poor |
| Earthworms (4) | 28.95 | Poor |
| Caterpillars (3) | 28.14 | Poor |

**Key Observations:**
- Highest mAP achieved across all experimental runs
- Cascade RCNN provides multi-stage refinement for better bounding box quality
- GIoU loss improves localization accuracy (AP75: 45.05% vs Run 1: 44.34%)
- Maintains good performance on large insects (APl=46.07%)
- High inter-class variance (28-76% AP range)

### Second Best Model (Run 1 - Multi-scale Anchors)

**Overall Metrics:**
- mAP@[0.5:0.95]: 43.35% (+1.70% over baseline)
- mAP@0.5: 74.41%
- mAP@0.75: 44.34%
- Excellent generalization (test ≈ validation, no overfitting)
- High detection rate (96.7% images have detections)

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce batch size in config
SOLVER:
  IMS_PER_BATCH: 2  # Reduce from 4 to 2

# Or reduce image size
INPUT:
  MIN_SIZE_TRAIN: (512, 544, 576)  # Smaller scales
```

**2. Dataset not found**
```bash
# Verify COCO conversion completed
ls outputs/coco_annotations/
# Should see: train_coco.json, valid_coco.json, test_coco.json

# Check paths match
python -c "import json; print(json.load(open('outputs/coco_annotations/test_coco.json'))['images'][0])"
```

**3. Import errors**
```bash
# Verify Detectron2 installation
python -c "import detectron2; print(detectron2.__version__)"

# Reinstall if necessary
pip uninstall detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**4. Low mAP on custom data**
- Ensure annotations are correct (run `yolo_to_coco.py` with `--visualize` flag)
- Check class distribution (use `scripts/data_analysis.py` if available)
- Verify image quality and resolution
- Consider longer training (increase `MAX_ITER`)

---

## Configuration Tuning

### Adjust Hyperparameters

Edit `configs/faster_rcnn_R50_FPN.yaml`:

```yaml
SOLVER:
  IMS_PER_BATCH: 4          # Batch size (reduce if OOM)
  BASE_LR: 0.001            # Learning rate (lower for fine-tuning)
  MAX_ITER: 25000           # Training iterations (increase for better results)
  STEPS: (15000, 20000)     # LR decay steps
  CHECKPOINT_PERIOD: 2500   # Save checkpoint frequency

INPUT:
  MIN_SIZE_TRAIN: (512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896)
  MIN_SIZE_TEST: 704        # Test image size
  MAX_SIZE_TRAIN: 1333
  MAX_SIZE_TEST: 1333

TEST:
  EVAL_PERIOD: 2500         # Validation frequency
  DETECTIONS_PER_IMAGE: 100 # Max detections per image
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{agropest12-faster-rcnn-2025,
  title={Insect Detection and Classification using Faster R-CNN: A Comparative Study},
  author={Your Team},
  booktitle={COMP9517 Group Project},
  year={2025}
}
```

---

## License

This project is part of COMP9517 coursework. For academic and educational use only.

---

## Acknowledgments

- Detectron2 framework by Facebook AI Research
- AgroPest-12 dataset from Kaggle
- PyTorch and torchvision libraries
- AutoDL cloud platform for GPU resources

---

## Contact & Support

For questions or issues:
1. Check existing issues in GitHub repository
2. Review `TRAINING_RECORD.md` for experimental details
3. Consult Detectron2 documentation: https://detectron2.readthedocs.io/

---

**Last Updated**: November 20, 2024
**Version**: 2.0
**Status**: Production-ready (Run 1 configuration)
