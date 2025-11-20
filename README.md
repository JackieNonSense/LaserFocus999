# COMP9517 Group Project - Insect Detection and Classification

AgroPest-12 dataset insect detection and classification using multiple computer vision methods.

## Project Structure

```
LaserFocus999/
├── data/AgroPest-12/       # Dataset (not in git)
│   ├── train/              # Training set (11,502 images)
│   │   ├── images/
│   │   └── labels/         # YOLO format (.txt)
│   ├── valid/              # Validation set (1,095 images)
│   └── test/               # Test set (546 images) - ONLY for final evaluation
│
├── classical_sift/         # Classical ML: SIFT + BoW + SVM
│   ├── src/                # Feature extraction and classification
│   ├── configs/            # Configuration files
│   ├── classical-results/  # Evaluation results
│   └── README.md           # Detailed documentation
│
├── faster-rcnn/            # Faster R-CNN (Detectron2)
│   ├── configs/            # Model configurations (baseline + improvements)
│   ├── scripts/            # Training, evaluation, visualization scripts
│   ├── src/                # Dataset registration, custom models
│   ├── faster-rcnn-results/ # Test set evaluation results
│   ├── TRAINING_RECORD.md  # Detailed experimental logs
│   └── README.md           # Complete documentation
│
├── yolov8s/                # YOLOv8s implementation
│   ├── ass1.ipynb          # Training and evaluation notebook
│   ├── data.yaml           # Dataset configuration
│   ├── yolov8s_val_summary.csv        # Validation metrics
│   └── yolov8s_val_per_class_AP.csv   # Per-class AP results
│
├── yolo11s/                # YOLOv11s implementation (latest YOLO)
│   ├── 9517_project_yolo11s.ipynb     # Training and evaluation notebook
│   ├── data.yaml           # Dataset configuration
│   └── results.csv         # Evaluation results
│
├── yolo/                   # Original YOLO experiments
│
├── comparison/             # Cross-method evaluation and comparison
│   ├── scripts/            # Unified evaluation scripts
│   └── results/            # Comparative results from all methods
│
├── utils/                  # Shared utility functions
├── docs/                   # Documentation and guides
├── figures/                # Figures and visualizations for report
└── detectron2/             # Detectron2 library (dependency)
```

## Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd LaserFocus999
```

### 2. Download Dataset

From Kaggle: https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

Using Kaggle CLI:

```bash
pip install kaggle
# Setup kaggle.json in ~/.kaggle/
kaggle datasets download -d rupankarmajumdar/crop-pests-dataset --unzip
```

Place extracted files in `data/AgroPest-12/`

### 3. Create Your Branch

```bash
git checkout -b YourName/MethodName
```

## Important Notes

### Dataset

- **Format**: YOLO format (images + labels)
- **Splits**: Pre-defined train/valid/test - **DO NOT modify or create your own**
- **YOLO team**: Use directly without conversion
- **Other methods**: Convert to required format (e.g., COCO for Detectron2)

### Git Workflow

```bash
# Work on your branch
git add .
git commit -m "Your commit message"
git push origin YourName/MethodName

# Create Pull Request to merge into main
```

### What NOT to Commit

Already configured in `.gitignore`:

- Dataset files (`data/`, images)
- Trained models (`*.pth`, `*.h5`, `models/`)
- Results and outputs (`results/`, `output/`)
- API keys (`.env`, `.claude/`)

## Required Evaluation Metrics

All methods must report on the **test set**:

**Detection**:

- Mean Average Precision (mAP)
- mAP@0.5, mAP@0.75

**Classification**:

- Precision, Recall, F1 Score (per-class and average)
- Accuracy
- AUC

**Efficiency**:

- Training time
- Inference time (FPS)
- GPU memory usage

## Implemented Methods

This project compares four different computer vision approaches for insect detection and classification:

### 1. Classical ML: SIFT + BoW + SVM

**Location**: `classical_sift/`

**Approach**:
- Feature extraction using SIFT (Scale-Invariant Feature Transform)
- Bag-of-Words (BoW) feature encoding with K-means clustering
- SVM (Support Vector Machine) classifier for classification
- Selective Search for region proposal (detection)

**Key Features**:
- No deep learning, handcrafted features
- Interpretable feature representations
- Lower computational requirements

**Documentation**: See `classical_sift/README.md`

### 2. Faster R-CNN (Detectron2)

**Location**: `faster-rcnn/`

**Approach**:
- Two-stage object detector with Region Proposal Network (RPN)
- ResNet-50-FPN backbone for feature extraction
- Pre-trained on COCO dataset, fine-tuned on AgroPest-12
- Multiple experimental configurations tested

**Best Model**: Cascade RCNN + GIoU loss (43.86% mAP)

**Key Features**:
- Multi-scale anchors for various insect sizes
- Test-time augmentation (TTA)
- Grad-CAM visualization for model interpretability

**Dataset Format**: COCO JSON (converted from YOLO)
- Conversion script: `faster-rcnn/scripts/yolo_to_coco.py`

**Documentation**: See `faster-rcnn/README.md` and `faster-rcnn/TRAINING_RECORD.md`

### 3. YOLOv8s

**Location**: `yolov8s/`

**Approach**:
- Single-stage real-time object detector
- YOLOv8 small variant optimized for speed-accuracy tradeoff
- End-to-end training on AgroPest-12

**Key Features**:
- Native YOLO format support (no conversion needed)
- Fast inference speed
- Jupyter notebook-based workflow

**Files**:
- Training notebook: `ass1.ipynb`
- Results: `yolov8s_val_summary.csv`, `yolov8s_val_per_class_AP.csv`

### 4. YOLOv11s

**Location**: `yolo11s/`

**Approach**:
- Latest YOLO architecture (November 2024 release)
- Improved feature extraction and detection head
- State-of-the-art single-stage detector

**Key Features**:
- Enhanced performance over YOLOv8
- Native YOLO format support
- Modern architecture improvements

**Files**:
- Training notebook: `9517_project_yolo11s.ipynb`
- Results: `results.csv`

## Method-Specific Notes

### Classical SIFT-BoW-SVM

- Detection uses Selective Search instead of bounding-box supervised training
- Feature vocabulary size: configurable K-means clusters
- See `classical_sift/README.md` for configuration details

### YOLO Methods (YOLOv8s, YOLOv11s)

- Dataset already in YOLO format - use directly
- No conversion needed
- Training via Jupyter notebooks
- Results exported to CSV format

### Faster R-CNN / Detectron2

- Requires YOLO → COCO format conversion
- Conversion script: `faster-rcnn/scripts/yolo_to_coco.py`
- Supports command-line training and evaluation
- Multiple experimental configurations available

## Deliverables

1. **Video**: Max 10 min, MP4, <100MB
2. **Report**: Max 10 pages, IEEE format, PDF, <10MB
3. **Code**: ZIP, <25MB (no models/data)

## Method Comparison Summary

| Method | Type | Architecture | Dataset Format | Training Time | Inference Speed | Best mAP | Key Advantage |
|--------|------|--------------|----------------|---------------|-----------------|----------|---------------|
| **SIFT-BoW-SVM** | Classical ML | Handcrafted features + SVM | YOLO (direct) | Fast | Fast | TBD | Interpretable, no GPU needed |
| **Faster R-CNN** | Two-stage DL | ResNet-50-FPN | COCO (converted) | ~50 min | Moderate | **43.86%** | Best accuracy, Cascade+GIoU |
| **YOLOv8s** | Single-stage DL | YOLOv8 small | YOLO (direct) | Moderate | Fast | TBD | Speed-accuracy balance |
| **YOLOv11s** | Single-stage DL | YOLOv11 small | YOLO (direct) | Moderate | Fastest | TBD | Latest architecture, real-time |

**Performance Notes**:
- Faster R-CNN achieves highest mAP (43.86%) with Cascade RCNN + GIoU configuration
- YOLO methods prioritize inference speed for real-time applications
- Classical SIFT-BoW-SVM provides baseline and interpretable features
- All methods evaluated on same AgroPest-12 test set (546 images, 12 classes)

**Recommendations**:
- **Best Accuracy**: Faster R-CNN (improve3 configuration)
- **Best Speed**: YOLOv11s
- **Best Interpretability**: SIFT-BoW-SVM
- **Best Balance**: YOLOv8s

## Resources

- **Dataset**: https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset
- **Detectron2**: https://github.com/facebookresearch/detectron2
- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **YOLOv11**: https://docs.ultralytics.com/models/yolo11/
