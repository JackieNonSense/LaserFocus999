# COMP9517 Group Project - Insect Detection and Classification

AgroPest-12 dataset insect detection and classification using multiple computer vision methods.

## Project Structure

```
LaserFocus999/
├── data/AgroPest-12/       # Dataset (not in git)
│   ├── train/              # Training set
│   │   ├── images/
│   │   └── labels/         # YOLO format (.txt)
│   ├── valid/              # Validation set
│   └── test/               # Test set (ONLY for final evaluation)
├── faster-rcnn/            # Faster R-CNN implementation
├── yolo/                   # YOLO implementation
├── comparison/             # Shared evaluation and comparison
│   ├── scripts/            # Unified evaluation scripts
│   └── results/            # Results from all methods
└── utils/                  # Shared utility functions
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

## Method-Specific Notes

### YOLO
- Dataset already in YOLO format - use directly
- No conversion needed

### Faster R-CNN / Detectron2
- Need to convert YOLO → COCO format
- Conversion script: `faster-rcnn/scripts/yolo_to_coco.py`

### Other Methods
- Implement format conversion in your method directory if needed

## Deliverables

1. **Video**: Max 10 min, MP4, <100MB
2. **Report**: Max 10 pages, IEEE format, PDF, <10MB
3. **Code**: ZIP, <25MB (no models/data)

## Resources

- **Dataset**: https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

