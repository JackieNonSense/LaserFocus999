# COMP9517 Group Project - Insect Detection and Classification

AgroPest-12 dataset insect detection and classification using multiple computer vision methods.

## Project Overview

**Goal**: Develop and compare different computer vision methods for detecting and classifying insects in agricultural environments.

**Dataset**: AgroPest-12 (12 insect classes)
- 11,502 training images
- 1,095 validation images
- 546 test images

## Team Structure

Each team member develops their method in a separate directory:

```
LaserFocus999/
├── data/                    # Shared dataset (not in git)
│   ├── AgroPest-12/        # Raw dataset
│   └── splits/             # IMPORTANT: Unified train/val/test splits
├── faster-rcnn/            # Faster R-CNN implementation
├── yolo/                   # YOLO implementation
├── comparison/             # Shared comparison and evaluation
├── utils/                  # Shared utility functions
└── docs/                   # Shared documentation
```

## Important: Data Splits

**CRITICAL**: All team members MUST use the same data splits to ensure fair comparison.

The unified splits are located in `data/splits/`:
- `train.txt` - Training set file list
- `val.txt` - Validation set file list
- `test.txt` - Test set file list

DO NOT create your own splits. Use these files to load data.

## Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd LaserFocus999
```

### 2. Download Dataset
Download AgroPest-12 from Kaggle:
https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

Extract to `data/AgroPest-12/`

### 3. Create Your Method Directory
```bash
mkdir your-method-name/
cd your-method-name/
# Add your implementation here
```

### 4. Use Shared Data Splits
When loading data, always refer to `../data/splits/train.txt`, `val.txt`, `test.txt`

## Required Evaluation Metrics

All methods must report the following metrics on the test set:

**Detection Performance**:
- Mean Average Precision (mAP)
- mAP@0.5
- mAP@0.75

**Classification Performance**:
- Precision (per-class and average)
- Recall (per-class and average)
- F1 Score (per-class and average)
- Accuracy
- AUC

**Efficiency**:
- Training time
- Inference time (FPS)
- GPU memory usage

## Workflow

1. **Development**: Work in your own method directory
2. **Testing**: Use shared evaluation scripts in `comparison/scripts/`
3. **Results**: Save results to `comparison/results/<your-method>_results.json`
4. **Comparison**: Run comparison notebook to generate final analysis

## Git Workflow

### Create Your Feature Branch
```bash
git checkout -b YourName/MethodName
```

### Work on Your Branch
```bash
# Make changes
git add .
git commit -m "Add feature X"
git push origin YourName/MethodName
```

### Merge to Main
Create a Pull Request for team review before merging.

## What NOT to Commit

The `.gitignore` is configured to exclude:
- Dataset files (`data/`, `*.zip`, images)
- Trained models (`*.pth`, `*.h5`, `models/`)
- Results (`results/`, `output/`)
- API keys and configs (`.env`, `.claude/`)

Only commit:
- Source code
- Configuration files
- Documentation
- Small figures for reports

## Deliverables

1. **Video Presentation** (max 10 min, <100MB MP4)
2. **Written Report** (max 10 pages, IEEE format, <10MB PDF)
3. **Source Code** (ZIP, <25MB, no models/data)

## Resources

- **Dataset**: https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset
- **Project Spec**: [Link to course website]
- **Team Meeting Notes**: `docs/meeting_notes.md`

## Contact

For questions, contact team members or consult during tutorial hours.
