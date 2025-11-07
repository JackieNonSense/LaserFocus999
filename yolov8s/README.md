# YOLOv8-s for AgroPest-12 Insect Detection and Classification

YOLOv8-s implementation using **Ultralytics** for insect detection and classification on the AgroPest-12 dataset.

---

## Directory Structure
yolov8s/
├── ass1.ipynb # Full training/validation/prediction pipeline
├── data.yaml # Dataset config (relative paths)
├── yolov8s_val_summary.csv # Overall validation metrics (Precision, Recall, mAP)
└── yolov8s_val_per_class_AP.csv # Per-class AP metrics



---

## Setup

### 1. Environment Setup

**Option A – Using pip (Recommended):**
```bash
pip install ultralytics numpy pandas pillow matplotlib
Option B – Using conda:


conda create -n yolov8 python=3.10 -y
conda activate yolov8
pip install ultralytics numpy pandas pillow matplotlib
2. Install PyTorch
Install PyTorch according to your CUDA version (or CPU-only if you don’t have a GPU):

GPU (CUDA 12.1):


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
CPU-only:


pip install torch torchvision torchaudio
Verify installation:


python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
3. Prepare Dataset
Download and extract the AgroPest-12 dataset into the root directory:


LaserFocus999/
└── data/
    └── AgroPest-12/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
data.yaml already uses relative paths such as ../data/AgroPest-12/train/images,
so no modification is required if you follow this folder structure.

Usage
Step 1 – Train and Validate
Open yolov8s/ass1.ipynb and run all cells in order.

Or, train directly from the command line:


yolo detect train data=yolov8s/data.yaml model=yolov8s.pt imgsz=640 epochs=50 batch=16 name=yolov8s_results
Step 2 – Evaluate
The notebook automatically saves validation metrics:

yolov8s_val_summary.csv → overall Precision, Recall, mAP@50, mAP@50-95

yolov8s_val_per_class_AP.csv → per-class AP metrics

Step 3 – Visualize Predictions
Predicted images with bounding boxes and labels are saved in:


runs/detect/predict/
Expected Results (Validation)
Metric	Value
Precision	0.833
Recall	0.699
mAP@50	0.7429
mAP@50-95	0.429

Configuration
Parameter	Description	Default
imgsz	Input image size	640
epochs	Training epochs	50
batch	Batch size	16
model	YOLOv8 variant	yolov8s.pt
seed	Random seed for reproducibility	42

Reproducibility
Notebook fixed with SEED = 42

No datasets, weights, or runs/ directories are committed (see .gitignore)

Relative dataset paths ensure consistent execution on any system

Notes
Works with or without GPU (CPU training is slower but fully functional)

Automatically downloads pretrained yolov8s.pt weights if missing

Produces evaluation CSVs for cross-model comparison (e.g., Faster R-CNN, SIFT)

Outputs
kotlin

runs/
├── detect/
│   ├── train/
│   ├── val/
│   └── predict/
└── yolov8s_val_summary.csv
Troubleshooting
Issue	Solution
CUDA out of memory	Reduce batch or imgsz in training command
Dataset not found	Verify paths in data.yaml
Import errors	Re-install Ultralytics and Torch: pip install ultralytics torch torchvision

Next Steps
Fine-tune larger variants (YOLOv8-m / YOLOv8-l) for improved accuracy.

Add confusion-matrix and PR-curve visualizations.

Compare metrics with Faster R-CNN and Classical SIFT baselines.
