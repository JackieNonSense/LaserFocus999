YOLOv8-s for AgroPest-12 Insect Detection and Classification

YOLOv8-s implementation using Ultralytics for insect detection + classification on the AgroPest-12 dataset.
This folder contains a reproducible notebook, the dataset config, and validation metrics.

Folder Contents

ass1.ipynb — full training / validation / prediction pipeline (GPU compatible)

data.yaml — dataset configuration (uses relative paths to dataset)

yolov8s_val_summary.csv — overall validation metrics (mAP50-95 / mAP50 / P / R)

yolov8s_val_per_class_AP.csv — per-class AP metrics

1) Environment Setup
   Option A — Using pip (Recommended)
   # core libs
pip install ultralytics numpy pandas pillow matplotlib

# (optional) if you prefer a fresh env with conda
# conda create -n yolov8 python=3.10 -y
# conda activate yolov8
# pip install ultralytics numpy pandas pillow matplotlib

Install PyTorch

Choose the command that matches your CUDA version, or CPU-only:
# Example for CUDA 12.1  (adjust if you have a different CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only
# pip install torch torchvision torchaudio

2) Prepare Dataset

Place the Kaggle AgroPest-12 dataset under the repo root using the following layout (YOLO format):
LaserFocus999/
└─ data/
   └─ AgroPest-12/
      ├─ train/
      │  ├─ images/
      │  └─ labels/
      ├─ valid/
      │  ├─ images/
      │  └─ labels/
      └─ test/
         ├─ images/
         └─ labels/
