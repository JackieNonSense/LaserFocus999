YOLOv8-s for AgroPest-12 Insect Detection and Classification

YOLOv8-s implementation using Ultralytics for insect detection + classification on the AgroPest-12 dataset.
This folder contains a reproducible notebook, the dataset config, and validation metrics.

Folder Contents

ass1.ipynb — Full training / validation / prediction pipeline (GPU compatible)

data.yaml — Dataset configuration (uses relative paths)

yolov8s_val_summary.csv — Overall validation metrics (mAP50-95 / mAP50 / P / R)

yolov8s_val_per_class_AP.csv — Per-class AP metrics

1️⃣ Environment Setup
Option A — Using pip (Recommended)
# Core libs
pip install ultralytics numpy pandas pillow matplotlib

Option B — Using conda
conda create -n yolov8 python=3.10 -y
conda activate yolov8
pip install ultralytics numpy pandas pillow matplotlib

### Install PyTorch

Choose the command matching your CUDA version, or CPU-only:


# CUDA 12.1 (adjust if you have a different CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only
pip install torch torchvision torchaudio

Verify Installation

import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
If you see “CUDA available: True” and a GPU name like RTX 4060, your GPU is working.

2️⃣ Dataset Structure
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

Place the AgroPest-12 dataset in this structure:
Your data.yaml should look like this:
train: ../data/AgroPest-12/train/images
val:   ../data/AgroPest-12/valid/images
test:  ../data/AgroPest-12/test/images
nc: 12
names:
  0: Ants
  1: Bees
  2: Beetles
  3: Caterpillars
  4: Earthworms
  5: Earwigs
  6: Grasshoppers
  7: Moths
  8: Slugs
  9: Snails
  10: Wasps
  11: Weevils

✅ No modification needed if you follow this layout.

3️⃣ How to Run
Option A — Notebook

Open yolov8s/ass1.ipynb

Run all cells sequentially

GPU will be used automatically if available

Outputs:

Trained model checkpoints (runs/detect/YOLOv8s/)

Validation results (yolov8s_val_summary.csv)

Predictions (runs/detect/predict/)

Option B — Command Line
# Train model
yolo detect train \
  data=yolov8s/data.yaml \
  model=yolov8s.pt \
  imgsz=960 epochs=50 batch=8 \
  project=AgroPest name=YOLOv8s

# Validate
yolo detect val \
  data=yolov8s/data.yaml \
  model=runs/detect/YOLOv8s/weights/best.pt

# Predict
yolo predict \
  model=runs/detect/YOLOv8s/weights/best.pt \
  source=data/AgroPest-12/valid/images \
  save=True

4️⃣ Validation Results (ours)
Metric	Value
Precision	0.833
Recall	0.699
mAP@50	0.743
mAP@50–95	0.429

📄 Files:

yolov8s_val_summary.csv → overall precision, recall, mAP

yolov8s_val_per_class_AP.csv → per-class AP

🖼️ Predictions saved in runs/detect/predict/.

YOLOv8 is a single-stage detector, meaning it performs detection + classification simultaneously, meeting both assignment requirements.

5️⃣ Reproducibility

Fixed random seed: SEED=42

GPU: NVIDIA RTX 4060 Laptop GPU

No dataset / weights committed (see .gitignore)

All paths use relative references

6️⃣ Troubleshooting
Problem	Fix
CUDA out of memory	Reduce batch (e.g. batch=4) or image size (imgsz=640)
No GPU found	Reinstall PyTorch with CUDA support
Empty predictions	Check that your labels match image filenames
7️⃣ Credits

Ultralytics YOLOv8

AgroPest-12 dataset (Kaggle)

Maintainer: Chacha — YOLOv8 Method Lead
For questions, please reach out via the GitHub project discussion.
