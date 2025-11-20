# COMP9517 Project – YOLOv8s Code

This folder is only the **YOLOv8s** branch of our group project and is not the full submission.

This folder contains the code and configuration for the **YOLOv8s** branch of our group project *“Insect Detection and Classification: A Comparative Study of Machine Learning and Deep Learning Approaches”*.

The notebook reproduces the experiments reported in the YOLOv8s sections of the report.

---

## 1. Folder Structure

```text
.
├── archive/
│   ├── data.yaml                  # YOLO dataset config (train / val / test)
│   ├── knowledge.json             # Notes about the cleaned AgroPest-12 dataset
│   ├── train/                     # Training images and labels (YOLO format) *
│   ├── valid/                     # Validation images and labels *
│   └── test/                      # Test images and labels *
│
├── ass1ver2.ipynb                 # Main notebook for YOLOv8s
├── yolov8s_val_summary.csv        # Overall validation metrics (mAP, Precision, Recall)
└── yolov8s_val_per_class_AP.csv   # Per-class AP on the validation set
```

\* The `train/`, `valid/`, and `test/` folders follow the standard YOLO structure:

```
train/
    images/
    labels/

valid/
    images/
    labels/

test/
    images/
    labels/
```

---

## 2. Environment

The code was tested with:

- Python 3.10  
- CUDA GPU (NVIDIA RTX 4060 Laptop)

Main Python packages:

- torch, torchvision  
- ultralytics  
- numpy  
- pandas  
- matplotlib  
- IPython  

Minimal installation:

```bash
pip install torch torchvision ultralytics numpy pandas matplotlib ipykernel
```

---

## 3. How to Run the Notebook

1. Open **ass1ver2.ipynb** in Jupyter Notebook, JupyterLab, or VS Code.  
2. Make sure the working directory is the project root (the folder containing `archive/` and `ass1ver2.ipynb`).  
3. The notebook assumes:
   - dataset config at `archive/data.yaml`
   - images/labels inside `archive/train`, `archive/valid`, `archive/test`

### Execution Steps

#### ✔ Reproducibility & Imports
- Sets random seed  
- Checks CUDA  
- Imports YOLO from ultralytics  

#### ✔ Load Base YOLOv8s Model
Loads `yolov8s.pt` (COCO pretrained).

#### ✔ Training
Uses:

```
data="archive/data.yaml"
epochs=50
imgsz=960
batch=8
device=0
project="AgroPest"
name="YOLOv8s"
patience=50
plots=True
```

> On a laptop GPU: approx. **1–2 hours**.  
> For marking: **training does not need to be re-run**; saved outputs are already in the notebook.

#### ✔ Display Training Curves
Shows losses, precision, recall, mAP.

#### ✔ Load Best Weights & Validate
Saves:

- `yolov8s_val_summary.csv`
- `yolov8s_val_per_class_AP.csv`

#### ✔ Generate Predictions
Runs:

```
model.predict(source="archive/valid/images", save=True)
```

Outputs saved in `runs/detect/val/`.

---

## 4. Notes for Markers

- The notebook has been fully executed once.  
- All main figures (training curves, confusion matrix, metric tables) are **embedded inside the notebook**.  
- Re-running heavy cells is **not required**.  
- Light cells (visualisation, printing metrics) can be re-run for verification.  
- Code is robust to minor updates in the ultralytics package.
